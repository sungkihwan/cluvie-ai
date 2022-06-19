import os
import pandas as pd

from pprint import pprint

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingWarmRestarts

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning import LightningModule, Trainer, seed_everything, loggers as pl_loggers

from transformers import ElectraForSequenceClassification, ElectraTokenizer, AdamW

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import re
import emoji
from soynlp.normalizer import repeat_normalize

args = {
    'random_seed': 42,  # Random Seed
    'output_dir': 'ckpt',
    'model_name_or_path': "monologg/koelectra-base-v3-discriminator",
    'task_name': '',
    'batch_size': 32,
    'lr': 3e-5,  # Learning Rate
    'max_epochs': 15,  # Max Epochs
    'max_seq_length': 128,  # Max Length input size
    'train_data_path': "data/hate-speech/train.tsv",  # Train Dataset file
    'val_data_path': "data/hate-speech/val.tsv",  # Validation Dataset file
    'test_mode': False,  # Test Mode enables `fast_dev_run`
    'optimizer': 'AdamW',
    'lr_scheduler': 'exp',  # ExponentialLR vs CosineAnnealingWarmRestarts
    'lr_parameter': 0.5,
    'fp16': True,  # Enable train on FP16
    'tpu_cores': 0,  # Enable TPU with 1 core or 8 cores
    'num_workers': 4,  # Multi thread
}

class Model(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()  # self.hparams 저장

        self.electra = ElectraForSequenceClassification.from_pretrained(self.hparams.model_name_or_path)
        self.tokenizer = ElectraTokenizer.from_pretrained(self.hparams.model_name_or_path)

    def forward(self, **kwargs):
        return self.electra(**kwargs)

    def step(self, batch, batch_idx):
        data, labels = batch
        output = self(input_ids=data, labels=labels)

        loss = output.loss
        logits = output.logits

        preds = logits.argmax(dim=-1)

        y_true = list(labels.cpu().numpy())
        y_pred = list(preds.cpu().numpy())

        return {
            'loss': loss,
            'y_true': y_true,
            'y_pred': y_pred,
        }

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def epoch_end(self, outputs, state='train'):
        loss = torch.tensor(0, dtype=torch.float)
        for i in outputs:
            loss += i['loss'].cpu().detach()
        loss = loss / len(outputs)

        y_true = []
        y_pred = []
        for i in outputs:
            y_true += i['y_true']
            y_pred += i['y_pred']

        self.log(state + '_loss', float(loss), on_epoch=True, prog_bar=True)
        self.log(state + '_acc', accuracy_score(y_true, y_pred), on_epoch=True, prog_bar=True)
        self.log(state + '_precision', precision_score(y_true, y_pred), on_epoch=True, prog_bar=True)
        self.log(state + '_recall', recall_score(y_true, y_pred), on_epoch=True, prog_bar=True)
        self.log(state + '_f1', f1_score(y_true, y_pred), on_epoch=True, prog_bar=True)
        return {'loss': loss}

    def train_epoch_end(self, outputs):
        return self.epoch_end(outputs, state='train')

    def validation_epoch_end(self, outputs):
        return self.epoch_end(outputs, state='val')

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr)

        if self.hparams.lr_scheduler == 'cos':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)
        elif self.hparams.lr_scheduler == 'exp':
            scheduler = ExponentialLR(optimizer, gamma=self.hparams.lr_parameter)
        else:
            raise NotImplementedError('Only cos and exp lr scheduler is Supported!')

        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
            # 'lr_scheduler': {
            #     'scheduler': scheduler,
            #     'monitor': 'val_acc',
            #     'interval': 'step',
            # }
        }

    def read_data(self, path):
        if path.endswith('xlsx'):
            return pd.read_excel(path)
        elif path.endswith('csv'):
            return pd.read_csv(path)
        elif path.endswith('tsv') or path.endswith('txt'):
            return pd.read_csv(path, sep='\t')
        else:
            raise NotImplementedError('Only Excel(xlsx)/Csv/Tsv(txt) are Supported')

    def preprocess_dataframe(self, df):
        emojis = ''.join(emoji.UNICODE_EMOJI.keys())
        pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+')
        url_pattern = re.compile(
            r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')

        # 텍스트 데이터 전처리
        def clean(x):
            x = pattern.sub(' ', x)
            x = url_pattern.sub('', x)
            x = x.strip()
            x = repeat_normalize(x, num_repeats=2)
            return x

        # hate speech label 데이터 전처리
        def string_to_number(x):
            if x == 'hate':
                return 0
            elif x == 'offensive':
                return 1
            else:
                return 2

        # df['comment'] = df['comment'].map(lambda x: self.tokenizer.encode(
        #     clean(str(x)),
        #     padding='max_length',
        #     max_length=self.hparams.max_length,
        #     truncation=True,
        # ))
        # df['hate'] = df['hate'].map(string_to_number)

        encoding = df['comment'].map(lambda x: self.tokenizer.encode_plus(
            clean(str(x)),
            add_special_tokens=True,
            padding='max_length',
            return_token_type_ids=True,
            return_attention_mask=True,
            max_length=self.hparams.max_seq_length,
            truncation=True,
            return_tensors='pt',
        ))

        input_ids_list = []
        token_type_ids_list = []
        attention_mask_list = []
        label_list = df['hate'].map(string_to_number).to_list()

        for feature in encoding:
            input_ids_list.append(feature['input_ids'])
            token_type_ids_list.append(feature['token_type_ids'])
            attention_mask_list.append(feature['attention_mask'])

        return dict(
            input_ids_list=input_ids_list,
            token_type_ids_list=token_type_ids_list,
            attention_mask_list=attention_mask_list,
            label_list=label_list
        )

    def dataloader(self, path, shuffle=False):
        df = self.read_data(path)
        df = self.preprocess_dataframe(df)

        # dataset = TensorDataset(
        #     torch.tensor(df['comment'].to_list(), dtype=torch.long),
        #     torch.tensor(df['label'].to_list(), dtype=torch.long),
        # )

        dataset = TensorDataset(
            torch.tensor(df['input_ids_list'], dtype=torch.long),
            torch.tensor(df['attention_mask_list'], dtype=torch.long),
            torch.tensor(df['token_type_ids_list'], dtype=torch.long),
            torch.tensor(df['label_list'], dtype=torch.long),
        )
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size or self.batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
        )

    def train_dataloader(self):
        return self.dataloader(self.hparams.train_data_path, shuffle=True)

    def val_dataloader(self):
        return self.dataloader(self.hparams.val_data_path, shuffle=False)

def main():
    print("Using PyTorch Ver", torch.__version__)
    print("args: ", args)
    seed_everything(args['random_seed'])
    model = Model(**args)

    gpus = min(1, torch.cuda.device_count())

    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath=args['output_dir'],
        filename='model_chp/epoch{epoch}-val_acc{val_acc:.4f}',
        verbose=True,
        save_last=False,
        mode='max',
        save_top_k=3
    )

    early_stop_callback = EarlyStopping(
        monitor='val_acc',
        patience=3,
        strict=False,
        verbose=False,
        mode='max'
    )

    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(args['output_dir'], 'tb_logs'))
    lr_logger_callback = LearningRateMonitor()

    print(":: Start Training ::")
    trainer = Trainer(
        callbacks=[checkpoint_callback, early_stop_callback, lr_logger_callback],
        max_epochs=args['max_epochs'],
        fast_dev_run=args['test_mode'],
        num_sanity_val_steps=None if args['test_mode'] else 0,
        logger=tb_logger,
        # For GPU Setup
        deterministic=torch.cuda.is_available(),
        gpus=gpus if torch.cuda.is_available() else None,
        precision=16 if args['fp16'] else 32,
        # For TPU Setup
        # tpu_cores=args.tpu_cores if args.tpu_cores else None,
    )
    trainer.fit(model)

main()