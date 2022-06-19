import os
import pandas as pd

from pprint import pprint

import torchmetrics
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingWarmRestarts

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning import LightningModule, LightningDataModule, Trainer, seed_everything, loggers as pl_loggers

from transformers import ElectraForSequenceClassification, ElectraTokenizer, AdamW

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import re
import emoji
from soynlp.normalizer import repeat_normalize

device = torch.device("cuda")

args = {
    'random_seed': 42,  # Random Seed
    'output_dir': 'ckpt',
    'model_name_or_path': "monologg/koelectra-base-v3-discriminator",
    'task_name': '',
    'doc_col': 'comment',
    'label_col': 'hate',
    'linear_layer_size': 515,
    'batch_size': 32,
    'lr': 3e-5,  # Learning Rate
    'max_epochs': 15,  # Max Epochs
    'max_length': 128,  # Max Length input size
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

class ElectraClassification(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()  # self.hparams 저장

        self.electra = ElectraForSequenceClassification.from_pretrained(self.hparams.model_name_or_path)
        self.tokenizer = ElectraTokenizer.from_pretrained(self.hparams.model_name_or_path)
        self.criterion = nn.BCELoss()

        self.classifier = nn.Sequential(
            nn.Linear(self.electra.config.hidden_size, self.hparams.linear_layer_size),
            nn.GELU(),
            nn.Dropout(self.hparams.dropout_rate),
            nn.Linear(self.hparams.linear_layer_size, self.hparams.num_labels),
        )

    def forward(self, input_ids=None, attention_mask=None, labels=None, token_type_ids=None):
        # output = self.electra(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # cls = output[0][:, 0]

        output = self.electra(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        output = self.classifier(output.pooler_outpu)
        output = torch.sigmoid(output)

        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output

    def step(self, batch, batch_idx, state):
        '''
        ##########################################################
        electra forward input shape information
        * input_ids.shape (batch_size, max_length)
        * attention_mask.shape (batch_size, max_length)
        * label.shape (batch_size,)
        ##########################################################
        '''

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        labels = batch["labels"]

        # change label shape (list -> torch.Tensor((batch_size, 1)))

        loss, outputs = self(input_ids, attention_mask, labels)


        if state == "train":
            step_name = "train_loss"
        elif state == "val":
            step_name = "val_loss"
        else:
            step_name = "test_loss"

        self.log(step_name, loss, prog_bar=True, logger=True)

        return {"loss": loss, "predictions": outputs, "labels": labels}

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val")['loss']

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "test")['loss']

    def epoch_end(self, outputs, state='train'):
        loss = torch.tensor(0, dtype=torch.float)
        for i in outputs:
            loss += i['loss'].cpu().detach()
        loss = loss / len(outputs)

        y_true = []
        y_pred = []
        for i in outputs:
            y_true += i['labels']
            y_pred += i['predictions']

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

class ElectraClassificationDataset(Dataset):
    def __init__(self, path, doc_col, label_col, max_length,
                 model_name_or_path, num_workers=1, labels_dict=None):
        self.tokenizer = ElectraTokenizer.from_pretrained(model_name_or_path)

        self.max_length = max_length
        self.doc_col = doc_col
        self.label_col = label_col

        # labels
        # None : label이 num으로 되어 있음
        # dict : label이 num이 아닌 것으로 되어 있음
        # ex : {True : 1, False : 0}
        self.labels_dict = labels_dict

        # dataset
        df = self.read_data(path)
        df[label_col] = df[label_col].map(self.string_to_number)
        # nan 제거
        df = df.dropna(axis=0)
        # 중복제거
        df.drop_duplicates(subset=[self.doc_col], inplace=True)
        self.dataset = df

    def read_data(self, path):
        if path.endswith('xlsx'):
            return pd.read_excel(path)
        elif path.endswith('csv'):
            return pd.read_csv(path)
        elif path.endswith('tsv') or path.endswith('txt'):
            return pd.read_csv(path, sep='\t')
        else:
            raise NotImplementedError('Only Excel(xlsx)/Csv/Tsv(txt) are Supported')

    def __len__(self):
        return len(self.dataset)

    # text 데이터 전처리
    def clean_text(self, text):
        emojis = ''.join(emoji.UNICODE_EMOJI.keys())
        pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+')
        url_pattern = re.compile(
            r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
        )
        processed = pattern.sub(' ', text)
        processed = url_pattern.sub(' ', processed)
        processed = processed.strip()
        processed = repeat_normalize(processed, num_repeats=2)

        return processed

    # hate speech 데이터 전처리
    def string_to_number(self, label):
        if label == 'hate':
            return 0
        elif label == 'offensive':
            return 1
        else:
            return 2

    def __getitem__(self, idx):
        document = self.clean_text(self.dataset[self.doc_col].iloc[idx])
        inputs = self.tokenizer(
            document,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            add_special_tokens=True
        )

        if self.labels_dict:
            labels = self.labels_dict[self.dataset[self.label_col].iloc[idx]]
        else:
            labels = self.dataset[self.label_col].iloc[idx]

        return {
            'input_ids': inputs['input_ids'][0],
            'attention_mask': inputs['attention_mask'][0],
            'token_type_ids': inputs['token_type_ids'][0],
            'labels': labels
        }


class ElectraClassificationDataModule(LightningDataModule):
    def __init__(self, train_data_path, val_data_path, max_length, batch_size,
                 doc_col, label_col, model_name_or_path, num_workers=1, labels_dict=None):
        super().__init__()
        self.batch_size = batch_size
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.max_length = max_length
        self.doc_col = doc_col
        self.label_col = label_col
        self.num_workers = num_workers
        self.model_name_or_path = model_name_or_path
        self.labels_dict = labels_dict

    def setup(self, stage=None):
        self.train = ElectraClassificationDataset(
            self.train_data_path, doc_col=self.doc_col, label_col=self.label_col,
            max_length=self.max_length, labels_dict=self.labels_dict, model_name_or_path=self.model_name_or_path
        )
        self.val = ElectraClassificationDataset(
            self.train_data_path, doc_col=self.doc_col, label_col=self.label_col,
            max_length=self.max_length, labels_dict=self.labels_dict, model_name_or_path=self.model_name_or_path
        )

    def train_dataloader(self):
        train = DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        return train

    def val_dataloader(self):
        val = DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        return val

    def test_dataloader(self):
        test = DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        return test

def main():
    print("Using PyTorch Ver", torch.__version__)
    print("args: ", args)
    seed_everything(args['random_seed'])

    model = ElectraClassification(**args)
    dm = ElectraClassificationDataModule(
        batch_size=args['batch_size'], train_data_path=args['train_data_path'],
        val_data_path=args['val_data_path'], max_length=args['max_length'], doc_col=args['doc_col'],
        label_col=args['label_col'], model_name_or_path=args['model_name_or_path'], num_workers=args['num_workers']
    )

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

    print(":: Start Training ::")
    trainer.fit(model, dm)

main()