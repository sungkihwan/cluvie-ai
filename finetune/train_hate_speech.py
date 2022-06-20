import os
import pandas as pd

from pprint import pprint

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingWarmRestarts

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning import LightningModule, LightningDataModule, Trainer, seed_everything, loggers as pl_loggers
from torchmetrics.functional import accuracy, f1_score, auroc, recall, precision

from transformers import ElectraModel, ElectraTokenizer, AdamW, get_linear_schedule_with_warmup

from sklearn.metrics import classification_report, multilabel_confusion_matrix

import re
import emoji
from soynlp.normalizer import repeat_normalize

args = {
    'random_seed': 42,  # Random Seed
    'output_dir': 'checkpoints',
    'model_name_or_path': "monologg/koelectra-base-v3-discriminator",
    'task_name': '',
    'filename': 'model_chp/epoch{epoch}-val_acc{val_acc:.4f}',
    'doc_col': 'comment',
    'batch_size': 32,
    'linear_layer_size': 515,
    'dropout_rate': 0.5,
    'lr': 3e-5,  # Learning Rate
    'max_epochs': 20,  # Max Epochs
    'max_length': 128,  # Max Length input size
    'train_data_path': "data/hate-speech/train.tsv",  # Train Dataset file
    'val_data_path': "data/hate-speech/val.tsv",  # Validation Dataset file
    'test_mode': False,  # Test Mode enables `fast_dev_run`
    'optimizer': 'AdamW',
    'lr_scheduler': 'exp',  # ExponentialLR(exp), CosineAnnealingWarmRestarts(cos), get_linear_schedule_with_warmup(warmup)
    'lr_parameter': 0.5,
    'fp16': False,  # Enable train on FP16
    'tpu_cores': 0,  # Enable TPU with 1 core or 8 cores
    'num_workers': 4,  # Multi thread
}

class HateSpeechClassification(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()  # self.hparams 저장

        self.model = ElectraModel.from_pretrained(self.hparams.model_name_or_path)

        self.classifier = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, self.hparams.linear_layer_size),
            nn.GELU(),
            nn.Dropout(self.hparams.dropout_rate),
            nn.Linear(self.hparams.linear_layer_size, self.hparams.label_size),
        )

        self.criterion = nn.BCELoss()

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        output = self.classifier(output.last_hidden_state[:, 0]) # bertmodel -> self.classifier(output.pooler_output)
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
        labels = batch["labels"]

        loss, output = self(input_ids, attention_mask, labels)

        self.log(state + "_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": output, "labels": labels}

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "test")

    def epoch_end(self, outputs, state='train'):
        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"].detach().cpu():
                predictions.append(out_predictions)

        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)

        for i, name in enumerate(self.hparams.label_columns):
            class_roc_auc = auroc(predictions[:, i], labels[:, i])
            self.log(f"{state}_roc_auc", class_roc_auc, on_epoch=True, prog_bar=True)

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
        elif self.hparams.lr_scheduler == "warmup":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.n_warmup_steps,
                num_training_steps=self.hparams.n_training_steps
            )
        else:
            raise NotImplementedError('Only cos, exp, warmup lr scheduler is Supported!')

        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                interval='step'
            )
        )

class HateSpeechDataset(Dataset):
    def __init__(
            self,
            data: pd.DataFrame,
            model_name_or_path,
            label_columns,
            max_length: int = 128,
            doc_col="comment"
    ):
        self.data = data
        self.tokenizer = ElectraTokenizer.from_pretrained(model_name_or_path)
        self.label_columns = label_columns
        self.max_length = max_length
        self.doc_col = doc_col

    # 한국어 데이터 전처리
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]    # 행 데이터 가져오기

        comment = self.clean_text(data_row[self.doc_col])   # 한국어 데이터 전처리후 글자 데이터 가져오기
        labels = data_row[self.label_columns]   # 레이블 데이터 가져오기

        encoding = self.tokenizer.encode_plus(
            comment,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return dict(
            comment=comment,
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            labels=torch.FloatTensor(labels)
        )

class HateSpeechDataModule(LightningDataModule):
    def __init__(self, batch_size, train_data, val_data, max_length,
                 doc_col, model_name_or_path, label_columns, num_workers=1):
        super().__init__()
        self.batch_size = batch_size
        self.train_data = train_data
        self.val_data = val_data
        self.max_length = max_length
        self.doc_col = doc_col
        self.model_name_or_path = model_name_or_path
        self.label_columns = label_columns
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = HateSpeechDataset(
            self.train_data, model_name_or_path=self.model_name_or_path, label_columns=self.label_columns,
            max_length=self.max_length, doc_col=self.doc_col
        )
        self.val_dataset = HateSpeechDataset(
            self.val_data, model_name_or_path=self.model_name_or_path, label_columns=self.label_columns,
            max_length=self.max_length, doc_col=self.doc_col
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


# 파일 읽기
def read_data(path):
    if path.endswith('xlsx'):
        return pd.read_excel(path)
    elif path.endswith('csv'):
        return pd.read_csv(path)
    elif path.endswith('tsv') or path.endswith('txt'):
        return pd.read_csv(path, sep='\t')
    else:
        raise NotImplementedError('Only Excel(xlsx)/Csv/Tsv(txt) are Supported')

def hate_speech_preprocessor(df, doc_col):
    # nan 제거
    df = df.dropna(axis=0)
    # 중복제거
    df.drop_duplicates(subset=[doc_col], inplace=True)

    df['offensive'] = df['hate'].map(lambda x: 1 if x == "offensive" else 0)
    df['hate'] = df['hate'].map(lambda x: 1 if x == "hate" else 0)
    df['others'] = df['bias'].map(lambda x: 1 if x == "others" else 0)
    df['gender'] = df['bias'].map(lambda x: 1 if x == "gender" else 0)
    df.drop(['bias'], axis=1, inplace=True)

    return df

def main():
    print("Using PyTorch Ver", torch.__version__)
    seed_everything(args['random_seed'])

    train_data = hate_speech_preprocessor(read_data(args['train_data_path']), args['doc_col'])
    val_data = hate_speech_preprocessor(read_data(args['val_data_path']), args['doc_col'])

    args['label_columns'] = val_data.columns.tolist()[2:]
    args['label_size'] = len(args['label_columns'])

    args['n_training_steps'] = len(train_data) // args['batch_size'] * args['max_epochs']
    args['n_warmup_steps'] = args['n_training_steps'] // 5

    print("args: ", args)

    model = HateSpeechClassification(**args)
    dm = HateSpeechDataModule(
        batch_size=args['batch_size'], train_data=train_data, val_data=val_data, max_length=args['max_length'],
        doc_col=args['doc_col'], model_name_or_path=args['model_name_or_path'],
        num_workers=args['num_workers'], label_columns=args['label_columns'],
    )

    gpus = max(1, torch.cuda.device_count())

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=args['output_dir'],
        filename=args['filename'],
        verbose=True,
        save_last=False,
        mode='min',
        save_top_k=3
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=3,
        strict=False,
        verbose=False,
        mode='min'
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