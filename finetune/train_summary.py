import argparse
import logging
import os
import pytorch_lightning as pl
import torch
import yaml

from pytorch_lightning import loggers as pl_loggers, seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from dataset import KobartSummaryModule
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup

logger = logging.getLogger()
logger.setLevel(logging.INFO)

class Base(pl.LightningModule):
    def __init__(self, hparams, trainer, **kwargs) -> None:
        super(Base, self).__init__()
        self.save_hyperparameters(hparams)
        self.trainer = trainer
    
    def setup_steps(self, stage=None):
        # pip install pl>=1.5.2
        train_loader = self.trainer._data_connector._train_dataloader_source.dataloader()

        return len(train_loader)

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr, correct_bias=False)
        num_workers = self.hparams.num_workers

        data_len = self.setup_steps(self)
        logging.info(f'number of workers {num_workers}, summary length {data_len}')
        num_train_steps = int(data_len / (self.hparams.batch_size * num_workers) * self.hparams.max_epochs)
        logging.info(f'num_train_steps : {num_train_steps}')
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        logging.info(f'num_warmup_steps : {num_warmup_steps}')
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler, 
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]


class KoBARTConditionalGeneration(Base):
    def __init__(self, hparams, trainer=None, **kwargs):
        super(KoBARTConditionalGeneration, self).__init__(hparams, trainer, **kwargs)
        self.model = BartForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
        self.model.train()
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(hparams.model_name_or_path)
        self.pad_token_id = self.tokenizer.pad_token_id

    def forward(self, inputs):

        attention_mask = inputs['input_ids'].ne(self.pad_token_id).float()
        decoder_attention_mask = inputs['decoder_input_ids'].ne(self.pad_token_id).float()
        
        return self.model(input_ids=inputs['input_ids'],
                          attention_mask=attention_mask,
                          decoder_input_ids=inputs['decoder_input_ids'],
                          decoder_attention_mask=decoder_attention_mask,
                          labels=inputs['labels'], return_dict=True)


    def training_step(self, batch, batch_idx):
        outs = self(batch)
        loss = outs.loss
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outs = self(batch)
        loss = outs['loss']
        return (loss)

    def validation_epoch_end(self, outputs):
        losses = []
        for loss in outputs:
            losses.append(loss)
        self.log('val_loss', torch.stack(losses).mean(), prog_bar=True)

class MakeBin():
    @staticmethod
    def save(model_binary_path=None, hparams_path=None):
        if hparams_path == None:
            hparams_path = args.hparams
        if model_binary_path == None:
            model_binary_path = args.model_binary

        with open(hparams_path) as file:
            hparams = yaml.safe_load(file)

        inf = KoBARTConditionalGeneration.load_from_checkpoint(model_binary_path, hparams=hparams)
        inf.model.save_pretrained(args.output_dir)

if __name__ == '__main__':
    seed_everything(42)
    cli_parser = argparse.ArgumentParser()

    cli_parser.add_argument("--task", type=str, default="summary", required=True)
    cli_parser.add_argument("--config_dir", type=str, default="config")
    cli_parser.add_argument("--config_file", type=str, default="kobart-base-v2.yml", required=True)
    cli_args = cli_parser.parse_args()

    with open(os.path.join(cli_args.config_dir, cli_args.task, cli_args.config_file)) as f:
        args = yaml.safe_load(f)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model_name_or_path)
    logging.info(args)

    dm = KobartSummaryModule(
                        args.train_file,
                        args.test_file,
                        tokenizer,
                        batch_size=args.batch_size,
                        max_len=args.max_len,
                        num_workers=args.num_workers)
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss',
                                                       dirpath=args.default_root_dir,
                                                       filename='model_chp/{epoch:02d}-{val_loss:.3f}',
                                                       verbose=True,
                                                       save_last=False,
                                                       mode='min',
                                                       save_top_k=3)

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=3,
        strict=False,
        verbose=False,
        mode='min'
    )

    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(args.default_root_dir, 'tb_logs'))
    lr_logger_callback = pl.callbacks.LearningRateMonitor()
    trainer = pl.Trainer.from_argparse_args(args, logger=tb_logger,
                                            callbacks=[checkpoint_callback, lr_logger_callback, early_stop_callback])

    model = KoBARTConditionalGeneration(args, trainer)
    trainer.fit(model, dm)
    MakeBin.save(os.path.join(args.default_root_dir, 'tb_logs', 'default/version_0/hparams.yaml'), checkpoint_callback.best_model_path)
