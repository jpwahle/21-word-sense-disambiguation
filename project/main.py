import os, argparse, logging, torch
import pytorch_lightning as pl

from transformers import AutoConfig, AutoTokenizer
from data import WSDDataset
from model import GilBERTFinetuner
from helpers import add_generic_args

# TODO: replace with pl args
# from args import ModelArguments, DataArguments

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from pytorch_lightning.loggers.neptune import NeptuneLogger, neptune
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import Callback

class SaveCallback(Callback):
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if trainer.global_rank == 0:
            print("Saving Model")
            pl_module.on_save_checkpoint()

def main():
    parser = argparse.ArgumentParser()
    add_generic_args(parser, os.getcwd())
    parser = GilBERTFinetuner.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()

    if 'MODEL' in os.environ:
        args.model_name_or_path = os.environ['MODEL']

    config = AutoConfig.from_pretrained(
        args.config_name if 'config_name' in args else args.model_name_or_path,
        num_labels=2,
        hidden_dropout_prob=args.hidden_dropout_prob,
        gradient_checkpointing=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if "tokenizer_name" in args else args.model_name_or_path,
        model_max_length=args.max_seq_length,
        use_fast=True
    )
    
    neptune_logger = None
    if args.neptune_logging:
        neptune_logger = NeptuneLogger(
            project_name="jpelhaw/GilBERT2"
        )
        
    train_dataset = WSDDataset(data_dir=args.data_dir, tokenizer=tokenizer, dataset_name="train", mlm=args.mlm, method=args.method)
    val_datasets= [              
        WSDDataset(data_dir=args.data_dir, tokenizer=tokenizer, dataset_name=eval_name, mlm=False, method=args.method) for eval_name in args.eval_names
    ]
    model = GilBERTFinetuner(args, config, tokenizer, train_dataset, val_datasets)
    
    # huggingface transformers uses parallel tokenizers which is a problem when we use parallel TPU/GPU backends because it spawns a large number of total threads
    if args.tpu_num_cores or torch.cuda.is_available() and torch.cuda.device_count() > 1 or args.num_workers > 1:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(args.output_dir, args.model_name_or_path))
    
    train_params = {}
    if args.fp16 and torch.cuda.is_available() or args.tpu_num_cores:
        train_params["precision"] = 16

    if args.gpus > 1:
        train_params["distributed_backend"] = "ddp"

    train_params["accumulate_grad_batches"] = args.accumulate_grad_batches
    train_params["limit_train_batches"] = 1.0 if args.do_train else 0.0
    train_params["limit_val_batches"] = 1.0 if args.do_eval else 0.0
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=args.output_dir, prefix="checkpoint", monitor="val_loss", mode="min", save_top_k=1
    )
    
    trainer = pl.Trainer.from_argparse_args(
        args,
        checkpoint_callback=checkpoint_callback,
        logger=[neptune_logger, tb_logger] if args.neptune_logging and neptune_logger else tb_logger,
        **train_params
    )
        
    trainer.fit(model)
    if args.do_predict:
        trainer.test(model)

if __name__ == '__main__':
    main()