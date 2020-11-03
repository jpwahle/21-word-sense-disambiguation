# Known issue by huggingface transformers https://github.com/huggingface/transformers/pull/7033
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os, argparse, torch, glob
import pytorch_lightning as pl

from project.model import GilBERTFinetuner
from project.helpers import add_generic_args

from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning import loggers as pl_loggers

def main():
    parser = argparse.ArgumentParser()
    add_generic_args(parser)
    parser = GilBERTFinetuner.add_model_specific_args(parser)
    args = parser.parse_args()
    pl.seed_everything(args.seed)
    
    neptune_logger = neptune_logger = NeptuneLogger(project_name="jpelhaw/GilBERT2") if args.neptune_logging else None
    model = GilBERTFinetuner(args)
    
    # huggingface transformers uses parallel tokenizers which is a problem when we use parallel TPU/GPU backends
    # because it spawns a large number of total threads. Same is true for dataloaders with multiple workers
    if args.tpu_cores or torch.cuda.is_available() and torch.cuda.device_count() > 1 or args.num_workers > 1:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(args.output_dir, args.model_name_or_path))
    
    train_params = {}
    if args.fp16 and torch.cuda.is_available() or args.tpu_cores:
        train_params["precision"] = 16

    if args.gpus > 1:
        train_params["distributed_backend"] = "ddp"

    train_params["accumulate_grad_batches"] = args.accumulate_grad_batches
    train_params["limit_train_batches"] = 1.0
    train_params["limit_val_batches"] = 1.0 if args.do_eval else 0
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, args.model_name_or_path, "best_checkpoint"),
        filename='{epoch}-{val_loss:.3f}',
        monitor="val_loss",
        mode="min",
        save_top_k=1
    )
    
    trainer = pl.Trainer.from_argparse_args(
        args,
        checkpoint_callback=checkpoint_callback,
        logger=[neptune_logger, tb_logger] if args.neptune_logging and neptune_logger else tb_logger,
        **train_params
    )
    
    if args.do_train:
        trainer.fit(model)
    if args.do_predict:
        checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, args.model_name_or_path, "best_checkpoint", "*.ckpt"), recursive=True)))
        model_test = model.load_from_checkpoint(checkpoints[-1], hparams=args)
        trainer.test(model_test)

if __name__ == '__main__':
    main()