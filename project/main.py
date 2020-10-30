import os
import glob
import torch
import argparse

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.loggers.neptune import NeptuneLogger

from project.helpers import add_generic_args
from project.module import WSDFinetuner

def main() -> None:
    parser = argparse.ArgumentParser()
    add_generic_args(parser)
    parser = WSDFinetuner.add_model_specific_args(parser)
    args = parser.parse_args()
    pl.seed_everything(args.seed)

    model = WSDFinetuner(**vars(args))

    # huggingface transformers uses parallel tokenizers which is a problem when we use parallel TPU/GPU backends
    # because it spawns a large number of total threads. Same is true for dataloaders with multiple workers
    if args.tpu_cores or torch.cuda.is_available() and torch.cuda.device_count() > 1 or args.num_workers >= 1:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(args.output_dir, args.model_name_or_path))

    train_params = {}
    train_params["accumulate_grad_batches"] = args.accumulate_grad_batches

    if args.fp16 and torch.cuda.is_available() or args.tpu_cores:
        train_params["precision"] = 16

    if args.gpus > 1:
        train_params["distributed_backend"] = "ddp"

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
        logger=[tb_logger],
        fast_dev_run=True,
        profiler="simple",
        **train_params
    )

    if args.do_train:
        trainer.fit(model)
    if args.do_predict and not args.mlm:
        checkpoints = list(
            sorted(glob.glob(os.path.join(args.output_dir, args.model_name_or_path, "best_checkpoint", "*.ckpt"), recursive=True))
        )
        model_test = model.load_from_checkpoint(checkpoints[-1], hparams=args)
        trainer.test(model_test)


if __name__ == '__main__':
    main()
