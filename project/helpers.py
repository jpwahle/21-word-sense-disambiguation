import torch
import multiprocessing
from typing import Iterable
from itertools import islice


def indexed_chunk_list(li: list, chunk_sizes: list) -> Iterable:
    it = iter(li)
    i = 0
    while True:
        if i >= len(chunk_sizes):
            break
        nxt = list(islice(it, chunk_sizes[i]))
        if nxt:
            i += 1
            yield nxt
        else:
            break


def add_generic_args(parser) -> None:
    #  To allow all pl args uncomment the following line
    #  parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )

    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O2",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--n_tpu_cores", dest="tpu_cores", type=int)
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the eval set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")
    parser.add_argument("--fast_dev_run", action="store_true",
                        help="Run a quick dev run to check if everything works fine")
    parser.add_argument("--eval_names", nargs='+', help="Whether to run predictions on the test set.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        dest="accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--model_name_or_path", type=str, help="Generic Model Name from huggingface models")
    parser.add_argument("--max_epochs", type=int, help="Number of epochs to train")
    parser.add_argument("--progress_bar_refresh_rate", type=int, default=10, help="Progress bar refresh rate")
    parser.add_argument("--val_check_interval", type=float, default=1.0, help="Eval every n-th epoch")
    parser.add_argument("--batch_size", type=int, help="Batch size for training, eval and test.")
    parser.add_argument("--auto_scale_batch_size", default='binsearch' if torch.cuda.device_count() ==
                        1 else False, help="Batch size for training, eval and test.")
    parser.add_argument("--method", type=str, help="Method to use (LMGC, MCGP)")
    parser.add_argument("--gpus", type=int, default=1,
                        help="Which GPUs to use. Either list of ids or num_gpus. Defaults to device_count().")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for training")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the training dataset or not")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Gradient checkpointing for large models")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of workers. Defualts to multiprocessing.cpu_count()")
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
    )
