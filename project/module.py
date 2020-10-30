import torch
import os
import pytorch_lightning as pl

from argparse import Namespace
from typing import Dict, Any, List
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup, AdamW
from torch.utils.data import DataLoader
from itertools import groupby

from project.helpers import indexed_chunk_list
from project.data import WSDDataset


class WSDFinetuner(pl.LightningModule):

    def __init__(self, 
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        print(self.hparams)

        self.mlm = self.hparams.mlm if 'mlm' in self.hparams else False
        self.create_components()
        self.train_dataset = WSDDataset(data_dir=self.hparams.data_dir, tokenizer=self.tokenizer,
                                        dataset_name="semcor", mlm=self.mlm, method=self.hparams.method)
        self.val_datasets = [
            WSDDataset(data_dir=self.hparams.data_dir, tokenizer=self.tokenizer, dataset_name=eval_name, mlm=self.mlm, method=self.hparams.method) for eval_name in self.hparams.eval_names
        ]

    def create_components(self) -> None:
        config = AutoConfig.from_pretrained(
            self.hparams.model_name_or_path,
            num_labels=2,
            hidden_dropout_prob=self.hparams.hidden_dropout_prob,
            gradient_checkpointing=self.hparams.gradient_checkpointing,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hparams.model_name_or_path,
            model_max_length=self.hparams.max_seq_length,
            use_fast=True
        )

        self.lm = AutoModelForSequenceClassification.from_pretrained(
            self.hparams.model_name_or_path,
            from_tf=bool(".ckpt" in self.hparams.model_name_or_path),
            config=config,
        )

    def forward(self, inputs: Dict[str, torch.Tensor]):
        # fwd
        return self.lm(**inputs)

    def training_step(self, batch: torch.Tensor, batch_nb: int = None) -> torch.Tensor:
        # fwd
        loss = self.forward(batch)[0]
        return loss

    def validation_step(self, val_batch: torch.Tensor, batch_idx: int = None, dataset_idx: int = None) -> Dict[str, Any]:
        # fwd
        loss = self.forward(val_batch)[0]
        return {'val_loss': loss, 'progress_bar': {'val_loss': loss}}

    def validation_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
        avg_loss = torch.stack([x['val_loss'] for el in outputs for x in el]).mean()
        return {'val_loss': avg_loss, 'progress_bar': {'val_loss': avg_loss}}

    def test_step(self, val_batch: torch.Tensor, batch_idx: int = None, dataset_idx: int = None) -> Dict[str, torch.Tensor]:
        # fwd
        values = self.forward(val_batch)
        y_hat = values[1]

        return {'y_hat': y_hat}

    def test_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        eval_results = {}

        # This assumes that the target_ids are sorted
        for dataset_idx, dataset_outputs in enumerate(outputs):
            predicted_synsets = self.get_lmgc_preds(dataset_outputs, dataset_idx)

            gold_keys = self.val_datasets[dataset_idx].gold_keys
            ok = notok = 0
            for key in gold_keys.keys():
                local_ok = local_notok = 0
                if key in predicted_synsets:
                    for answer in predicted_synsets[key]:
                        if answer in gold_keys[key]:
                            local_ok += 1
                        else:
                            local_notok += 1
                    ok += local_ok / len(predicted_synsets[key])
                    notok += local_notok / len(predicted_synsets[key])
                else:
                    break

            precision = ok / (ok + notok)
            recall = ok / len(gold_keys)
            if precision + recall == 0.0:
                f1 = 0.0
            else:
                f1 = (2 * precision * recall) / (precision + recall)

            eval_results[f'{self.val_datasets[dataset_idx].dataset_name}_precision'] = precision
            eval_results[f'{self.val_datasets[dataset_idx].dataset_name}_recall'] = recall
            eval_results[f'{self.val_datasets[dataset_idx].dataset_name}_f1'] = f1

        return eval_results

    def get_lmgc_preds(self, dataset_outputs: List[Dict[str, torch.Tensor]], dataset_idx: int) -> Dict[str, torch.Tensor]:
        y_hats = torch.cat([x['y_hat'] for x in dataset_outputs if 'y_hat' in x])
        target_ids = [self.val_datasets[dataset_idx].data[idx][0]
                      for idx in range(min(len(self.val_datasets[dataset_idx]), len(y_hats)))]
        sense_keys = [self.val_datasets[dataset_idx].data[idx][-1]
                      for idx in range(min(len(self.val_datasets[dataset_idx]), len(y_hats)))]

        # Calculate f1 list for the official scorer
        predicted_synsets = {}
        target_id_groups = [list(group) for _, group in groupby(target_ids)]
        counts = [len(group) for group in target_id_groups]
        split_y_hats = torch.split(y_hats, counts)
        split_sense_keys = list(indexed_chunk_list(sense_keys, counts))

        for sub_y_hat, sub_sense_key, target_id_group in zip(split_y_hats, split_sense_keys, target_id_groups):
            sub_y_hat = torch.softmax(sub_y_hat, dim=-1)
            best_pos = torch.argmax(sub_y_hat, dim=0)[1]
            synset_predictions = [sub_sense_key[best_pos]]
            predicted_synsets[target_id_group[0]] = synset_predictions

        return predicted_synsets

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW([p for p in self.parameters() if p.requires_grad], lr=self.hparams.learning_rate)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, shuffle=self.hparams.shuffle, num_workers=self.hparams.num_workers, batch_size=self.hparams.batch_size, pin_memory=True if torch.cuda.is_available() else False)

    def val_dataloader(self) -> List[DataLoader]:
        return [
            DataLoader(val_dataset, num_workers=self.hparams.num_workers, batch_size=self.hparams.batch_size, pin_memory=True if torch.cuda.is_available() else False, shuffle=False) for val_dataset in self.val_datasets
        ]

    def test_dataloader(self) -> List[DataLoader]:
        return [
            DataLoader(test_datasets, num_workers=self.hparams.num_workers, batch_size=self.hparams.batch_size, pin_memory=True if torch.cuda.is_available() else False, shuffle=False) for test_datasets in self.val_datasets
        ]

    def setup(self, stage):
        if stage == 'fit':
            # Get dataloader by calling it - train_dataloader() is called after setup() by default
            train_loader = self.train_dataloader()

            # Calculate total steps
            self.total_steps = (
                (len(train_loader.dataset) // (self.hparams.batch_size * max(1, self.hparams.gpus)))
                // self.hparams.accumulate_grad_batches
                * float(self.hparams.max_epochs)
            )

    def configure_optimizers(self):
        model = self.lm
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps
        )
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]


    @pl.utilities.rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        save_path = os.path.join(self.hparams.output_dir, self.hparams.model_name_or_path, 'best_checkpoint')
        self.lm.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    @pl.utilities.rank_zero_only
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        save_path = os.path.join(self.hparams.output_dir, self.hparams.model_name_or_path, 'best_checkpoint')
        self.lm.from_pretrained(save_path)
        self.tokenizer.from_pretrained(save_path)

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument(
            "--max_seq_length",
            default=128,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
        )
        parser.add_argument(
            "--mlm", action="store_true", help="Use Masked Language Modeling or not"
        )
        parser.add_argument("--mlm_prob", type=float, default=0.15, help="Masked language modeling Probability")
        parser.add_argument("--hidden_dropout_prob", type=float, default=0.2, help="dropout probability for training")

        return parser
