import torch, os, multiprocessing, json, subprocess, tqdm
import pytorch_lightning as pl

from subprocess import STDOUT, PIPE
from itertools import groupby
from typing import Dict, Any

from transformers import AutoModelForSequenceClassification, AutoModelForPreTraining, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from helpers import squeeze_collator, indexed_chunk_list

class GilBERTFinetuner(pl.LightningModule):

    def __init__(self, hparams, config, tokenizer, train_dataset, val_datasets):
        super(GilBERTFinetuner, self).__init__()
        
        if hparams.mlm:
            self.lm = AutoModelForPreTraining.from_pretrained(
                hparams.model_name_or_path,
                from_tf=bool(".ckpt" in hparams.model_name_or_path),
                config=config
            )
        else:
            self.lm = AutoModelForSequenceClassification.from_pretrained(
                hparams.model_name_or_path,
                from_tf=bool(".ckpt" in hparams.model_name_or_path),
                config=config,
            )

        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_datasets = val_datasets
        self.learning_rate = hparams.learning_rate
        self.batch_size = hparams.batch_size
        self.hparams = hparams
        self.mlm = hparams.mlm if 'mlm' in hparams else False
    
    def forward(self, inputs):
        # fwd
        return self.lm(**inputs)

    def training_step(self, batch, batch_nb=None):        
        # fwd
        loss = self.forward(batch)[0]
        if self.hparams.mlm:
            loss = loss.mean()
        return loss

    def validation_step(self, val_batch, batch_idx=None, dataset_idx=None):
        # fwd
        values = self.forward(val_batch)
        loss = values[0]
        
        return {'val_loss': loss, 'progress_bar': {'val_loss': loss}}

    def validation_epoch_end(self, outputs):            
        avg_loss = torch.stack([x['val_loss'] for el in outputs for x in el]).mean()
        
        return {'avg_val_loss': avg_loss, 'progress_bar': {'avg_val_loss': avg_loss}}
    
    def test_step(self, val_batch, batch_idx=None, dataset_idx=None):
        # fwd
        values = self.forward(val_batch)

        loss = values[0]
        y_hat = values[1]
        
        return {'val_loss': loss, 'y_hat': y_hat}
        
    def test_epoch_end(self, outputs):
        eval_results = {}
            
        # This assumes that the target_ids are sorted
        for dataset_idx, dataset_outputs in enumerate(outputs):
            if self.mlm:
                avg_val_loss = torch.stack([x['val_loss'] for x in dataset_outputs]).mean()
                return {f'{self.val_datasets[dataset_idx].dataset_name}_avg_val_loss': avg_val_loss}
            elif self.hparams.method == 'bgp':
                predicted_synsets = self.get_bgp_preds(dataset_outputs, dataset_idx)
            elif self.hparams.method == 'mcgp':
                predicted_synsets = self.get_mcgp_preds(dataset_outputs)
            else:
                raise NotImplementedError(f"Method {self.hparams.method} with mlm={self.mlm} not Implemented")
                    
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
                    ok += local_ok/len(predicted_synsets[key])
                    notok += local_notok/len(predicted_synsets[key])
                else:
                    break
                        
            precision = ok / (ok + notok)
            recall = ok / len(gold_keys)
            if precision + recall == 0.0:
                f1 = 0.0
            else:
                f1 = (2 * precision  * recall) / (precision + recall)
            
            eval_results[f'{self.val_datasets[dataset_idx].dataset_name}_precision'] = precision
            eval_results[f'{self.val_datasets[dataset_idx].dataset_name}_recall'] = recall
            eval_results[f'{self.val_datasets[dataset_idx].dataset_name}_f1'] = f1
            
        avg_loss = torch.stack([x['val_loss'] for el in outputs for x in el]).mean()
        eval_results['avg_val_loss'] = avg_loss
        
        return eval_results
                        
    def get_bgp_preds(self, dataset_outputs, dataset_idx):
        y_hats = torch.cat([x['y_hat'] for x in dataset_outputs if 'y_hat' in x])
        target_ids = [self.val_datasets[dataset_idx].data[idx][0] for idx in range(min(len(self.val_datasets[dataset_idx]), len(y_hats)))]
        sense_keys = [self.val_datasets[dataset_idx].data[idx][-1] for idx in range(min(len(self.val_datasets[dataset_idx]), len(y_hats)))]

        # Calculate f1 list for the official scorer
        predicted_synsets = {}
        target_id_groups =  [list(group) for _, group in groupby(target_ids)]      
        counts = [len(group) for group in target_id_groups]
        split_y_hats = torch.split(y_hats, counts)
        split_sense_keys = list(indexed_chunk_list(sense_keys, counts))
        
        for sub_y_hat, sub_sense_key, target_id_group in zip(split_y_hats, split_sense_keys, target_id_groups): 
            # If there are only negatives, just take the best positive
            sub_y_hat = torch.softmax(sub_y_hat, dim=-1)
            best_pos = torch.argmax(sub_y_hat, dim=0)[1]
            synset_predictions = [sub_sense_key[best_pos]]
            predicted_synsets[target_id_group[0]] = synset_predictions

        return predicted_synsets

    def get_mcgp_preds(self, dataset_outputs):
        predicted_synsets = {}
        for example in dataset_outputs:
            y_hat = example['y_hat']
            target_ids = example['target_id']
            sense_keys = example['sense_key']

            # Calculate f1 list for the official scorer
            selected_prediction = torch.argmax(torch.softmax(y_hat, dim=1))[1]
            
            synset_prediction = sense_keys[selected_prediction]
            predicted_synsets[target_ids[0]] = synset_prediction
        
        return predicted_synsets
    
    def compile_java(self, java_file):
        subprocess.check_call(["javac", java_file])

    def run_scorer(self, java_file, test_dataset, output_eval_file):
        java_class, _ = os.path.splitext(java_file)
        cmd = ["java", java_class, os.path.join(self.hparams.data_dir, 'gold_keys', f'{test_dataset.dataset_name}.gold.key.txt'), output_eval_file]
        proc = subprocess.Popen(cmd, stdout=PIPE, stderr=STDOUT)
        stdout, _ = proc.communicate(input="SomeInputstring")
        result = stdout.decode("utf-8").split("\n")
        result = [s.split("=\t") for s in result]
        result = {s[0]: s[1][:-1].replace(",", ".") for s in result if len(s) == 2}
        return result
    
    def configure_optimizers(self):
        return torch.optim.AdamW([p for p in self.parameters() if p.requires_grad], lr=self.learning_rate)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=self.hparams.shuffle, num_workers=(self.hparams.num_workers or multiprocessing.cpu_count()), batch_size=self.batch_size, collate_fn=squeeze_collator, pin_memory=True if torch.cuda.is_available and torch.cuda.device_count() > 1 else False)
    
    def val_dataloader(self):
        return [              
            DataLoader(val_dataset, num_workers=(self.hparams.num_workers or multiprocessing.cpu_count()), batch_size=self.batch_size, collate_fn=squeeze_collator, pin_memory=True if torch.cuda.is_available and torch.cuda.device_count() > 1 else False, shuffle=False) for val_dataset in self.val_datasets
        ]
        
    def test_dataloader(self):
        return [              
            DataLoader(val_dataset, num_workers=(self.hparams.num_workers or multiprocessing.cpu_count()), batch_size=self.batch_size, collate_fn=squeeze_collator, pin_memory=True if torch.cuda.is_available and torch.cuda.device_count() > 1 else False, shuffle=False) for val_dataset in self.val_datasets
        ]
        
    @pl.utilities.rank_zero_only
    def on_save_checkpoint(self) -> None:
        save_path = self.output_dir.joinpath("best_tfmr")
        self.model.config.save_step = self.step_count
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
    
    @staticmethod
    def add_model_specific_args(parser, root_dir):
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
        parser.add_argument("--hidden_dropout_prob", type=float, default=0.2, help="Learning rate for training")

        return parser