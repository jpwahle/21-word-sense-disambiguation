import os
import torch
import csv
import linecache

from os import PathLike
from typing import Dict
from multiprocessing import Manager
from transformers import AutoTokenizer


class WSDDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: PathLike, tokenizer: AutoTokenizer, mlm: bool, method: str, dataset_name: str, mlm_prob: float = 0.15) -> None:
        self.manager = Manager()
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.mlm = mlm
        self.method = method
        self.mlm_probability = mlm_prob
        if 'semcor' not in dataset_name:
            self.gold_keys = self.load_gold_keys(os.path.join(data_dir, 'gold_keys', f'{dataset_name}.gold.key.txt'))
            self.file_path = os.path.join(data_dir, 'examples', f'{dataset_name}_test_token_cls.csv')
            self.load_keys(os.path.join(data_dir, 'examples', f'{dataset_name}_test_token_cls.csv'))
        else:
            self.file_path = os.path.join(data_dir, 'examples', f'{dataset_name}_train_token_cls.csv')
            self.load_keys(os.path.join(data_dir, 'examples', f'{dataset_name}_train_token_cls.csv'))

        with open(self.file_path) as f:
            self.dataset_size = len(f.readlines()) - 1

    def load_keys(self, input_file: PathLike) -> None:
        # Don't use standard list because of known memory leak with copy-on-read https://github.com/pytorch/pytorch/issues/13246
        self.data = self.manager.list()
        with open(input_file, "r", encoding="'iso-8859-1'") as f:
            reader = csv.reader(f, delimiter="\t")
            # Skip the header
            next(reader)
            if self.method == 'lmgc':
                self.data = self.manager.list([el for el in reader])
                return
            elif self.method == 'mcgp':
                prev_target_id = None
                tmp_list = []
                for el in reader:
                    if not prev_target_id:
                        prev_target_id = el[0]
                        continue
                    if prev_target_id != el[0]:
                        prev_target_id = el[0]
                        self.data.append(tmp_list)
                        tmp_list = []
                    tmp_list.append(el)
            else:
                raise NotImplementedError(f"Method {self.method} not implemented.")

    def load_gold_keys(self, gold_key_file: PathLike) -> Dict[str, str]:
        gold_keys = {}
        with open(gold_key_file, "r", encoding="utf-8") as f:
            s = f.readline().strip()
            while s:
                tmp = s.split()
                gold_keys[tmp[0]] = tmp[1:]
                s = f.readline().strip()

        return gold_keys

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.data[idx]
        sentence = str(data[2])
        gloss = str(data[3])
        wsd_label = int(data[1])
        word_start = int(data[-3])
        word_end = int(data[-2])

        words = sentence.split()
        words = words[:word_start] + ['"'] + words[word_start:word_end] + ['"'] + words[word_end:]
        gloss_words = gloss.split()
        gloss_words = words[word_start:word_end] + [':'] + gloss_words
        sentence = ' '.join(words)
        gloss = ' '.join(gloss_words)

        batch_encoding = self.tokenizer(
            sentence,
            gloss,
            padding="max_length",
            truncation=True,
            return_tensors='pt',
            max_length=self.tokenizer.model_max_length
        )
        batch_encoding['labels'] = torch.tensor(wsd_label)
        batch_encoding = {key: value.squeeze() for key, value in batch_encoding.items()}

        return batch_encoding

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:

        batch_encoding = self.tokenizer(
            [x['sentence'] for x in batch],
            [x['gloss'] for x in batch],
            padding="max_length",
            truncation=True,
            return_tensors='pt',
            max_length=self.tokenizer.model_max_length
        )
        batch_encoding['labels'] = torch.tensor([x['wsd_label'] for x in batch])

        return batch_encoding

    def __len__(self) -> int:
        return self.dataset_size



