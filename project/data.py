import os, torch, csv

from os import PathLike
from typing import List
from multiprocessing import Manager
from transformers import AutoTokenizer

class WSDDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: PathLike, tokenizer: AutoTokenizer, mlm: bool, method: str, dataset_name: str = '', mlm_prob: float = 0.15):
        self.manager = Manager()
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.mlm = mlm
        self.method = method
        self.mlm_probability = mlm_prob
        if dataset_name != 'train':
            self.gold_keys = self.load_gold_keys(os.path.join(data_dir, 'gold_keys', f'{dataset_name}.gold.key.txt'))
            self.load_keys(os.path.join(data_dir, 'examples', f'{dataset_name}_test_token_cls.csv'))
        else:
            self.load_keys(os.path.join(data_dir, 'examples', 'semcor_train_token_cls.csv'))
    
    def load_keys(self, input_file: PathLike):
        # Don't use standard list because of known memory leak with copy-on-read https://github.com/pytorch/pytorch/issues/13246
        self.data = self.manager.list()
        with open(input_file, "r", encoding="'iso-8859-1'") as f:
            reader = csv.reader(f, delimiter="\t")
            # Skip the header
            next(reader)
            if self.method == 'bgp':
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
                
    def load_gold_keys(self, gold_key_file: PathLike):
        gold_keys = {}
        with open(gold_key_file, "r", encoding="utf-8") as f:
            s = f.readline().strip()
            while s:
                tmp = s.split()
                gold_keys[tmp[0]] = tmp[1:]
                s = f.readline().strip()

        return gold_keys
            
    def mask_tokens_wsd(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling without masking the ambigous token: 80% MASK, 10% random, 10% original.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels        
        
    def __getitem__(self, idx: int):
        data = self.data[idx]                    
        if self.method == 'bgp':
            sentence = str(data[2])
            gloss = str(data[3])
            wsd_label = int(data[1])
            word_start = int(data[-3])
            word_end = int(data[-2])
        elif self.method == 'mcgp':
            sentence = [str(d[2]) for d in data]
            gloss = [str(d[3]) for d in data]
            wsd_label = [int(d[1]) for d in data]
        else:
            raise NotImplementedError(f"Method {self.method} not implemented.")
                
        words = sentence.split()
        words = words[:word_start] + ['"'] + words[word_start:word_end] + ['"'] + words[word_end:]
        gloss_words = gloss.split()
        gloss_words = words[word_start:word_end] +  [':'] + gloss_words
        sentence = ' '.join(words)
        gloss = ' '.join(gloss_words)
        
        tokens = self.tokenizer(
            sentence, gloss,
            max_length=self.tokenizer.max_len,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        )

        if self.mlm:
            tokens['input_ids'], mlm_labels = self.mask_tokens_wsd(tokens['input_ids'])
            tokens['labels'] = mlm_labels
        else:
            tokens['labels'] = torch.tensor(wsd_label)
        
        return tokens

    def __len__(self):
        return len(self.data)