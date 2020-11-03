import os, csv
import numpy as np
from transformers import AutoTokenizer

def test_seq_len():
    data_dir = os.path.join(os.getcwd(), 'data', 'examples')
    thresh = 160

    tokenizer = AutoTokenizer.from_pretrained(
        'bert-base-uncased',
        use_fast=True
    )

    for file in os.listdir(data_dir):
        with open(os.path.join(data_dir, file), "r", encoding="'iso-8859-1'") as f:
            reader = csv.reader(f, delimiter="\t")
            # Skip the header
            next(reader)
            # Read all lines including special tokens
            lengths = [len(tokenizer.tokenize(el[2], el[3])) + 5 for el in reader]
            num_truncated = np.count_nonzero([el > thresh for el in lengths])
            result = f'Elements larger than {thresh}: {num_truncated}. Elements in total {len(lengths)}. Percentage to be truncated {num_truncated/len(lengths)}'
            print('#' * len(result))
            print(file)
            print(f'Max: {np.max(lengths)}, Mean: {np.mean(lengths)}')
            print(result)    
            assert num_truncated/len(lengths) < 0.005
            
if __name__ == '__main__':
    test_seq_len()