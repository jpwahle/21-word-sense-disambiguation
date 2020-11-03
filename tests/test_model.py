import os, multiprocessing
import pytorch_lightning as pl

# Known issue by huggingface transformers https://github.com/huggingface/transformers/pull/7033
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from argparse import Namespace
from project.model import GilBERTFinetuner

def test_model_args():
    pl.seed_everything(1234)

    args = dict()
    args['output_dir'] = os.getcwd()
    args['data_dir'] = os.path.join(os.getcwd(), 'data')
    args['model_name_or_path'] = 'bert-base-uncased'
    args['max_epochs'] = 1
    args['fast_dev_run'] = True
    args['do_eval'] = True
    args['do_train'] = True
    args['do_predict'] = True
    args['max_seq_length'] = 160
    args['batch_size'] = 2
    args['method'] = 'bgp'
    args['hidden_dropout_prob'] = 0.1
    args['gradient_checkpointing'] = False
    args['learning_rate'] = 2e-5
    args['shuffle'] = False
    args['num_workers'] = multiprocessing.cpu_count()
    args['checkpoint_callback'] = False
    args['eval_names'] = ["ALL", "semeval2007", "semeval2013", "semeval2015", "senseval2", "senseval3"]
    
    args= Namespace(**args)
    
    model = GilBERTFinetuner(args)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model)
    trainer.test(model)
    
if __name__ == '__main__':
    test_model_args()