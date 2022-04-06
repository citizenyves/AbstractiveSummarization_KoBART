import torch
import random
import numpy as np
import easydict
import pprint


config = easydict.EasyDict({
    "clean": True, 
    "seed": 42,
    "pretrained_model_name": "gogamza/kobart-base-v2",

    # path(read and export)    
    "save_fpath": "/Users/mac/project/ABSSUM_KoBART/data",
    "sports_news_data": "/Users/mac/project/ABSSUM_KoBART/data/sports_news_data.csv",
    "pororo_abs_sum_df": "/Users/mac/project/ABSSUM_KoBART/data/pororo_abs_df.csv",
    "aihub_path": "/Users/mac/project/ABSSUM_KoBART/data/train_original.json",
    "aihub_7000": "/Users/mac/project/ABSSUM_KoBART/data/aihub_df.csv",
    "train": "/Users/mac/project/ABSSUM_KoBART/data/train.tsv", 
    "valid": "/Users/mac/project/ABSSUM_KoBART/data/valid.tsv", 
    "test": "/Users/mac/project/ABSSUM_KoBART/data/test.tsv",  
    "ref_pred_path": "/Users/mac/project/ABSSUM_KoBART/data",
    "sbert": "/Users/mac/project/ABSSUM_KoBART/training_klue_sts_jhgan-ko-sroberta-multitask_2022-03-15_15-37-56",

    # Training arguments.
    "ckpt": "/Users/mac/project/ABSSUM_KoBART/model",          
    "logs": "/Users/mac/project/ABSSUM_KoBART/logdir",             
    "model_fpath": "/Users/mac/project/ABSSUM_KoBART/model",   
    "batch_size": 8,       
    "gradient_accumulation_steps": 16, 
    "lr": 5e-5,           
    "weight_decay": 1e-2, 
    'adam_beta1': 0.9,
    'adam_beta2': 0.999,
    'adam_epsilon': 1e-8,
    "max_grad_norm": 1.0,
    "warmup_ratio": 0.0,  
    "n_epochs": 10,       
    "inp_max_len": 1024,  
    "tar_max_len": 150,    
    
    # Inference arguments
    "gpu_id": 0,
    "beam_size": 3,             
    "length_penalty": 1.0,      
    "no_repeat_ngram_size": 3,  
    "var_len": False,
})


def print_elements(a: dict) -> None:
    pprint.PrettyPrinter(indent=4).pprint(a)


def set_seed(config):
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])

