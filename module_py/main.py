import os, sys
import torch
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from modeling.trainer import model, tokenizer, Trainer
from src.utils import config, set_seed


# device type
if torch.cuda.is_available():
    device = torch.device("cuda")
    # print(f"# available GPUs : {torch.cuda.device_count()}")
    # print(f"GPU name : {torch.cuda.get_device_name()}")
else:
    device = torch.device("cpu")

# set seed
set_seed(config)

# driver
if __name__ == '__main__':
    trainer = Trainer(model, tokenizer)
    trainer.train()
    
    # save model
    fname = 'model2.pt'
    torch.save({
        "model": trainer.model.state_dict(),
        "config": config,
        "tokenizer": tokenizer,
    }, Path(f"{config.model_fpath}/{fname}"))
