import os, sys
from pathlib import Path
import pandas as pd
import torch

# 절대 경로 참조
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from modeling.predict import predict
from src.utils import config, set_seed
from data.dataset import (
    Collator,
    get_inputs,
)
from modeling.trainer import model, tokenizer
from data.make_input import test


# set seed
set_seed(config)

# Load model
fname = 'model.pt'
saved_data = torch.load(
    f"{config.model_fpath}/{fname}",
    map_location=torch.device('cpu'),
    # cuda gpu 사용시 아래 주석 한 줄 해제하여 map_location 설정
    # map_location="cpu" if config.gpu_id < 0 else "cuda:%d" % config.gpu_id,
)

# Parse
saved_model = saved_data['model']
train_config = saved_data['config']

# Load weights
model.load_state_dict(saved_model)

# Get test inputs
test_dataset = get_inputs(tokenizer, fpath=Path(config.test), mode='test')
test_dataloader = torch.utils.data.DataLoader(
          test_dataset,
          batch_size=config.batch_size,
          shuffle=False,
          num_workers=4,
          collate_fn=Collator(bos_token_id=tokenizer.bos_token_id,
                              eos_token_id=tokenizer.eos_token_id,
                              pad_token_id=tokenizer.pad_token_id,
                              inp_max_len=config.inp_max_len,
                              tar_max_len=config.tar_max_len,
                              mode="test",
                              ),
  )


####### Predict ! #######
outputs = predict(model, tokenizer, test_dataloader)

# 예측값 데이터 프레임화
id = [i['id'] for i in outputs]
pred = [i['output'] for i in outputs]
pred_df = pd.DataFrame(data={'id':id, 'predict':pred})

# 기존 테스트 데이터 프레임화
id = [i['id'] for i in test]
cont = [i['text'] for i in test]
ref = [i['summary'] for i in test]
ref_df = pd.DataFrame(data={'id':id,
                            'content':cont,
                            'reference':ref})

# id 기준으로 두 데이터프레임 merge
ref_pred = pd.merge(ref_df, pred_df, how='outer', on='id')

# Export CSV
fname = f'{config.tar_max_len / 3}_{config.beam_size}_ref_pred.csv'
ref_pred.to_csv(f"{config.ref_pred_path}/{fname}")

print(ref_pred.head())