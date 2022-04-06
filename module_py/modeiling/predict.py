import torch
import transformers
from tqdm import tqdm
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.utils import config, set_seed


# set seed
set_seed(config)

# predict 함수
def predict(model, tokenizer, test_dataloader):
    global min_length
    # no_grad
    with torch.no_grad():
        ### cuda gpu 사용시 아래 두 줄 주석 해제 ###
        # if config.gpu_id >= 0:
        #     model.cuda(config.gpu_id)
        device = next(model.parameters()).device

        # eval mode
        model.eval()

        # 예측값 리스트
        outputs = []

        # batch 단위로 예측
        for batch in tqdm(test_dataloader, total=len(test_dataloader)):
            id = batch["id"]
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

            # inputs to device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # min/max lens for generating
            min_length = config.tar_max_len // 3
            max_length = config.tar_max_len

            # Generate ids of summaries.
            output = model.generate(
                input_ids, 
                attention_mask=attention_mask,
                max_length=max_length,                  # maximum summarization size
                min_length=min_length,                  # minimum summarization size
                early_stopping=True,                    # stop the beam search when at least 'num_beams' sentences are finished per batch
                num_beams=config.beam_size,             # beam search size
                bos_token_id=tokenizer.bos_token_id,    # <s> = 0
                eos_token_id=tokenizer.eos_token_id,    # <\s> = 1
                pad_token_id=tokenizer.pad_token_id,    # 3
                length_penalty=config.length_penalty,   # value > 1.0 in order to encourage the model to produce longer sequences
                no_repeat_ngram_size=config.no_repeat_ngram_size,   # same as 'trigram blocking'
            )
            
            """
            문장마다 디코딩을 하려면 batch_encode 대신 decode 메서드 활용
            """
            # 디코딩 
            output = tokenizer.batch_decode(
                output.tolist(), 
                skip_special_tokens=True,
            )

            # 최종 outputs
            outputs.extend([{"id": id_, "output": output_} for id_, output_ in zip(id, output)])

    return outputs