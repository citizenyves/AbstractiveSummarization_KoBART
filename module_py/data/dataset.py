import torch
import pathlib
import os, sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict

# 절대 경로 참조
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.utils import config, set_seed


# set seed
set_seed(config)

# input 생성을 위한 최종 document 함수
def make_final_document(df):
    # aihub 데이터셋
    if len(df) == 7000:
        li = []
        for i in range(len(df)):
            dic = {}
            dic['id'] = i + 9050
            dic['summary'] = df['SUMMARY'][i]
            dic['text'] = df['TITLE_CONT'][i]
            li.append(dic)
    
    # 기본 데이터셋
    elif len(df) != 7000:
        li = []
        for i in range(len(df)):
            dic = {}
            dic['id'] = i
            dic['summary'] = df['SUMMARY'][i]
            dic['text'] = df['TITLE_CONT'][i]
            li.append(dic)        
    
    return li


def read_tsv(fpath: pathlib.PosixPath) -> pd.DataFrame:
    return pd.read_csv(fpath, index_col=False, sep="\t", encoding="utf-8")


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, tokenizer, fpath: pathlib.PosixPath, mode: str = "train"):
        super(CustomDataset, self).__init__()

        
        self.df = read_tsv(fpath)
        # # self.tok = tokenizer -> don't keep

        # Mode
        assert mode in ['train', 'test']
        self.mode = mode

        # Apply tokenizer first to speed up in training phase and make code more simply.
        tqdm.pandas(desc='Tokenizing input texts')
        self.df.loc[:, "text_tok"] = self.df.loc[:, "text"].progress_apply(lambda x: tokenizer.encode(x))
        self.df.loc[:, "text_tok_len"] = self.df.loc[:, "text_tok"].apply(lambda x: len(x))

        if self.mode == "train":
            tqdm.pandas(desc="Tokenizing target summaries")
            self.df.loc[:, "summary_tok"] = self.df.loc[:, "summary"].progress_apply(lambda x: tokenizer.encode(x))
            self.df.loc[:, "summary_tok_len"] = self.df.loc[:, "summary_tok"].apply(lambda x: len(x))

        self.df.sort_values(by=['text_tok_len'], axis=0, ascending=False, inplace=True)
    
    def __len__(self) -> int:
        return self.df.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        instance = self.df.iloc[idx]
        
        # id, text, length, summary
        return_value = {
            'id':instance['id'], ## for sorting in inference mode
            'text':instance['text_tok'],
            'length':len(instance['text_tok']),
        }

        if self.mode == 'train':
            return_value['summary'] = instance['summary_tok']
        
        return return_value


class Collator():

    def __init__(
        self,
        bos_token_id: int,
        eos_token_id: int,
        pad_token_id: int,
        inp_max_len: int = config.inp_max_len, 
        tar_max_len: int = config.tar_max_len,
        ignore_index: int = -100,
        mode: str = "train",
    ):
        super(Collator, self).__init__()

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.inp_max_len = inp_max_len
        self.tar_max_len = tar_max_len
        self.ignore_index = ignore_index

        ## Mode.
        assert mode in ["train", "test"]
        self.mode = mode


    def _pad(self, sentences: List[List[int]], token_id: int) -> np.ndarray:
        # 각 배치 당 max length로 패딩 
        ## We will pad as max length per batch, not "inp_max_len(=1024, etc)".
        max_length_per_batch = max([len(i) for i in sentences])

        # pad토큰으로 batch의 max_len보다 짧은 문장들 채움
        ## Stack as dimension 0 (batch dimension).
        ## "token_id" can be "tokenizer.pad_token_id(=3)" or "ignore_index(=-100)"
        return np.stack([i + [token_id] * (max_length_per_batch - len(i)) for i in sentences], axis=0)


    def _train_collator(self, samples: List[Dict[str, List[int]]]) -> Dict[str, List[int]]:
        ## Unpack.

        ## If input max length > 1024, you can see below error:
        ##   1) Assertion `srcIndex < srcSelectDimSize` failed
        ##   2) Device-side assert triggered
        tokenized_texts     = [s["text"][:self.inp_max_len]        for s in samples]
        tokenized_summaries = [s["summary"][:self.tar_max_len - 1] for s in samples] ## <bos> or <eos> token index

        ## Inputs for encoder.
        input_ids = self._pad(tokenized_texts, token_id=self.pad_token_id)  ## numpy format
        attention_mask = (input_ids != self.pad_token_id).astype(float)     ## numpy format

        ## Inputs for decoder (generator).
        decoder_input_ids = [[self.bos_token_id] + i for i in tokenized_summaries]      ## bos
        decoder_input_ids = self._pad(decoder_input_ids, token_id=self.pad_token_id)    ## eos
        decoder_attention_mask = (decoder_input_ids != self.pad_token_id).astype(float)

        ## Answer. (labels)
        labels = [i + [self.eos_token_id] for i in tokenized_summaries]
        labels = self._pad(labels, token_id=self.ignore_index) ## why != "padding_id" ???

        ## We ensure that generator's inputs' and outputs' shapes are equal.
        assert decoder_input_ids.shape == labels.shape
        
        ## Pack as pre-defined arguments. See:
        ##   https://huggingface.co/docs/transformers/model_doc/bart#transformers.BartForConditionalGeneration
        return {
            "input_ids":                torch.from_numpy(input_ids),
            "attention_mask":           torch.from_numpy(attention_mask),
            "decoder_input_ids":        torch.from_numpy(decoder_input_ids),
            "decoder_attention_mask":   torch.from_numpy(decoder_attention_mask),
            "labels":                   torch.from_numpy(labels),
        }


    def _test_collator(self, samples: List[Dict[str, List[int]]]) -> Dict[str, List[int]]:
        ## Unpack.
        ids              = [s["id"]                      for s in samples]
        tokenized_texts  = [s["text"][:self.inp_max_len] for s in samples]   ## no <bos> token included

        ## Inputs for encoder.
        input_ids = self._pad(tokenized_texts, token_id=self.pad_token_id)  ## numpy format
        attention_mask = (input_ids != self.pad_token_id).astype(float)     ## numpy format

        ## Pack as pre-defined arguments:
        ## See: https://huggingface.co/docs/transformers/model_doc/bart#transformers.BartForConditionalGeneration
        return {
            "input_ids":        torch.from_numpy(input_ids),
            "attention_mask":   torch.from_numpy(attention_mask),
            ## Additional information to make answer.
            "id":               ids,
        }


    def __call__(self, samples: List[Dict[str, List[int]]]) -> Dict[str, List[int]]:
        return self._train_collator(samples) if self.mode == "train" else self._test_collator(samples)


def get_inputs(tokenizer, fpath: pathlib.PosixPath, mode: str = "train"):
    return CustomDataset(tokenizer, fpath, mode=mode)


# tsv 파일로 train, valid, test 저장
def save_lines(mode: str, documents: List[Dict[str, str]]) -> None:
    assert mode in ["train", "valid", "test"]

    ## Save as: ./data/train.tsv, ./data/valid.tsv, ./data/test.tsv.
    fpath = f"{config.save_fpath}/{mode}.tsv"
    pd.DataFrame(documents).to_csv(fpath, index=False, sep="\t", encoding="utf-8")

    print(f"File {fpath} saved.")