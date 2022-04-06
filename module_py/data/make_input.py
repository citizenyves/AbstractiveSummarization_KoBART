import re
import os, sys
import pathlib
from tqdm import tqdm
from pathlib import Path
from operator import itemgetter
import pandas as pd
from sklearn.model_selection import train_test_split
import transformers

# 절대 경로 참조
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.utils import config, set_seed
from label import labeling
from .preprocess import(
    read_json,
    CleanNewspaperArticleBase,
    CleanNewspaperArticle,
    extract_lines,
)
from .dataset import (
    make_final_document,
    read_tsv,
    CustomDataset,
    Collator,
    get_inputs,
    save_lines,
)


# set seed
set_seed(config)

######## 추가 데이터셋 (aihub 데이터셋) ########

# 추가 데이터셋 읽어오기 (aihub 데이터셋)
aihub_corpus = read_json(config.aihub_path)
aihub_corpus = sorted(aihub_corpus,
                      key=itemgetter("id"),
                      reverse=False,
                     )

# 전처리 함수 적용
aihub_doc = extract_lines(aihub_corpus)

# 데이터프레임화
aihub_df = pd.DataFrame(aihub_doc)
aihub_df = aihub_df[['text', 'summary']].rename(columns={'text':'TITLE_CONT',
                                                         'summary':'SUMMARY'})
# 중복값 제거
aihub_df = aihub_df.drop_duplicates(['TITLE_CONT', 'SUMMARY'], keep='first')

# 238,801개의 데이터 중 7,000개 랜덤샘플링
aihub_df = aihub_df.sample(n=7000, random_state=config.seed).reset_index(drop=True)

# CSV파일로 저장하기
aihub_df.to_csv(config.aihub_7000, index = False)



######## 최종 document 생성 ########

# 기본데이터(라벨링된 상태) 호출
df = labeling.df

# 컬럼명 변경
df.rename(columns={"TRUE SUMMARY" : "SUMMARY"}, inplace=True)

# 최종 document 생성
"""
1. aihub
: aihub 데이터에서 스포츠 기사를 제외하고, 랜덤샘플링으로 7,000개를 추출한 document
: len(aihub) = 7000

2. base
: 최종 가공이 끝나고, ground truth 라벨이 함께 있는 기본 document
: len(base) = 9050
"""
aihub = make_final_document(aihub_df)
base = make_final_document(df)



######## trainset, validset, testset 분할 ########
"""
1) 기본데이터에서 최종테스트를 위한 test 사이즈 2,000을 떼어 냄 
- 9,050 -> (7,050, 2,000)

2) 2,000을 떼어낸 기본데이터에 aihub 데이터를 결합
- 7,050 + 7,000 -> 14,050

3) 결합한 데이터를 train / valid(10%) 셋으로 분할<br>
- 14,050 -> (12,645, 1,405) 

4) 최종 데이터셋
- train : 12,645
- valid : 1,405
- test  : 2,000
"""
# train, test split
train_total, test = train_test_split(base, test_size=2000, shuffle=True, random_state=config.seed)

# 기본데이터(train) + 추가데이터셋 결합
train_total.extend(aihub)

# train, valid split : valid ratio : 0.1
train, valid = train_test_split(train_total, test_size=0.1, shuffle=True, random_state=config.seed)

# tsv 파일로 저장
save_lines("train", train) 
save_lines("valid", valid) 
save_lines("test", test)



######## 최종 Inputs 생성 ########

# Get pretrained tokenizer
tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(config.pretrained_model_name)

# Get inputs
train_dataset = get_inputs(tokenizer, fpath=Path(config.train))
valid_dataset = get_inputs(tokenizer, fpath=Path(config.valid))

