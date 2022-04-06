import os, sys
import pandas as pd
from .preprocess import preprocessing
from .ground_truth import ground_truth
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.utils import config, set_seed

# set seed
set_seed(config)

"""
! README !
ground truth 생성 코드는 실행이 오래 걸리는 이슈로 미리 생성해둔 csv 파일을 읽어온다.
"""

# # file path 설정
# READ_PATH = "/Users/mac/project/ABSSUM_KoBART/data/sports_news_data.csv"
# EXPORT_PATH = "/Users/mac/project/ABSSUM_KoBART/data"

# # 데이터 로드
# df = pd.read_csv(READ_PATH)

# # 전처리 적용
# df = preprocessing(df)

# # 라벨 생성
# df = ground_truth(df)

# # export CSV 
# df.to_csv(EXPORT_PATH, index = False)

GROUND_TRUTH_CSV_PATH = "/Users/mac/project/ABSSUM_KoBART/data/pororo_abs_df.csv"

df = pd.read_csv(GROUND_TRUTH_CSV_PATH)