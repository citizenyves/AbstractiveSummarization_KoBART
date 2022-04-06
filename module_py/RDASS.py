import os, sys
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from predict import ref_pred
from src.utils import config, set_seed
from src.metric import (
    rdass_pred_ref, 
    rdass_pred_cont, 
    make_embedding, 
    cosine_similarity,
    RDASS,
)

# set seed
set_seed(config)

# 사전훈련된 sbert모델 가져오기
SBERT = SentenceTransformer(f"./{config.sbert}")

# data pairs 생성
pred_refer = rdass_pred_ref(ref_pred)
pred_cont = rdass_pred_cont(ref_pred)

# 문장 embedding 생성 (768차원)
embedding_pred = make_embedding(SBERT, ref_pred, 'predict')
embedding_refer = make_embedding(SBERT, ref_pred, 'reference')
embedding_cont = make_embedding(SBERT, ref_pred, 'content')

# paired-wise cosine similarity
cosine_sim_pred_refer = cosine_similarity(embedding_pred, embedding_refer=embedding_refer, embedding_cont=None)
cosine_sim_pred_cont = cosine_similarity(embedding_pred, embedding_refer=None, embedding_cont=embedding_cont)

# RDASS 계산
"""
- pred_df   : id, text(content), label(reference), predict(predict), 
            cosine_sim_pred_refer, cosine_sim_pred_cont, RDASS
            열로 구성된 데이터프레임
- avg_rdass : 예측 문장 각 rdass 점수의 전체 평균값
"""
pred_df, avg_rdass = RDASS(ref_pred, cosine_sim_pred_refer, cosine_sim_pred_cont)

"""
Inference Hypermarameters
- min_length = max_len // 3
- num_beam = 3

RDASS
- 0~1 사이의 숫자 값. 숫자가 높을수록 더 높은 성능
"""

# 최종 결과 출력
print(f"문장성능 평가결과 평균값            : {avg_rdass}")
print(f"코사인유사도(pred <-> reference) : {np.mean(cosine_sim_pred_refer)}")
print(f"코사인유사도(pred <-> content)   : {np.mean(cosine_sim_pred_cont)}")