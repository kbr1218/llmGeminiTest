# set_up_vectorstore_sea.py
import os
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 환경 설정
DATA_PATH = './vector_database_sea'
EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-multitask"

# 데이터 로드
df = pd.read_csv(DATA_PATH, encoding='cp949')

# 임베딩 모델 초기화
embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# 벡터화할 데이터 텍스트 준비
data_texts = [
    f"해수욕장: {row['해수욕장']}, 주소: {row['주소']}, 평균최고수온: {row['평균최고수온']}℃, 평균최저수온: {row['평균최저수온']}℃, "
    f"수영복 두께(최고): {row['최고수온_수영복두께']}, 수영복 설명(최고): {row['최고수온_수영복설명']}, "
    f"수영복 두께(최저): {row['최저수온_수영복두께']}, 수영복 설명(최저): {row['최저수온_수영복설명']}, "
    f"리뷰: {row['해수욕장리뷰']}, 맛집: {row['해수욕장1km근방맛집']}"
    for _, row in df.iterrows()
]

# 벡터 DB 생성 및 저장
try:
    vectorstore = Chroma(persist_directory=DATA_PATH, embedding_function=embeddings_model)
    vectorstore.add_texts(texts=data_texts, embeddings=[embeddings_model.embed_query(text) for text in data_texts])
    print("벡터 DB 생성 및 저장 완료.")
except Exception as e:
    print(f"벡터 DB 생성 중 오류 발생: {e}")
