# setup_vectorstore.py


from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader

from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS, Chroma
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import pandas as pd

# 환경변수 로드
load_dotenv()

# 00. 데이터 불러오기
data_df = pd.read_csv("data/sample_without_meta.csv", encoding='cp949')


# 02. 텍스트 벡터화 ('가맹점명'을 주요 텍스트로 사용)
data = data_df['가맹점명'].tolist()
meta_data = [
    {
        '기준년월': row['기준년월'],
        '가맹점명': row['가맹점명'],
        '업종': row['업종'],
        '5시_11시_평균기온': row['5시_11시_평균기온'],
        '12시_13시_평균기온': row['12시_13시_평균기온'],
        '14시_17시_평균기온': row['14시_17시_평균기온'],
        '18시_22시_평균기온': row['18시_22시_평균기온'],
        '23시_4시_평균기온': row['23시_4시_평균기온'],
        '월_평균기온': row['월_평균기온'],
        '지역': row['지역'],
        '주소': row['주소'],
        '월별_업종별_이용건수_순위': row['월별_업종별_이용건수_순위'],
        '월별_업종별_이용금액_순위': row['월별_업종별_이용금액_순위'],
        '건당_평균_이용금액_순위': row['건당_평균_이용금액_순위'],
        '월요일_이용_건수_비중': row['월요일_이용_건수_비중'],
        '화요일_이용_건수_비중': row['화요일_이용_건수_비중'],
        '수요일_이용_건수_비중': row['수요일_이용_건수_비중'],
        '목요일_이용_건수_비중': row['목요일_이용_건수_비중'],
        '금요일_이용_건수_비중': row['금요일_이용_건수_비중'],
        '토요일_이용_건수_비중': row['토요일_이용_건수_비중'],
        '일요일_이용_건수_비중': row['일요일_이용_건수_비중'],
        '5시_11시_이용_건수_비중': row['5시_11시_이용_건수_비중'],
        '12시_13시_이용_건수_비중': row['12시_13시_이용_건수_비중'],
        '14시_17시_이용_건수_비중': row['14시_17시_이용_건수_비중'],
        '18시_22시_이용_건수_비중': row['18시_22시_이용_건수_비중'],
        '23시_4시_이용_건수_비중': row['23시_4시_이용_건수_비중'],
        '현지인_이용_건수_비중': row['현지인_이용_건수_비중'],
        '남성_회원수_비중': row['남성_회원수_비중'],
        '여성_회원수_비중': row['여성_회원수_비중'],
        '20대_이하_회원수_비중': row['20대_이하_회원수_비중'],
        '30대_회원수_비중': row['30대_회원수_비중'],
        '40대_회원수_비중': row['40대_회원수_비중'],
        '50대_회원수_비중': row['50대_회원수_비중'],
        '60대_이상_회원수_비중': row['60대_이상_회원수_비중'],
    }
    for _, row in data_df.iterrows()
]


# embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask",
#                                    encode_kwargs={'normalize_embeddings':True}) # 임베딩을 정규화하여 모든 벡터가 같은 범위의 갖도록 함


# 01. Hugging Face 임베딩 모델 불러오기
embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

# 03. 벡터화 수행
# vectors = embedding_model.embed_documents(data)

# FAISS 벡터 스토어에 벡터와 메타 데이터 저장
vector_db = FAISS.from_texts(texts=data,
                             embedding=embedding_model,
                             metadatas=meta_data)

# 벡터 스토어 저장
vector_db.save_local("sample_1000_from_gpt")

# # 00. 데이터 불러오기
# data = pd.read_csv("data/sample_without_meta.csv", encoding='cp949')
# meta = pd.read_csv("data/data_ALL_only_metadata.csv", encoding='cp949')

# # 01. HuggingFace 임베딩 모델 불러오기
# embedding_model = HuggingFaceEmbeddings(model_name='jhgan/ko-sroberta-multitask')

# # 02. 데이터와 메타 정보를 함께 벡터화
# data_vector = data[:].tolist()
# meta_vector = meta[:].tolist()

# # 벡터화 작업 수행
# vectors = [embedding_model.embed(data_vector[i]) for i in range(len(data_vector))]

# # 03. FAISS로 벡터 데이터베이스 생성
# vector_db = FAISS.from_vectors(vectors, meta_vector)

# # 벡터DB 저장
# vector_db.sava("database_gpt_test")



# 1. CSV 파일에서 문서 로드
# loader = CSVLoader(file_path="data/sample_without_meta.csv",
#                    encoding="cp949",
#                    csv_args={
#                     'fieldnames': list(metadata_descriptions.keys()),
#                     'delimiter': ','
#                   },
#                   metadata_columns=list(metadata_descriptions.keys()))



# # 2. 로드한 문서에 메타데이터 설명 추가
# pages = loader.load()
# for page in pages:
#   descriptions = {key: metadata_descriptions[key] for key in page.metadata.keys() if key in metadata_descriptions}
#   page.metadata['descriptions'] = ", ".join([f"{k}: {v}" for k, v in descriptions.items()])
#   # page.metadata['descriptions'] = {key: metadata_descriptions[key] for key in page.metadata.keys() if key in metadata_descriptions}

# print(f"문서 수: {len(pages)}")



# # 2. HuggingFace 임베딩 생성
# embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask",
#                                    encode_kwargs={'normalize_embeddings':True}) # 임베딩을 정규화하여 모든 벡터가 같은 범위의 갖도록 함

# # 3. Chroma 벡터스토어에 문서 저장
# vectorstore = Chroma.from_documents(pages, embeddings, persist_directory="./database_1000")
# vectorstore.persist()
# print("벡터스토어 저장 완료!")