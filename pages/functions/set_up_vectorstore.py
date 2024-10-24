# setup_vectorstore.py
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from dotenv import load_dotenv
import pandas as pd

# 환경변수 로드
load_dotenv()

# 01. CSV 파일에서 문서 로드
loader = CSVLoader(file_path="data/sample_with_meta.csv",
                   encoding='cp949')
pages = loader.load()
print(f"문서 수: {len(pages)}")


# 02. HuggingFace Embedding 생성
embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask",)


# 03. Chroma vectorstore에 문서 저장
vectorstore = Chroma.from_documents(pages,
                                    embeddings,
                                    persist_directory="./database_1000")
print("벡터스토어 저장 완료!")