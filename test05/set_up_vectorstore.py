# setup_vectorstore.py

from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# 1. CSV 파일에서 문서 로드
loader = CSVLoader(file_path="data\sample_1000_with_meta.csv", encoding="cp949")
pages = loader.load()
print(f"문서 수: {len(pages)}")

# 2. HuggingFace 임베딩 생성
embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

# 3. Chroma 벡터스토어에 문서 저장
vectorstore = Chroma.from_documents(pages, embeddings, persist_directory="./database")
print("벡터스토어 저장 완료!")