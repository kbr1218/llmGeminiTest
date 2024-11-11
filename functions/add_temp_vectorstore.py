# add_temp_vectorstore.py
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader

from langchain_community.vectorstores import Chroma

from langchain.schema import Document 
from langchain.text_splitter import RecursiveCharacterTextSplitter

import pandas as pd

DB_PATH = './temperature_vectorstore'

### 01. 임베딩 모델 생성 ###
embeddings = HuggingFaceEmbeddings(model_name='jhgan/ko-sroberta-multitask')

### 02. 새로운 데이터 로드 ###
new_loader = CSVLoader('data\data_TEMP.csv', encoding='cp949')
new_docs = new_loader.load()
print(f"[TEMP] 문서의 수: {len(new_docs)}")

### 03. pandas로 새로운 데이터프레임 칼럼명 가져오기 ###
csv_path = 'data/data_TEMP.csv'
df_new = pd.read_csv(csv_path, encoding='cp949')

### 04. 메타데이터 추가 ###
new_docs = []
for _, row in df_new.iterrows():
  metadata = {
    '기준년월': row['기준년월'],
    '지역': row['지역'],
  }
  doc = Document(
    page_content=str(row.to_dict()),
    metadata=metadata
  )
  new_docs.append(doc)

print(f"[TEMP] 문서의 수: {len(new_docs)}")
print('[[TEMP] 메타데이터 예시]\n', new_docs[3].metadata)

### 05. 데이터 청크 나누기 ###
text_splitter = RecursiveCharacterTextSplitter(
  chunk_size=300, chunk_overlap=0
)
new_splits = text_splitter.split_documents(new_docs)
print("[TEMP] split된 문서의 수:", len(new_splits))

### 06. 기온 벡터스토어 생성 ###
temp_vectorstore = Chroma.from_documents(
  documents=new_splits,
  embedding=embeddings,
  persist_directory=DB_PATH,
)
print("기온 벡터스토어 저장 완료!")
