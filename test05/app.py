# app.py
import streamlit as st
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# # 1. 초기화: CSV 파일에서 문서 로드 및 임베딩 처리
# loader = CSVLoader(file_path="../data/sample_1000_with_meta.csv", encoding="cp949")
# pages = loader.load()

# HuggingFace 임베딩 생성
embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

# 1. Chroma 벡터스토어 로드
vectorstore = Chroma(persist_directory="./database", embedding_function=embeddings)

# 2. 검색기 생성
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 10}) 
# K: k개의 문서 검색

# 3. 프롬프트 템플릿 설정
template = """
[context]: {context}
---
[질의]: {query}
---
[예시]
제주도에 위치한 맛집입니다
**가게명**: 제주도지역구분, 가게주소, 추천 이유
---
제주도 내 핫플레이스 맛집을 추천하는 대화형 AI assistant 역할을 해주세요.
주어진 데이터는 신한카드에 등록된 가맹점 중 매출 상위 9,252개 요식업종(음식점, 카페 등)입니다.
데이터에 대한 메타 데이터는 첫번째와 두번째 행에 있습니다.

위의 [context] 정보 내에서 [질의]에 대해 답변 [예시]와 같이 술어를 붙여서 답하세요.
사용자가 구체적인 숫자를 제시하지 않았다면, 중복되지 않는 3-5개의 맛집을 추천해주세요.
추천 이유는 구체적일 수록 좋습니다. 왜 사용자에게 이런 맛집을 추천했는지 비중 데이터를 근거로 설명해주세요.
"""
prompt = ChatPromptTemplate.from_template(template)

# 4. Google Gemini 모델 생성
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# 5. 검색 결과 병합 함수
def merge_pages(pages):
    merged = "\n\n".join(page.page_content for page in pages)
    return merged

# 6. LangChain 체인 구성
chain = (
    {"query": RunnablePassthrough(), "context": retriever | merge_pages}
    | prompt
    | llm
    | StrOutputParser()
)


# 7. Streamlit UI
st.set_page_config(page_title="친절한 제주℃", layout="wide")

st.title("gemini chatbot test")

# 사용자 입력창
user_input = st.text_input("질문을 입력하세요", placeholder="예: 추자도 맛집을 추천해줘")

if user_input:
    with st.spinner("추천을 생성 중입니다..."):
        answer = chain.invoke(user_input)
        st.success("추천 결과:")
        st.write(answer)

# 사이드바 정보
st.sidebar.title("📍 정보")
st.sidebar.write("이 챗봇은 제주도 맛집 데이터를 바탕으로 RAG 방식을 사용해 응답합니다.")
st.sidebar.write("**모델**: Google Gemini (Generative AI)")
st.sidebar.write("**벡터스토어**: Chroma")
st.sidebar.write("**임베딩**: ko-sroberta-multitask")