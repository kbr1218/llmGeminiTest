# chat.py
import streamlit as st

from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate

from langchain_teddynote import logging
from dotenv import load_dotenv
import os

# 챗봇 이미지 링크 선언
botImgPath = 'https://raw.githubusercontent.com/kbr1218/streamlitTest/main/imgs/dolhareubang3.png'

# 페이지 제목 설정
st.set_page_config(page_title="chat", page_icon="💬", layout="wide",
                   initial_sidebar_state='expanded')

from pages.subpages import sidebar, chat_search

# 사이드바
with st.sidebar:
    sidebar.show_sidebar()


##########################
### 00. 환경변수 로드 ###
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
# langsmith 추적 설정
logging.langsmith("bigcon_langchain_test")


### 1. HuggingFace 임베딩 생성 ###
embeddings  = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")


## 2. Chroma 벡터스토어 로드 ###
vectorstore = Chroma(persist_directory="./sample_1500_vectorstore", embedding_function=embeddings)


## 3. 사용자 정보 기반 지역 필터링 ###
user_name = st.session_state.get('user_name', [])
user_age = st.session_state.get('age', [])
visit_dates = st.session_state.get('visit_dates', [])
visit_times = st.session_state.get('visit_times', [])
visit_region = st.session_state.get('region', [])


## 4. 검색기 생성 ###
retriever = vectorstore.as_retriever(
    search_type="mmr",   
    search_kwargs={"k": 10,              # 반환할 문서 수 (default: 4)
                   "fetch_k": 50,        # MMR 알고리즘에 전달할 문서 수
                   "lambda_mult": 0.5}   # 결과 다양성 조절 (default: 0.5)
)


## 5. 프롬프트 템플릿 설정 (수정 필요: 날씨에 기반하여 대답하도록 수정) ###
template = """
You are an assistant for question-answering tasks named '친절한 제주도°C', which recommends good restaurants in Jeju Island based on the given temperature and restaurant infomation data.

Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know. Recommend three palces maximum and keep the answer concise.

Restaurant Recommendations Criteria: restaurants with a similar temperature in the month of {visit_dates} and the time zone {visit_times} where the user wants to visit has a high percentage/priority of use.

[context]: {context}
---
[질의]: {query}
---
[예시 응답]
{user_name}님, {visit_dates}월 {visit_region} {visit_times} 맛집 추천드립니다!
{visit_dates}월 {visit_region}의 평균 기온은 [평균기온]°C이고, {visit_times} 시간대의 평균 기온은 [시간대 평균기온]°C입니다. 이 시간대 이용 건수 비중이 높은 곳을 고려하여 추천드릴게요.

추천 맛집:
1. [**가맹점명**]:
- {visit_dates}월 {visit_region} 지역의 평균 기온과 유사한 시기에 {visit_times} 시간대 이용 건수 비중이 [이용건수비중]%로 높았습니다.
- [월별-업종별 이용건수 순위]위를 기록했으며, 연령대 {user_age}의 방문 비율이 [연령별 회원수 비중]%로 비슷한 연령대 고객이 많이 찾습니다.

----
[데이터 설명]
{user_age}: 사용자의 연령대,
{visit_dates}: 사용자가 제주도를 방문하는 기간,
{visit_times}: 사용자가 맛집을 방문할 시간,
{visit_region}: 사용자가 방문하는 제주도 지역,
기준년월-2023년 1월~12월,
업종-요식관련 30개 업종으로 구분 (업종이 '커피'일 경우 '카페'를 뜻함 ),
지역-제주도를 10개의 지역으로 구분(동부/서부/남부/북부/산지/가파도/마라도/비양도/우도/추자도),
주소-가맹점 주소,
월별_업종별_이용건수_순위: 월별 업종별 이용건수 분위수 구간을 6개 구간으로 집계 시 해당 가맹점의 이용건수가 포함되는 분위수 구간 * 1:상위 10%이하 2:상위 10~25% 3:상위 25~50% 4:상위 50~75% 5:상위 75~90% 6:상위 90% 초과(하위 10%이하),
월별_업종별_이용금액_순위: 월별 업종별 이용금액 분위수 구간을 6개 구간으로 집계 시 해당 가맹점의 이용금액이 포함되는 분위수 구간 * 1:상위 10%이하 2:상위 10~25% 3:상위 25~50% 4:상위 50~75% 5:상위 75~90% 6:상위 90% 초과(하위 10%이하),
건당_평균_이용금액_순위: 월별 업종별 건당평균이용금액 분위수 구간을 6개 구간으로 집계 시 해당 가맹점의 건당 평균 이용금액이 포함되는 분위수 구간 * 1:상위 10%이하 2:상위 10~25% 3:상위 25~50% 4:상위 50~75% 5:상위 75~90% 6:상위 90% 초과(하위 10%이하),
현지인_이용_건수_비중: 고객 자택 주소가 제주도인 경우를 현지인으로 정의
"""
prompt = ChatPromptTemplate.from_template(template)


### 6. Google Gemini 모델 생성 ###
# @st.cache_resource
def load_model():
    system_instruction = (
        "당신은 제주도 여행객에게 제주도 맛집을 추천하는 '친절한 제주°C' 챗봇입니다. "
        "사용자가 사전에 제공한 데이터(사용자 이름: {user_name}, 연령대: {user_age}, 방문기간: {visit_dates}, 방문 시간대: {visit_times}, 방문 지역: {visit_region})를 기반으로 얘기하세요."
    )
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        max_tokens=5000,
        system_instruction=system_instruction
    )
    print("model loaded...")
    return model


### 7. 검색 결과 필터링 & 병합 함수 ###
# visit_region 데이터 필터링
def filter_results_by_region(docs, visit_region):
    return [doc for doc in docs if doc.metadata.get('지역') in visit_region]

def format_docs(docs):
  return "\n\n".join(doc.page_content for doc in docs)


## 8. LangChain 체인 구성 ###
rag_chain = (
  {"context": retriever
   | (lambda docs: filter_results_by_region(docs, visit_region))
   | format_docs(),
    "query":RunnablePassthrough(),
    "user_name":RunnablePassthrough(),
    "user_age":RunnablePassthrough(),
    "visit_dates":RunnablePassthrough(),
    "visit_times":RunnablePassthrough(),
    "visit_region":RunnablePassthrough()
  }
  # question(사용자의 질문) 기반으로 연관성이 높은 문서 retriever 수행 >> format_docs로 문서를 하나로 만듦
  | prompt               # 하나로 만든 문서를 prompt에 넘겨주고
  | load_model()         # llm이 원하는 답변을 만듦
  | StrOutputParser()
)


### 9. Streamlit UI ###
st.subheader("🍊:orange[제주°C]에게 질문하기")
st.divider()

user_input = st.chat_input(
    placeholder="질문을 입력하세요. (예: 추자도에 있는 가정식 맛집을 추천해줘)",
    max_chars=150
)

chat_col1, search_col2 = st.columns([2, 1])
with search_col2:
    chat_search.show_search_restaurant()

    # 채팅 기록 초기화
    if st.button("채팅 기록 초기화", type='primary'):
        st.session_state.messages = [
            {"role": "assistant", "content": """안녕하세요!  
            제주도의 지역/시간별 기온 데이터에 기반하여 맛집을 추천하는 :orange[**친절한 제주°C**]입니다.  
            언제든지 질문해주세요."""}
        ]
        st.rerun()

with chat_col1:
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": """안녕하세요!  
            제주도의 지역/시간별 기온 데이터에 기반하여 맛집을 추천하는 :orange[**친절한 제주°C**]입니다.  
            언제든지 질문해주세요."""}
        ]
    # 필수 정보가 입력되지 않았을 경우 오류 메시지 출력
    if not (user_age and visit_dates and visit_times and visit_region):
        st.error("사용자 정보(연령대, 방문 날짜, 시간, 지역)가 누락되었습니다. \n왼쪽 사이드바에서 정보를 입력해 주세요.")
        st.stop()  # 이후 코드를 실행하지 않도록 중단


    for message in st.session_state.messages:
        avatar = "🧑🏻" if message['role'] == 'user' else botImgPath
        with st.chat_message(message['role'], avatar=avatar):
            st.markdown(message['content'])

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑🏻"):
            st.markdown(user_input)

        # 추천 생성 중 스피너
        with st.spinner("맛집 찾는 중..."):
            # chain.invoke에서 개별 변수로 전달
            assistant_response = rag_chain.invoke(user_input+f"""
                                              user_name: {user_name},
                                              user_age: {user_age},
                                              visit_region: {visit_region},
                                              visit_dates: {visit_dates},
                                              visit_times: {visit_times},
                                              chat_history: {st.session_state.messages}
                                              """)

        # Assistant 응답 기록에 추가 및 출력
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        with st.chat_message("assistant", avatar=botImgPath):
            st.markdown(assistant_response)  
