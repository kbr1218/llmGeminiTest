# chat.py
import streamlit as st
from datetime import datetime

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
vectorstore = Chroma(persist_directory="./sample_1000_vectorstore", embedding_function=embeddings)
temperature_vectorstore = Chroma(persist_directory="./temperature_vectorstore", embedding_function=embeddings)


## 3. 사용자 정보 기반 지역 필터링 ###
user_name = st.session_state.get('user_name', None)
user_age = st.session_state.get('age', None)
visit_times = st.session_state.get('visit_times', None)
visit_region = st.session_state.get('region', [])
visit_dates = st.session_state.get('visit_dates', None)
# 월 정보만 출력
visit_month = f"{visit_dates.month}월" if visit_dates else ""

### 3-1. 사용자 데이터와 일치하는 컬럼명 텍스트 생성 ###
age_col = f'{user_age} 회원수 비중'

weekday_idx = visit_dates.weekday()
weekdays = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
weekday_col = f'{weekdays[weekday_idx]} 이용건수 비중'

time_col = {
    "아침 (05-11시)": "5시-11시 이용건수 비중",
    "점심 (12-13시)": "12시-13시 이용건수 비중",
    "오후 (14-17시)": "14시-17시 이용건수 비중",
    "저녁 (18-22시)": "18시-22시 이용건수 비중",
    "심야 (23-04시)": "23시-4시 이용건수 비중"
    }.get(visit_times)

### 4. 기온 데이터 로드 ###
temp_retriever = temperature_vectorstore.as_retriever(
    search_type="mmr",   
    search_kwargs={"k": 5}  # 가장 관련성 높은 한 개의 문서만 가져오기
)

### 5. 검색기 생성 ###
retriever = vectorstore.as_retriever(
    search_type="mmr",   
    search_kwargs={"k": 10,              # 반환할 문서 수 (default: 4)
                   "fetch_k": 50,        # MMR 알고리즘에 전달할 문서 수
                   "lambda_mult": 0.5,    # 결과 다양성 조절 (default: 0.5),
                   'filter': {'지역': {'$in':visit_region}}
                   }
    # filters={"지역":visit_region}
)

# Do not include unnecessary information. 
### 6. 프롬프트 템플릿 설정 ###
template = """
You are an assistant named '친절한 제주°C' specializing in recommending restaurants in Jeju Island based on specific data.
Use the provided data to answer accurately. If unsure, respond that you don't know.

- When the user's question is a general recommendation request, follow the structured format below.
- If the user's question is asking for a specific piece of information (e.g., "What is the local visitation rate for 공명식당?"), provide only the requested information without additional formatting or explanation.
- Always base your answer strictly on the given data context.

When making recommendations, consider the user's visiting day, age group, and preferred time slot, recommending 1-3 places at most.
You have to start with a summary of the relevant temperature information from the retrieved context for {visit_month} and {visit_times}, and then continue with restaurant recommendations.

The following columns are relevant for finding the best recommendations:
- Weekday column: {{day_column}}
- Time slot column: {{time_column}}
- Age group column: {{age_column}}

**Structured Format** (for general recommendations):
"**{{user_name}}**님! {{visit_month}}월 {{visit_times}}에 {{visit_region}} 지역에서 인기 있는 맛집을 추천드리겠습니다! \n
{{visit_month}}월 {{visit_times}}의 {{visit_region}}의 평균 기온은 {{average_temperature}}입니다. 여행에 참고하시길 바랍니다. \n

**{{가맹점명}}**:
- 주소: {{주소}}
- {{visit_month}}월 {{visit_region}} 지역에서 {{user_age}}의 방문 비율이 {{age_col}}%로 {{user_name}}님과 비슷한 연령대의 고객이 많이 찾았습니다.
- {{user_name}}님이 방문하시려는 {{weekdays[weekday_idx]}}에는 방문 비중이 {{weekday_col}}%입니다.
- {{visit_times}}의 이용 건수 비중은 {{time_col}}% 으로 높은/낮은 편입니다.
- 이 맛집의 월별 업종별 이용건수 분위수 구간은 {{월별 업종별 이용건수 비중}}에 속하며, 월별 업종별 이용금액 분위수 구간는 {{월별 업종별 이용금액 분위수 구간}}입니다. 방문하시기 전에 참고하세요!
- 주변 관광지: 맛집과 가까운 곳에 **{{맛집 주변 관광지}}**이(가) 있습니다.

즐거운 식사 되시길 바랍니다!"

**When providing specific details for questions like:** "What is the local visitation rate for 공명식당?" Answer only the specific information, in simple and polite format with specific rate.

Use the provided context and user information strictly:
[context]: {context}
---
[질의]: {query}
"""
prompt = ChatPromptTemplate.from_template(template)


### 7. Google Gemini 모델 생성 ###
# @st.cache_resource
def load_model():
    system_instruction = """당신은 제주도 여행객에게 맛집을 추천하는 '친절한 제주°C' 챗봇입니다. 
        각 대화에서 필요한 정보를 정확히 제공하고, 사용자의 질문이 후속 질문인 경우 이전 대화의 내용을 바탕으로 답변하세요.
        필요한 경우 간결하게 정보를 제공하고, 대화의 맥락을 유지하여 질문과 관계 없는 정보를 생략하세요.
        """
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        max_tokens=5000,
        system_instruction=system_instruction
    )
    print("model loaded...")
    return model


### 8. 검색 결과 필터링 & 병합 함수 ###
# visit_region 데이터 필터링
def filter_results_by_region(docs, visit_region):
    return [doc for doc in docs if doc.metadata.get('지역') in visit_region]

def format_docs(docs):
  return "\n\n".join(doc.page_content for doc in docs)

def retrieve_and_filter_context(_input):
    # temp_retriever와 retriever 각각 호출 및 필터링 후 병합
    temp_docs = filter_results_by_region(temp_retriever.invoke(_input), visit_region)
    main_docs = filter_results_by_region(retriever.invoke(_input), visit_region)

    # 필터링된 결과가 없다면 오류 메시지 반환
    if not temp_docs and not main_docs:
        return "말씀하신 지역에 대한 맛집 데이터를 찾을 수 없습니다. 사이드바에서 방문하실 지역을 다시 선택해주세요."
    # 병합 후 형식화
    return format_docs(temp_docs + main_docs)

## 9. LangChain 체인 구성 ###
rag_chain = (
  {"query":RunnablePassthrough(),
    "context": retrieve_and_filter_context,
    "user_name":RunnablePassthrough(),
    "user_age":RunnablePassthrough(),
    "visit_times":RunnablePassthrough(),
    "visit_month":RunnablePassthrough(),
    "visit_region":RunnablePassthrough()
  }
  # question(사용자의 질문) 기반으로 연관성이 높은 문서 retriever 수행 >> format_docs로 문서를 하나로 만듦
  | prompt               # 하나로 만든 문서를 prompt에 넘겨주고
  | load_model()         # llm이 원하는 답변을 만듦
  | StrOutputParser()
)


### 10. Streamlit UI ###
st.subheader("🍊:orange[제주°C]에게 질문하기")
st.divider()

say_hi_to_user = """안녕하세요!  
제주도의 지역/시간별 기온 데이터에 기반하여 맛집을 추천하는 :orange[**친절한 제주°C**]입니다.  
언제든지 질문해주세요."""

user_input = st.chat_input(
    placeholder="질문을 입력하세요. (예: 추자도에 있는 가정식 맛집을 두 개만 추천해줘)",
    max_chars=150
)

chat_col1, search_col2 = st.columns([2, 1])
with search_col2:
    chat_search.show_search_restaurant()

    # 채팅 기록 초기화
    if st.button("채팅 기록 초기화", type='primary'):
        st.session_state.messages = [
            {"role": "assistant", "content": say_hi_to_user}
        ]
        st.rerun()

with chat_col1:
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": say_hi_to_user}
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
            query_text = (
                f"질문: {user_input}\n\n"
                f"user_name: {user_name}\n"
                f"user_age: {user_age}\n"
                f"visit_region: {visit_region}\n"
                f"visit_month: {visit_month}\n"
                f"visit_times: {visit_times}"
            )
            # chain.invoke에서 개별 변수로 전달
            assistant_response = rag_chain.invoke(query_text)

        # Assistant 응답 기록에 추가 및 출력
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        with st.chat_message("assistant", avatar=botImgPath):
            st.markdown(assistant_response)  
