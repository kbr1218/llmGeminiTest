# chat.py
import streamlit as st
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
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
vectorstore = Chroma(persist_directory="./database_all_with_meta", embedding_function=embeddings)


## 3. 사용자 정보 기반 지역 필터링 ###
user_name = st.session_state.get('user_name', [])
user_age = st.session_state.get('age', [])
visit_dates = st.session_state.get('visit_dates', [])
visit_times = st.session_state.get('visit_times', [])
visit_region = st.session_state.get('region', [])

# 필터 조건 구성
region_filter = {"지역": {"$in": visit_region}}
print(f"필터링된 지역: {visit_region}")
print(f"필터 조건: {region_filter}")


## 4. 검색기 생성 ###
retriever = vectorstore.as_retriever(
    search_type="mmr",   
    search_kwargs={"k": 10,
                   "fetch_k": 10,
                   "filters": region_filter}
)


## 5. 프롬프트 템플릿 설정 (수정 필요: 날씨에 기반하여 대답하도록 수정) ###
template = """
[context]: {context}
---
[질의]: {query}
---
[예시 응답]
{visit_dates}에 {visit_region} 지역 방문을 계획하신 {user_age}에게 {visit_times}에 방문하기 좋은 맛집을 추천해드리겠습니다!
추천 기준:
 - 기온: {visit_dates}월 {visit_times}에 평균 기온과 유사한 시가에 방문객이 많았던 맛집을 우선 고려했습니다.
 - 시간대: {visit_times} 이용 건수 비중이 높은 가맹점을 우선 고려했습니다.  

추천 맛집:
- **가맹점명**: {visit_dates}월의 평균 기온은 xx.xx°이고, 특히, {visit_times}의 평균 기온은 xx.xx°입니다. {visit_times}의 이용 건수 비중이 xx.xx%로 높습니다. 월별/업종별 이용 건수 순위도 x위로 높은 편입니다.

----
[추가 정보]
당신은 주어진 [context]에 맞게 응답해야 합니다. If you can't find data, say you don't know.
you must fill "xx.xx" in example answer based on {visit_dates}, {visit_times} average temperature data or 월별_업종별_이용건수_순위, 시간별 이용건수_비중 data.
{visit_dates}와 {visit_region}, {visit_times}에 따라 맞춤형 맛집을 2개 또는 3개 추천하고, 이유를 데이터 기반으로 설명하세요, 데이터를 찾지 못한다면 통계를 생략해도 좋습니다.
추천 기준은 다음과 같습니다.
- 기온: 제공된 월별&시간별 평균 기온 데이터를 기반으로 비슷한 기온의 날에 방문객이 많았던 가맹점를 추천합니다.
- 시간대: 시간대별 이용 건수 비중이 높은 가맹점을 우선 추천합니다.

[데이터 설명]
{user_age}: 사용자의 연령대,
{visit_dates}: 사용자가 제주도를 방문하는 기간,
{visit_times}: 사용자가 맛집을 방문할 시간,
{visit_region}: 사용자가 방문하는 제주도 지역,
기준년월-2023년 1월~12월,
업종-요식관련 30개 업종으로 구분 (업종이 '커피'일 경우 '카페' 뜻함 ),
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
        "사용자가 사전에 제공한 데이터({user_age}, {visit_dates}, {start_month}, {end_month}, {visit_times}, {visit_region})를 기반으로 얘기하세요."
    )
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        max_tokens=5000,
        system_instruction=system_instruction
    )
    print("model loaded...")
    return model
model = load_model()



### 7. 검색 결과 병합 함수 ###
def merge_pages(pages):
    merged = "\n\n".join(page.page_content for page in pages)
    return merged


## 8. LangChain 체인 구성 ###
chain = (
    {"query": RunnablePassthrough(),
     "context": retriever | merge_pages,      # retriever로 검색된 문서를 merge_pages 함수에 전달
     "visit_dates":RunnablePassthrough(),
     "visit_times":RunnablePassthrough(),
     "visit_region":RunnablePassthrough(),
     "user_age":RunnablePassthrough(),
     "user_name":RunnablePassthrough()
    }
    | prompt
    | load_model()
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
        st.error("사용자 정보(나이, 방문 날짜, 시간, 지역)가 누락되었습니다. 사이드바에서 정보를 입력해 주세요.")
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
            assistant_response = chain.invoke(user_input+f"""
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
