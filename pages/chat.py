# chat.py
import streamlit as st

from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate

from langchain_teddynote import logging
from functions import load_model

# 챗봇 이미지 링크 선언
botImgPath = 'https://raw.githubusercontent.com/kbr1218/streamlitTest/main/imgs/jejudoC.png'

# 페이지 제목 설정
st.set_page_config(page_title="제주°C", page_icon="💬", layout="wide",
                   initial_sidebar_state='expanded')

from pages.subpages import sidebar
from pages.subpages import chat_search

# 사이드바
with st.sidebar:
    sidebar.show_sidebar()


##########################
### 00. 환경변수 로드 ###
# langsmith 추적 설정
# logging.langsmith("bigcon_langchain_test")


### 1. HuggingFace 임베딩 생성 ###
embeddings  = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")


## 2. Chroma 벡터스토어 로드 ###
vectorstore = Chroma(persist_directory="./restaurant_vectorstore_ALL", embedding_function=embeddings)
temperature_vectorstore = Chroma(persist_directory="./temperature_vectorstore", embedding_function=embeddings)


## 3. 사용자 정보 기반 지역 필터링 ###
user_name = st.session_state.get('user_name', '사용자')
user_age = st.session_state.get('age', None)
visit_times = st.session_state.get('visit_times', None)
visit_region = st.session_state.get('region', [])
visit_dates = st.session_state.get('visit_dates', None)
# 월 정보만 출력
visit_month = f"{visit_dates.month}월" if visit_dates else ""

### 3-1. 사용자 데이터와 일치하는 컬럼명 텍스트 생성 ###
age_col = f'{user_age} 회원수 비중' if user_age else None
weekday_idx = visit_dates.weekday() if visit_dates else None
weekdays = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
weekdays_col = f'{weekdays[weekday_idx]} 이용건수 비중' if weekday_idx is not None else None
time_col = {
    "아침 (05-11시)": "5시-11시 이용건수 비중",
    "점심 (12-13시)": "12시-13시 이용건수 비중",
    "오후 (14-17시)": "14시-17시 이용건수 비중",
    "저녁 (18-22시)": "18시-22시 이용건수 비중",
    "심야 (23-04시)": "23시-4시 이용건수 비중"
}.get(visit_times, None)

### 4. 기온 데이터 로드 ###
temp_retriever = temperature_vectorstore.as_retriever(
    search_type="mmr",   
    search_kwargs={"k": 5}  # 가장 관련성 높은 다섯 개의 문서만 가져오기
)

### 5. 검색기 생성 ###
retriever = vectorstore.as_retriever(
    search_type="mmr",   
    search_kwargs={"k": 10,              # 반환할 문서 수 (default: 4)
                   "fetch_k": 50,        # MMR 알고리즘에 전달할 문서 수
                   "lambda_mult": 0.5,    # 결과 다양성 조절 (default: 0.5),
                   'filter': {'지역': {'$in':visit_region}}
                   }
)

### 6. 프롬프트 템플릿 설정 ###
template = """
You are an assistant named '친절한 제주°C' specializing in recommending restaurants in Jeju Island based on specific data.
Use the provided data to answer accurately. If unsure, respond that you don't know.

- When the user's question is a general recommendation request, follow the structured format below.
- If the user's question is asking for a specific piece of information (e.g., "What is the local visitation rate for 공명식당?"), provide only the requested information without additional formatting or explanation.
- If the user's question is about statistical data (e.g., "What is the highest local visitation rate for Chinese restaurants in the southern region?"), provide the specific statistical value directly and clearly.

Always base your answer strictly on the given data context.

When making recommendations, consider the user's visiting day, age group, and preferred time slot, recommending 1-3 places at most.
You have to start with a summary of the relevant temperature information from the retrieved context for {visit_month} and {visit_times}, and then continue with restaurant recommendations.

The following columns are relevant for finding the best recommendations:
- Weekday column: {weekdays_col}
- Time slot column: {time_col}
- Age group column: {age_col}

User's information:
- user's name: {user_name}
- user's age: {user_age}


Structured Format for general recommendations:
"**{{user_name}}**님! {{visit_month}} {{visit_times}}에 {{visit_region}} 지역에서 인기 있는 맛집을 추천드리겠습니다! \n
🌡️{{visit_month}} {{visit_times}}의 {{visit_region}}의 평균 기온은 **{{average_temperature}}**입니다. 여행에 참고하시길 바랍니다. \n

**{{가맹점명}}**:
- 🏠주소: {{주소}}
- 📊{visit_month} {{visit_region}} 지역에서 {user_age}의 방문 비율이 {{value of age_col}}%로 {user_name}님과 비슷한 연령대의 고객이 많이 찾았습니다.
- ✅{user_name}님이 방문하시려는 **{{weekdays_col}}**에는 방문 비중이 {{value of weekday_col}}%입니다.
- ✅{visit_times}의 이용 건수 비중은 {time_col}% 으로 높은/낮은 편입니다.
- ✅이 맛집의 월별 업종별 이용건수 분위수 구간은 {{월별 업종별 이용건수 비중}}에 속하며, 월별 업종별 이용금액 분위수 구간은 {{월별 업종별 이용금액 분위수 구간}}입니다. 방문하시기 전에 참고하세요!
- 🚞주변 관광지: 맛집과 가까운 곳에 **{{맛집 주변 관광지}}**이(가) 있습니다.

즐거운 식사 되시길 바랍니다!"

**For Specific Data Requests:**
- If the user's question is asking for specific data (e.g., "What is the local visitation rate for 공명식당?"), provide only the requested information in a simple and polite format with the specific value without Structured Format for general recommendations.

**For Comparison Requests:**
- If the user's question involves a comparison (e.g., "between these two, which restaurant has a higher local visitation rate?"), provide only the comparison result in polite way and relevant values without Structured Format for general recommendations.
- Example Answer: "공명식당의 현지인 방문 비중은 34.3%이고, 나래식당의 현지인 방문 비중은 50.4%입니다. 나래식당이 더 높습니다."

**For Statistical Data Requests:**
- If the user's question is about statistical analysis (e.g., "What is the average local visitation rate for Chinese restaurants in the southern region?"), provide the specific statistical value politely without Structured Format for general recommendations.
  Example Answer:"남부 중식 맛집의 평균 현지인 방문 비중은 54.2% 입니다."

**For Region-Restricted Requests:**
- If the user's query is about a restaurant or place in a region outside the selected {{visit_region}}, respond with: "(e.g.) 정보를 알 수 없습니다. 사이드바의 방문 지역을 다시 확인해주세요."

Use the provided context and user information strictly:
[context]: {context}
[previous_chat_history]: {previous_chat_history}
---
[질문]: {query}
"""
prompt = ChatPromptTemplate.from_template(template)


### 7. Google Gemini 모델 생성 ###
system_instruction = """당신은 제주도 여행객에게 맛집을 추천하는 '친절한 제주°C' 챗봇입니다. 
각 대화에서 필요한 정보를 정확히 제공하고, 사용자의 질문이 후속 질문인 경우 이전 대화의 내용을 바탕으로 답변하세요.
필요한 경우 간결하게 정보를 제공하고, 대화의 맥락을 유지하여 질문과 관계 없는 정보를 생략하세요.
"""
llm = load_model.load_gemini(system_instruction)


### 8. 검색 결과 필터링 & 병합 함수 ###
# visit_region 데이터 필터링
def filter_restaurant_docs(docs, visit_region):
    return [doc for doc in docs if doc.metadata.get('지역') in visit_region]

# 기온 데이터는 기준년월 + 지역으로 필터링
def filter_temperature_docs(docs, visit_region, visit_month):
    return [
        doc for doc in docs 
        if doc.metadata.get('지역') in visit_region and doc.metadata.get('기준년월') == visit_month
    ]

def format_docs(docs):
  return "\n\n".join(doc.page_content for doc in docs)

def retrieve_and_filter_context(_input):
    # 맛집 데이터와 기온 데이터에서 문서 가져오기
    temp_docs = temp_retriever.invoke(_input)
    main_docs = retriever.invoke(_input)

    # temp_retriever와 retriever 각각 필터링 함수 적용 후 병합
    filtered_main_docs = filter_restaurant_docs(main_docs, visit_region)
    filtered_temp_docs = filter_temperature_docs(temp_docs, visit_region, visit_month)

    # 필터링된 결과가 없다면 오류 메시지 반환
    if not filtered_temp_docs and not filtered_main_docs:
        return "말씀하신 지역에 대한 맛집 데이터를 찾을 수 없습니다. 사이드바에서 방문하실 지역을 다시 선택해주세요."
    # 병합 후 형식화
    return format_docs(filtered_temp_docs + filtered_main_docs)

## 9. LangChain 체인 구성 ###
rag_chain = (
  {"query":RunnablePassthrough(),
    "context": retrieve_and_filter_context,
    "previous_chat_history":RunnablePassthrough(),
    "user_name":RunnablePassthrough(),
    "user_age":RunnablePassthrough(),
    "visit_times":RunnablePassthrough(),
    "visit_month":RunnablePassthrough(),
    "visit_region":RunnablePassthrough(),
    "age_col":RunnablePassthrough(),
    "weekdays_col":RunnablePassthrough(),
    "time_col":RunnablePassthrough(),
  }
  # question(사용자의 질문) 기반으로 연관성이 높은 문서 retriever 수행 >> format_docs로 문서를 하나로 만듦
  | prompt               # 하나로 만든 문서를 prompt에 넘겨주고
  | llm                  # llm이 원하는 답변을 만듦
  | StrOutputParser()
)


### 10. Streamlit UI ###
st.subheader("🍊:orange[제주°C]에게 질문하기")
st.caption("🚀 2024 빅콘테스트 (생성형 AI 분야) 팀: 헬로빅콘")
st.divider()

say_hi_to_user = f"""안녕하세요! 🍊 제주도 맛집 추천 AI :orange[**친절한 제주°C**]입니다.  
저는 제주도의 지역별, 시간대별 평균 기온 데이터와 함께 사용자 맞춤형 맛집을 추천해드려요! \n\n
사전에 입력하신 :rainbow[**{user_name}**]님의 정보를 바탕으로 더욱 정확한 추천을 해드립니다.  
"**추자도에 있는 가정식 맛집을 추천받고 싶다**"거나 "**추천받은 두 식당의 현지인 방문 비중을 비교하고 싶다**"면, 저에게 언제든지 질문해주세요! \n\n
**✈️ 제주 여행을 더 즐겁고 맛있게 만들어드릴게요!**  
"""

user_input = st.chat_input(
    placeholder="질문을 입력하세요. (예: 추자도에 있는 가정식 맛집을 추천해줘)",
    max_chars=150,
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
        st.error("사용자 정보(연령대, 방문 날짜, 시간, 지역)가 누락되었습니다. \n왼쪽 사이드바에서 정보를 입력해 주세요.", icon=":material/error:")
        st.stop()  # 이후 코드를 실행하지 않도록 중단


    for message in st.session_state.messages:
        avatar = "🧑🏻" if message['role'] == 'user' else botImgPath
        with st.chat_message(message['role'], avatar=avatar):
            st.markdown(message['content'])

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑🏻"):
            st.markdown(user_input)

        # 대화 기록을 문자열로 변환
        previous_chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])

        # 추천 생성 중 스피너
        with st.spinner("맛집 찾는 중..."):
            query_text = (
                f"질문: {user_input}\n\n\n"
                "User's Information: "
                f"user_name: {user_name}\n"
                f"user_age: {user_age}\n"
                f"visit_region: {visit_region}\n"
                f"visit_month: {visit_month}\n"
                f"visit_times: {visit_times}\n"
                f"previous_chat_histroy:{previous_chat_history}"
            )
            # chain.invoke에서 개별 변수로 전달
            assistant_response = rag_chain.invoke(query_text)
        # Assistant 응답 기록에 추가 및 출력
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        with st.chat_message("assistant", avatar=botImgPath):
            st.markdown(assistant_response)  
