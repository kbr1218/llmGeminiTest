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

# 페이지 제목 설정
st.set_page_config(page_title="main", page_icon="💬", layout="wide",
                   initial_sidebar_state='expanded')

from pages.subpages import sidebar, chat_search
from pages.subpages.modal import more

# CSS 파일 불러오기
with open('style/chat_page.css', encoding='utf-8') as css_file:
    st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)

# 사이드바
with st.sidebar:
    sidebar.show_sidebar()


########################################
# 환경변수 로드
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# langsmith 추적 설정
logging.langsmith("bigcon_langchain_test")

# HuggingFace 임베딩 생성
embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

### 1. Chroma 벡터스토어 로드 (테스트용 database_1000에서 불러옴 나중에 수정 필요) ###
vectorstore = Chroma(persist_directory="./database_1000", embedding_function=embeddings)

### 2. 사용자 정보 기반 지역 필터링 ###
user_name = st.session_state.get('user_name', [])
user_age = st.session_state.get('age', [])
visit_dates = st.session_state.get('visit_dates', [])
visit_times = st.session_state.get('visit_times', [])
visit_region = st.session_state.get('region', [])

# 필터 조건 구성
region_filter = {
    "area": {"$in": visit_region}
}

### 3. 필터를 적용하여 검색기 생성 ###
retriever = vectorstore.as_retriever(search_type="mmr",
                                     search_kwargs={"k": 8,            # K: k개의 문서 검색
                                                    "fetch_k": 10,
                                                    "filters":region_filter}) 


### 4. 프롬프트 템플릿 설정 (수정 필요: 날씨에 기반하여 대답하도록 수정) ###
template = """
[context]: {context}
---
[질의]: {query}
---
[예시]
선택하신 제주도 [선택한 지역]에 위치한 맛집을 추천해드리겠습니다!
[선택한 방문 시간]에 방문할 만한 [아침식사] 맛집 찾으시는군요.  
[visit_dates의 month]의 오전의 평균 기온은 약 00.0도입니다.
[식당이름]의 [3월] 오전(5시-11시) 방문율은 약 00.00%로 높은 편입니다.

추천 이유:

추가 정보:
---
당신은 주어진 [context]와 필터 조건에 맞게 응답해야 합니다.
필터된 지역과 문서에 따라 맞춤형 맛집을 3~5개 추천하고, 이유를 데이터 기반으로 설명하세요.
"""

# 위의 [context] 정보 내에서 [질의]에 대해 답변 [예시]와 같이 술어를 붙여서 답하세요.
# 사용자가 구체적인 숫자를 제시하지 않았다면, 3-5개의 맛집을 추천해주세요.
# 'visit_region'은 area 변수를 기준으로 선택되었습니다. 
# 추천 이유는 구체적일 수록 좋습니다. 왜 사용자에게 이런 맛집을 추천했는지 비중 데이터를 근거로 설명해주세요.

prompt = ChatPromptTemplate.from_template(template)


### 5. Google Gemini 모델 생성 ###
@st.cache_resource
def load_model():
    system_instruction = "당신은 제주도 여행객에게 제주도 맛집을 추천하는 친절한 제주도°C 챗봇입니다. 거짓말을 할 수 없으며, 주어진 데이터를 기반으로 얘기하세요."
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                                   temperature=0.5,
                                   max_tokens=5000,
                                   system_instruction=system_instruction)
    print("model loaded...")
    return model
model = load_model()


### 6. 검색 결과 병합 함수 ###
def merge_pages(pages):
    merged = "\n\n".join(page.page_content for page in pages)
    return merged


### 7. LangChain 체인 구성 ###
chain = (
    {"query": RunnablePassthrough(),
     "context": retriever | merge_pages,    # retriever로 검색된 문서를 merge_pages 함수에 전달
     "user_name":RunnablePassthrough(),     # RunnablePassThrough: 값을 변경하지 않고 그대로 통과시킴
     "user_age":RunnablePassthrough(),
     "visit_times":RunnablePassthrough(),
     "visit_region": RunnablePassthrough(),
     "visit_dates": RunnablePassthrough(),
    }
    | prompt
    | load_model()
    | StrOutputParser()
)


### 8. streamlit UI ###
st.title("chat page")
st.divider()

user_input = st.chat_input(
    placeholder="질문을 입력하세요. (예: 추자도에 있는 맛집을 알려줘)",
    max_chars=150
)

chat_col1, search_col2 = st.columns([2, 1])

with chat_col1:
    # 대화 이력 초기화 및 첫 번째 메시지
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": """안녕하세요!  
            제주도의 지역/시간별 기온 데이터에 기반하여 인기있는 맛집을 찾아드릴 **친절한 제주도℃**입니다.  
            궁금한 게 있다면 언제든 질문해주세요."""}
        ]

    # 이전 채팅 기록 출력
    for message in st.session_state.messages:
        avatar = "😊" if message['role'] == 'user' else "🍊"
        with st.chat_message(message['role'], avatar=avatar):
            st.markdown(message['content'])

    # 사용자 입력
    if user_input :
        st.session_state.messages.append({"role":"user", "content":user_input})
        with st.chat_message("user", avatar="😊"):
            st.markdown(user_input)

        # 추천 생성 중 스피너
        with st.spinner("맛집 찾는 중..."):
            assistant_response = chain.invoke(user_input)

        # Assistant 응답 기록에 추가 및 출력
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        with st.chat_message("assistant", avatar="🍊"):
            st.markdown(assistant_response)

with search_col2:
    chat_search.show_search_restaurant()

    if st.button("지도로 확인하기"):
        more.show_more_modal()


    #-----------------------------------------------------------

    # if user_input:
    #     # 사용자 입력을 history에 저장
    #     st.session_state.chat_history.append({"role":"user", "content":user_input})
    #     with st.spinner("추천을 생성 중입니다..."):
    #         # 실시간 spinner로 중간 결과 출력
    #         message_placeholder = st.empty()
    #         full_response = ""
    #         pages = retriever.get_relevant_documents(user_input)   # 검색 결과
    #         context = merge_pages(pages)
    #         query = {"query":user_input, "context":context}

    #         # response_stream = chain.invoke_stream(query)    # 실시간 응답 받기

    #         # for chunk in response_stream:
    #         #     full_response += chunk.text
    #         #     message_placeholder.markdown(full_response)

    #         full_response = chain.invoke(query)
            
    #         # 최종 응답 저장
    #         st.session_state.chat_history.append({"role": "ai", "content": full_response})
    #         st.success("추천 결과:")
    #         st.write(full_response)
