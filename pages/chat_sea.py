# chat_sea.py
import os
import streamlit as st

from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# 이미지 링크 선언
botImgPath = 'https://raw.githubusercontent.com/kbr1218/streamlitTest/main/imgs/dolhareubang_sea.png'
seaImgPath = 'https://raw.githubusercontent.com/kbr1218/streamlitTest/main/imgs/sea_img.jpg'

# 페이지 제목 설정
st.set_page_config(page_title="제주도SEA", page_icon="🐬", layout="wide",
                   initial_sidebar_state='expanded')

from pages.subpages import sidebar
from pages.subpages import chat_search

# 사이드바
with st.sidebar:
    sidebar.show_sidebar()


### 00. 환경변수 로드 ###
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')


# 임베딩 및 벡터스토어 설정
EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-multitask"
VECTOR_DB_DIR = "./vector_database_sea"
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
vectorstore = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embedding_model)


# Google Gemini 모델 설정
def load_model():
    system_instruction = (
        "당신은 제주도 여행객을 위한 추천 챗봇입니다. "
        "사용자 질문에 적합한 정보를 제공하세요. 제공된 데이터만 활용하며, "
        "추측으로 답하지 않습니다. 데이터가 존재하는데, 없다고 답하지 않습니다."
    )
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        max_tokens=5000,
        system_instruction=system_instruction,
        api_key=GOOGLE_API_KEY
    )

# 새로운 프롬프트 템플릿
prompt_template = """
특정 데이터를 기반으로 '제주도 내 해수욕장'과, '해당 해수욕장 1km 이내의 근처 맛집'을 추천하는 전문 어시스턴트 '친절한 제주도SEA🏖️'입니다.
제공된 데이터를 사용하여 정확하게 답변합니다. 확실하지 않은 경우 모른다고 답변합니다.
- 사용자의 질문을 기억하고 멀티턴 방식으로 답변합니다.
- 사용자의 질문이 일반적인 추천 요청인 경우 아래의 구조화된 형식을 따르세요.
- 사용자의 질문이 특정 정보(예: "해당 월에 가장 따뜻한 해수욕장은 어디인가요?")를 요구하는 경우, 추가 형식이나 설명 없이 요청된 정보만 제공합니다.
- 사용자의 질문이 통계 데이터에 관한 것인 경우(예: "해당 월에 가장 따뜻한 해수욕장을 내림차순으로 5개만 알려주세요.") 구체적인 통계(필터링 후 정렬) 값을 직접적이고 명확하게 제공합니다.

항상 주어진 데이터 컨텍스트에 따라 답변을 엄격하게 작성하세요.

추천할 때는 사용자에게 입력 받은 날짜 정보중에 '월(Month)', '평균최고수온'/'평균최저수온'을 고려하여 , '해수욕장(이름)' 과 '해수욕장1km근방맛집'을 묶어서 최대 1~3곳을 추천하세요.

검색된 컨텍스트의 관련 해수욕장 정보 요약으로 시작하여 {visit_month}에 대해 해수욕장 및 레스토랑 추천을 계속해야 합니다.

다음 열은 최상의 권장 사항을 찾는 데 관련이 있습니다:
- ['월'] 칼럼: {{월}}
- ['평균최고수온'] 칼럼: {{평균최고수온}}
- ['평균최저수온'] 칼럼: {{평균최저수온}}

제공된 컨텍스트와 사용자 정보를 엄격하게 사용하세요:
[context]: {context}
[previous_chat_history]: {previous_chat_history}
---
[질의]: {query}
"""
prompt = ChatPromptTemplate.from_template(prompt_template)


# LangChain 체인 구성
rag_chain = (
    {
        "query": RunnablePassthrough(),
        "context": lambda q: vectorstore.similarity_search(q["query"], k=22),
        "visit_month": RunnablePassthrough(),
        "recommendations": RunnablePassthrough(),  # Ensure recommendations is passed
        "previous_chat_history": RunnablePassthrough()  # 추가된 필드 전달
    }
    | prompt
    | load_model()
    | StrOutputParser()
)


# Streamlit 상태 초기화
if "conversation" not in st.session_state:
    st.session_state["conversation"] = []
if "user_name" not in st.session_state:
    st.session_state["user_name"] = None
if "age" not in st.session_state:
    st.session_state["age"] = None
if "visit_dates" not in st.session_state:
    st.session_state["visit_dates"] = None
if "visit_times" not in st.session_state:
    st.session_state["visit_times"] = None
if "region" not in st.session_state:
    st.session_state["region"] = []
if "selected_option" not in st.session_state:
    st.session_state["selected_option"] = "제주도SEA 챗봇과 바로 대화하기"
if "context" not in st.session_state:
    st.session_state["context"] = ""
if "last_recommended_beach" not in st.session_state:
    st.session_state["last_recommended_beach"] = None

# 방문 월 계산 (visit_month)
visit_dates = st.session_state.get("visit_dates")
visit_month = visit_dates.month if visit_dates else None



### 10. Streamlit UI ###
st.subheader("🐬:blue[제주도 SEA]에게 질문하기")
st.caption("🚀 2024 빅콘테스트 (생성형 AI 분야) 팀: 헬로빅콘")
st.divider()

st.markdown(
    """
    안녕하세요😁 제주도 해수욕장 추천 챗봇 🐬:blue[**제주도 SEA**]입니다 :)  
    제주도 바다 수온을 기반으로 수영하기 좋은 **해수욕장**🏖️과 **물놀이 복장**🩱을 추천하고,  
    추천된 해수욕장 반경 1km 내 맛집을 추천해드립니다🍊  
    (맛집 데이터: 신한카드 제주 가맹점 이용 데이터)
    """
)

# 바다 이미지
seaImg = (f"""
<div>
    <img src="{seaImgPath}" alt="sea image" width=100%>
</div>
""")
st.markdown(seaImg, unsafe_allow_html=True)

say_hi_to_user_sea = """🐬 제주도 해수욕장에 대해 궁금한 점을 물어보세요.  
입력하신 월 정보를 토대로 해수욕장을 추천드리고 있어요 :)  
"""

chat_input = st.chat_input(
    placeholder="질문을 입력하세요. (예: 우도에 있는 해수욕장을 추천해줘)",
    max_chars=150,
)

if 'messages_sea' not in st.session_state:
    st.session_state["messages_sea"] = [
        {"role": "assistant", "content": say_hi_to_user_sea}
    ]

for message in st.session_state["messages_sea"]:
    role = "user" if message["role"] == "user" else "assistant"
    avatar = "🧑🏻" if role == "user" else botImgPath
    if role == "assistant":
        with st.chat_message(message['role'], avatar=botImgPath):
            st.markdown(message["content"])
    else:
        with st.chat_message(role, avatar=avatar):
            st.markdown(message["content"])

if chat_input:
    st.session_state["messages_sea"].append({"role": "user", "content": chat_input})
    with st.chat_message("user", avatar="🧑🏻"):
        st.markdown(chat_input)

    # 이전 대화 내용을 포함하여 추천 정보 생성
    previous_chat_history = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in st.session_state.get("messages_sea", [])]
    )

    with st.spinner("추천 정보를 생성 중..."):
        response = rag_chain.invoke({
            "query": chat_input,
            "visit_month": visit_month,
            "context": st.session_state["context"],
            "recommendations": "",  # 기본 값 설정
            "previous_chat_history": previous_chat_history,  # 추가된 필드
        })

        st.session_state["messages_sea"].append({"role": "assistant", "content": response})
        with st.chat_message("assistant", avatar=botImgPath):
            st.markdown(response)
