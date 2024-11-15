# chat_sea.py
import streamlit as st

from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate

from functions import load_model

# 이미지 링크 선언
botImgPath = 'https://raw.githubusercontent.com/kbr1218/streamlitTest/main/imgs/jejudoSea.png'
seaImgPath = 'https://raw.githubusercontent.com/kbr1218/streamlitTest/main/imgs/sea_img.jpg'

# 페이지 제목 설정
st.set_page_config(page_title="제주도SEA", page_icon="🐬", layout="wide",
                   initial_sidebar_state='expanded')

# 제주'C -> 제주도SEA 화면 전환 버튼 상태 변경
if 'jeju_sea' not in st.session_state:
    st.session_state['jeju_sea'] = True

from pages.subpages import sidebar
from pages.subpages import chat_search

# 사이드바
with st.sidebar:
    sidebar.show_sidebar()


### 01. 임베딩 및 벡터스토어 설정 ###
EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-multitask"
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
vectorstore = Chroma(persist_directory='./vector_database_sea', embedding_function=embedding_model)


### 02. Google Gemini 모델 설정 ###
system_instruction = """당신은 제주도 여행객을 위한 추천 챗봇입니다.
사용자 질문에 적합한 정보를 제공하세요. 제공된 데이터만 활용하며, 추측으로 답하지 않습니다. 데이터가 존재하면 없다고 답하지 않습니다."""
llm = load_model.load_gemini(system_instruction)


### 03. 프롬프트 템플릿 설정 ###
prompt_template = """
안녕하세요. {user_name}님, **{visit_month}** 제주도 해수욕장 정보를 답변 드릴게요.

특정 데이터를 기반으로 '제주도 내 해수욕장'과, '해당 해수욕장 1km 이내의 근처 맛집'을 추천하는 전문 어시스턴트 '친절한 제주도SEA🏖️'입니다.
제공된 데이터를 사용하여 정확하게 답변합니다. 확실하지 않은 경우 모른다고 답변합니다.

- 사용자의 질문을 기억하고 멀티턴 방식으로 답변합니다.
- 사용자의 질문이 일반적인 추천 요청인 경우 아래의 구조화된 형식을 따르세요.
- 사용자의 질문이 여러 개의 추천 정보를 요청하는 경우 (예: "제주도 남부에 위치한 해수욕장 3개를 추천해줘") 각 추천 정보를 **번호 형식으로 나열**하여 명확히 표현하세요.
- 사용자의 질문이 특정 정보(예: "해당 월에 가장 따뜻한 해수욕장은 어디인가요?")를 요구하는 경우, 추가 형식이나 설명 없이 요청된 정보만 제공합니다.
- 사용자의 질문이 통계 데이터에 관한 것인 경우(예: "해당 월에 가장 따뜻한 해수욕장을 내림차순으로 5개만 알려주세요.") 구체적인 통계(필터링 후 정렬) 값을 직접적이고 명확하게 제공합니다.

- 사용자의 정보는 다음과 같습니다
  1. 사용자 이름: {user_name}
  2. 방문할 월: {visit_month}

### 일반적인 권장 사항을 위한 구조화된 형식:
**{user_name}** 님! {visit_month}에 가장 따뜻한 바다를 추천드리겠습니다!  
🌡️ {{visit_month}}월의 {{beach}}의 {{visit_region}}의 평균 최고 수온은 **{{avg_high_temp}}** ℃ 입니다. 여행에 참고하시길 바랍니다.

**{{beach}}**:
- 🛶주소: {{address}} [위경도: {{latitude}}, {{longitude}}]
- ✍️수온측정위치: {{measurement_location}}
- 🌡️평균최고수온: {{user_name}}님이 방문하시려는 **{{visit_month}}** 에는 {{avg_high_temp}} ℃ 입니다.
- 🥽최고수온 기준 권장 수영복: {{high_temp_swimsuit_thickness}}, {{high_temp_swimsuit_details}}
- 🌡️평균최저수온: {{user_name}}님이 방문하시려는 **{{visit_month}}** 에는 {{avg_low_temp}} ℃ 입니다.
- 🥽최저수온 기준 권장 수영복: {{low_temp_swimsuit_thickness}}, {{low_temp_swimsuit_details}}
- ✅해수욕장 리뷰: {{beach_reviews}}
- ❤️해수욕장 1km 이내 맛집 추천: {{beach}}과 가까운 곳에 **{{nearby_restaurants}}**이(가) 있습니다.

😁 즐거운 여행 되시길 바랍니다!"

### 여러 추천 정보를 요청한 경우의 구조화된 형식:
사용자가 여러 개의 추천을 요청한 경우, 다음 형식으로 답변하세요:

1. **{{beach_1}}**:
   - 🛶주소: {{address}}
   - 🌡️평균 최고 수온: **{{avg_high_temp}}** ℃
   - ❤️1km 이내 맛집: {{nearby_restaurants}}

2. **{{beach_2}}**:
   - 🛶주소: {{address}}
   - 🌡️평균 최고 수온: **{{avg_high_temp}}** ℃
   - ❤️1km 이내 맛집: {{nearby_restaurants}}

3. **{{beach_3}}**:
   - 🛶주소: {{address}}
   - 🌡️평균 최고 수온: **{{avg_high_temp}}** ℃
   - ❤️1km 이내 맛집: {{nearby_restaurants}}

### 추가 지시 사항:
- 질문에 맞는 정보를 정확히 추출하여 제공하세요.
- 컨텍스트와 사용자의 정보를 반드시 활용하여 답변을 구성하세요:
[context]: {context}
[previous_chat_history]: {previous_chat_history}
---
[질의]: {query}
"""
prompt = ChatPromptTemplate.from_template(prompt_template)


### 04. LangChain 체인 구성 ###
rag_chain = (
    {
        "query": RunnablePassthrough(),
        "context": lambda q: vectorstore.similarity_search(q["query"], k=22),
        "user_name": RunnablePassthrough(),
        "visit_month": RunnablePassthrough(),
        "recommendations": RunnablePassthrough(),
        "previous_chat_history": RunnablePassthrough()  # 추가된 필드 전달
    }
    | prompt
    | llm
    | StrOutputParser()
)


### 05. Streamlit 상태 초기화 ###
user_name = st.session_state.get('user_name', '사용자')
user_age = st.session_state.get('age', None)
visit_times = st.session_state.get('visit_times', None)
visit_region = st.session_state.get('region', [])
visit_dates = st.session_state.get('visit_dates', None)
# 월 정보만 출력
visit_month = f"{visit_dates.month}월" if visit_dates else ""

# if "conversation" not in st.session_state:
#     st.session_state["conversation"] = []
# if "user_name" not in st.session_state:
#     st.session_state["user_name"] = "사용자"
# if "age" not in st.session_state:
#     st.session_state["age"] = None
# if "visit_dates" not in st.session_state:
#     st.session_state["visit_dates"] = None
# if "visit_times" not in st.session_state:
#     st.session_state["visit_times"] = None
# if "region" not in st.session_state:
#     st.session_state["region"] = []
# if "context" not in st.session_state:
#     st.session_state["context"] = ""
# if "last_recommended_beach" not in st.session_state:
#     st.session_state["last_recommended_beach"] = None

# 방문 월 계산 (visit_month)
# visit_dates = st.session_state.get("visit_dates")
# visit_month = visit_dates.month if visit_dates else None



### 06. Streamlit UI ###
st.subheader("🐬:blue[제주도 SEA]에게 질문하기")
st.caption("🚀 2024 빅콘테스트 (생성형 AI 분야) 팀: 헬로빅콘")
st.divider()

say_hi_to_user_sea = """🐬 제주도 해수욕장에 대해 궁금한 점을 물어보세요.  
사전에 입력하신 **방문 일자** 정보를 토대로 해수욕장을 추천드리고 있어요 :)"""

chat_input = st.chat_input(
    placeholder="질문을 입력하세요. (예: 우도에 있는 해수욕장을 추천해줘)",
    max_chars=150,
)

chat_col1, search_col2 = st.columns([2, 1])
with search_col2:
    chat_search.show_search_restaurant()

    # 채팅 기록 초기화
    if st.button("채팅 기록 초기화", type='primary'):
        st.session_state["messages_sea"] = [
            {"role": "assistant", "content": say_hi_to_user_sea}
        ]
        st.rerun()

with chat_col1:
    # 필수 정보가 입력되지 않았을 경우 오류 메시지 출력
    if not (user_age and visit_dates and visit_times and visit_region):
        st.error("사용자 정보(연령대, 방문 날짜, 시간, 지역)가 누락되었습니다. \n왼쪽 사이드바에서 정보를 입력해 주세요.", icon=":material/error:")
        st.stop()  # 이후 코드를 실행하지 않도록 중단

    st.markdown(
        """안녕하세요😁 제주도 해수욕장 추천 챗봇 🐬:blue[**제주도 SEA**]입니다 :)  
        제주도 바다 수온을 기반으로 수영하기 좋은 **해수욕장**🏖️과 **물놀이 복장**🩱을 추천하고,  
        추천된 해수욕장 반경 1km 내 맛집을 추천해드립니다🍴  
        :gray[(맛집 데이터: 신한카드 제주 가맹점 이용 데이터)]
        """
    )

    # 바다 이미지
    seaImg = (f"""
    <div>
        <img src="{seaImgPath}" alt="sea image" width=100%>
    </div>""")
    st.markdown(seaImg, unsafe_allow_html=True)

    if 'messages_sea' not in st.session_state:
        st.session_state["messages_sea"] = [
            {"role": "assistant", "content": say_hi_to_user_sea}
        ]

    for message in st.session_state["messages_sea"]:
        role = "user" if message["role"] == "user" else "assistant"
        avatar = "🧑🏻" if role == "user" else botImgPath
        if role == "assistant":
            with st.chat_message(message['role'], avatar=avatar):
                st.markdown(message["content"])
        else:
            with st.chat_message(role, avatar=avatar):
                st.markdown(message["content"])

    if chat_input:
        st.session_state["messages_sea"].append({"role": "user", "content": chat_input})
        with st.chat_message("user", avatar="🧑🏻"):
            st.markdown(chat_input)

        # 이전 대화 내용을 문자열로 변환 후 다음 추천 정보 생성에 반영
        previous_chat_history = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in st.session_state.get("messages_sea", [])]
        )

        with st.spinner("추천 정보를 생성 중..."):
            response = rag_chain.invoke({
                "query": chat_input,
                "visit_month": visit_month,
                "user_name": user_name,
                # "context": st.session_state["context"],
                "recommendations": "",  # 기본 값 설정
                "previous_chat_history": previous_chat_history,  # 추가된 필드
            })

            st.session_state["messages_sea"].append({"role": "assistant", "content": response})
            with st.chat_message("assistant", avatar=botImgPath):
                st.markdown(response)
