# tab_chat.py
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

# 환경변수 로드
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# langsmith 추적 설정
logging.langsmith("bigcon_langchain_test")

# HuggingFace 임베딩 생성
embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

### 1. Chroma 벡터스토어 로드 (테스트용 database_1000에서 불러옴 나중에 수정 필요) ###
vectorstore = Chroma(persist_directory="./database_1000", embedding_function=embeddings)

### 2. 검색기 생성 ###
retriever = vectorstore.as_retriever(search_type="mmr",
                                     search_kwargs={"k": 8, "fetch_k": 10})  # K: k개의 문서 검색



### 3. 프롬프트 템플릿 설정 (수정 필요: 날씨에 기반하여 대답하도록 수정) ###
template = """
[context]: {context}
---
[질의]: {query}
---
[예시]
제주도에 위치한 맛집입니다
**가게명**: 지역 구분, 추천 이유
---
제주도 내 핫플레이스 맛집을 추천하는 대화형 AI assistant 역할을 해주세요.
주어진 데이터는 신한카드에 등록된 가맹점 중 매출 상위 9,252개 요식업종(음식점, 카페 등)입니다.
데이터에 대한 메타 데이터는 첫번째와 두번째 행에 있습니다.

위의 [context] 정보 내에서 [질의]에 대해 답변 [예시]와 같이 술어를 붙여서 답하세요.
사용자가 구체적인 숫자를 제시하지 않았다면, 3-5개의 맛집을 추천해주세요.
'제주도 지역 구분'은 area 변수를 참고해서 답변해주세요. 
추천 이유는 구체적일 수록 좋습니다. 왜 사용자에게 이런 맛집을 추천했는지 비중 데이터를 근거로 설명해주세요.
"""
prompt = ChatPromptTemplate.from_template(template)



### 4. Google Gemini 모델 생성 ###
@st.cache_resource
def load_model():
    # ChatGoogleGenerativeAI.configure(api_key=GOOGLE_API_KEY)
    system_instruction = "당신은 제주도 여행객에게 제주도 맛집을 추천하는 친절한 제주도°C 챗봇입니다. 거짓말을 할 수 없으며, 주어진 데이터를 기반으로 얘기하세요."
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                                   temperature=0.5,
                                   max_tokens=5000,
                                   system_instruction=system_instruction)
    
    print("model loaded...")
    return model
model = load_model()


### 5. 검색 결과 병합 함수 ###
def merge_pages(pages):
    merged = "\n\n".join(page.page_content for page in pages)
    return merged


### 6. LangChain 체인 구성 ###
chain = (
    {"query": RunnablePassthrough(), "context": retriever | merge_pages}
    | prompt
    | load_model()
    | StrOutputParser()
)



### 7. streamlit UI ###
def show_tab_chat():
    st.subheader("gemini chatbot here")

    # 대화 이력 초기화
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # 이전 채팅 기록 출력
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    # 사용자 입력
    if user_input := st.chat_input("질문을 입력하세요. (예: 추자도 맛집을 추천해줘)"):
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


    #-----------------------------------------------------------

    # # 사용자 입력창
    # user_input = st.text_input("질문을 입력하세요",
    #                            placeholder="질문을 입력하세요. (예: 추자도 맛집을 추천해줘)",  # 예시도 수정 필요
    #                            label_visibility='collapsed')

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

    #         # answer = chain.invoke(user_input)
    #         # st.success("추천 결과:")
    #         # st.write(answer)

    # # 대화 이력 출력
    # for message in st.session_state.chat_history:
    #     role = "user" if message["role"] == "user" else "ai"
    #     with st.chat_message(role):
    #         st.markdown(message["content"])