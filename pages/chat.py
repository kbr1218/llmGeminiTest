# chat.py
import streamlit as st
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
# from langchain_community.vectorstores import Chroma

from langchain_teddynote import logging
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from dotenv import load_dotenv
import os

# 페이지 제목 설정
st.set_page_config(page_title="main", page_icon="💬", layout="wide",
                   initial_sidebar_state='expanded')

from pages.subpages import sidebar, chat_search

# CSS 파일 불러오기
with open('style/chat_page.css', encoding='utf-8') as css_file:
    st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)

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
test_embedding = embeddings.embed_query("산지 맛집")


## 2. Chroma 벡터스토어 로드 ###
vectorstore = Chroma(persist_directory="./database_1000", embedding_function=embeddings)

## 3. 사용자 정보 기반 지역 필터링 ###
user_name = st.session_state.get('user_name', [])
user_age = st.session_state.get('age', [])
visit_dates = st.session_state.get('visit_dates', [])
visit_times = st.session_state.get('visit_times', [])
visit_region = st.session_state.get('region', [])

# 필터 조건 구성
region_filter = {"area": {"$in": visit_region}}
print(f"필터링된 지역: {visit_region}")
print(f"필터 조건: {region_filter}")

## 4. 필터를 적용하여 검색기 생성 ###
retriever = vectorstore.as_retriever(
    search_type="mmr",   
    search_kwargs={"k": 8, "fetch_k": 10, "filters": region_filter}
)

## 5. 프롬프트 템플릿 설정 (수정 필요: 날씨에 기반하여 대답하도록 수정) ###
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

추천 이유: {context}  # 문서에서 가져온 데이터를 포함

추가 정보:
당신은 주어진 [context]와 필터 조건에 맞게 응답해야 합니다.
필터된 지역과 문서에 따라 맞춤형 맛집을 3~5개 추천하고, 이유를 데이터 기반으로 설명하세요.
"""

prompt = ChatPromptTemplate.from_template(template)

### 6. Google Gemini 모델 생성 ###
@st.cache_resource
def load_model():
    system_instruction = (
        "당신은 제주도 여행객에게 제주도 맛집을 추천하는 친절한 제주도°C 챗봇입니다. "
        "거짓말을 할 수 없으며, 주어진 데이터를 기반으로 얘기하세요."
    )
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", temperature=0, max_tokens=5000, system_instruction=system_instruction
    )
    print("model loaded...")
    return model
model = load_model()

### 7. 검색 결과 병합 함수 ###
def merge_pages(pages):
    merged = "\n\n".join([page.page_content for page in pages if page.page_content])
    for page in pages:
        print(f"검색된 문서: {page.metadata['area']}")
    return merged

## 8. LangChain 체인 구성 ###
chain = (
    {"query": RunnablePassthrough(),
     "context": retriever | merge_pages,
     "user_name": RunnablePassthrough(),
     "user_age": RunnablePassthrough(),
     "visit_times": RunnablePassthrough(),
     "visit_region": RunnablePassthrough(),
     "visit_dates": RunnablePassthrough()}
    | prompt
    | load_model()
    | StrOutputParser()
)

### 9. Streamlit UI ###
st.title("chat page")
st.divider()

user_input = st.chat_input(
    placeholder="질문을 입력하세요. (예: 추자도에 있는 맛집을 알려줘)",
    max_chars=150
)

chat_col1, search_col2 = st.columns([2, 1])
with chat_col1:
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": """안녕하세요!  
            제주도의 지역/시간별 기온 데이터에 기반하여 인기있는 맛집을 찾아드릴 **친절한 제주도°C**입니다.  
            궁금한 게 있다면 언제든 질문해주세요."""}
        ]

    for message in st.session_state.messages:
        avatar = "😊" if message['role'] == 'user' else "🍊"
        with st.chat_message(message['role'], avatar=avatar):
            st.markdown(message['content'])

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="😊"):
            st.markdown(user_input)

        with st.spinner("맛집 찾는 중..."):
            assistant_response = chain.invoke(user_input)

        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        with st.chat_message("assistant", avatar="🍊"):
            st.markdown(assistant_response)

with search_col2:
    chat_search.show_search_restaurant()

### 01. FAISS 벡터스토어 로드 ###
# embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
# vector_db = FAISS.load_local("sample_1000_from_gpt",
#                              embedding_model,
#                              allow_dangerous_deserialization=True)


# ### 02. 사용자 정보 기반 지역 필터링 ###
# visit_region = st.session_state.get('region', [])

# # 메타 데이터에서 지역에 맞는 데이터 필터링
# def filter_by_region(region_list, vector_db):
#     if not region_list:
#         return list(vector_db.docstore._dict.values())  # 지역 선택이 없을 때 모든 데이터 반환
    
#     filtered_results = []
#     for doc in vector_db.docstore._dict.values():
#         if '지역' in doc.metadata and doc.metadata['지역'] in region_list:
#             filtered_results.append(doc)
#     return filtered_results

# # # 지역에 맞는 데이터 필터링 수행
# # filtered_docs = filter_by_region(visit_region, vector_db)

# # # 필터링된 결과를 기반으로 새 벡터 스토어 생성
# # if filtered_docs:
# #     # filtered_docs가 문자열 리스트일 경우 처리
# #     documents = [Document(page_content=doc, metadata={"지역": visit_region}) if isinstance(doc, str) else doc for doc in filtered_docs]
    
# #     filtered_vector_db = FAISS.from_documents(
# #         [doc.page_content for doc in documents],  # page_content는 문자열이 되어야 함
# #         embedding_model,
# #         metadatas=[doc.metadata for doc in documents]  # metadata도 각 문서의 메타데이터
# #     )
# # else:
# #     st.write("선택한 지역에 해당하는 데이터가 없습니다.")
# #     filtered_vector_db = None



# ### 03. 검색기 생성 ###
# def search_vector_db(query, vector_db):
#     if not query:
#         return "질문을 입력해주세요."
#     retriever = vector_db.as_retriever()
#     result = retriever.get_relevant_documents(query)
#     return result


# ### 04. 프롬프트 템플릿 설정 ###
# template = """
# [context]: {context}
# ---
# [질의]: {query}
# ---
# [예시]
# 선택하신 제주도 [선택한 지역]에 위치한 맛집을 추천해드리겠습니다!
# [선택한 방문 시간]에 방문할 만한 [아침식사] 맛집 찾으시는군요.  
# [visit_dates의 month]의 오전의 평균 기온은 약 00.0도입니다.
# [식당이름]의 [3월] 오전(5시-11시) 방문율은 약 00.00%로 높은 편입니다.
# 추천 이유:
# -------
# 추가 정보:
# 당신은 주어진 [context]와 필터 조건에 맞게 응답해야 합니다.

# 데이터 설명:
# 기준년월-2023년 1월~12월
# 업종-요식관련 30개 업종으로 구분
# 지역-제주도를 10개의 지역으로 구분(동부/서부/남부/북부/산지/가파도/마라도/비양도/우도/추자도)
# 주소-가맹점 주소
# 월별_업종별_이용건수_순위-월별 업종별 이용건수 분위수 구간을 6개 구간으로 집계 시 해당 가맹점의 이용건수가 포함되는 분위수 구간 * 1:상위10%이하 2:상위10~25% 3:상위25~50% 4:상위50~75% 5:상위75~90% 6:상위90% 초과(하위10%이하) * 상위 30% 매출 가맹점 내 분위수 구간임
# 월별_업종별_이용금액_순위-월별 업종별 이용금액 분위수 구간을 6개 구간으로 집계 시 해당 가맹점의 이용금액이 포함되는 분위수 구간 * 1:상위10%이하 2:상위10~25% 3:상위25~50% 4:상위50~75% 5:상위75~90% 6:상위90% 초과(하위10%이하) * 상위 30% 매출 가맹점 내 분위수 구간임
# 건당_평균_이용금액_순위-월별 업종별 건당평균이용금액 분위수 구간을 6개 구간으로 집계 시 해당 가맹점의 건당평균이용금액이 포함되는 분위수 구간 * 1:상위10%이하 2:상위10~25% 3:상위25~50% 4:상위50~75% 5:상위75~90% 6:상위90% 초과(하위10%이하) * 상위 30% 매출 가맹점 내 분위수 구간임
# 현지인_이용_건수_비중-고객 자택 주소가 제주도인 경우를 현지인으로 정의
# """
# prompt = ChatPromptTemplate.from_template(template)


# ### 05. Google Gemini 모델 생성 ###
# @st.cache_resource
# def load_model():
#     system_instruction = "당신은 제주도 여행객에게 제주도 맛집을 추천하는 친절한 제주도°C 챗봇입니다. 주어진 데이터를 기반으로 응답하세요."
#     model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2, max_tokens=5000, system_instruction=system_instruction)
#     print("model loaded...")
#     return model

# model = load_model()


# ### 06. 검색 결과 병합 함수 ###
# def merge_pages(pages):
#     merged = "\n\n".join([page.page_content for page in pages if page.page_content])
#     return merged


# ### 07. LangChain 체인 구성 ###
# chain = (
#     {"query": RunnablePassthrough(),
#      "context": search_vector_db | merge_pages,
#      "visit_region": RunnablePassthrough(),
#      "visit_dates": RunnablePassthrough(),
#     }
#     | prompt
#     | model
#     | StrOutputParser()
# )


# ### 8. streamlit UI ###
# st.title("chat page")
# st.divider()

# user_input = st.chat_input(
#     placeholder="질문을 입력하세요. (예: 추자도에 있는 맛집을 알려줘)",
#     max_chars=150
# )

# chat_col1, search_col2 = st.columns([2, 1])

# with chat_col1:
#     # 대화 이력 초기화 및 첫 번째 메시지
#     if 'messages' not in st.session_state:
#         st.session_state.messages = [
#             {"role": "assistant", "content": """안녕하세요!  
#             제주도의 지역/시간별 기온 데이터에 기반하여 인기있는 맛집을 찾아드릴 **친절한 제주도℃**입니다.  
#             궁금한 게 있다면 언제든 질문해주세요."""}
#         ]

#     # 이전 채팅 기록 출력
#     for message in st.session_state.messages:
#         avatar = "😊" if message['role'] == 'user' else "🍊"
#         with st.chat_message(message['role'], avatar=avatar):
#             st.markdown(message['content'])

#     # if filtered_vector_db:
#         # 사용자 입력
#         if user_input :
#             search_results = search_vector_db(user_input, filtered_vector_db)

#             st.session_state.messages.append({"role":"user", "content":user_input})
#             with st.chat_message("user", avatar="😊"):
#                 st.markdown(user_input)

#             # 추천 생성 중 스피너
#             with st.spinner("맛집 찾는 중..."):
#                 assistant_response = chain.invoke(user_input)

#             # Assistant 응답 기록에 추가 및 출력
#             st.session_state.messages.append({"role": "assistant", "content": assistant_response})
#             with st.chat_message("assistant", avatar="🍊"):
#                 st.markdown(assistant_response)

# with search_col2:
#     chat_search.show_search_restaurant()


# HuggingFace 임베딩 생성
# embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
# test_embedding = embeddings.embed_query("산지 맛집")

# ## 1. Chroma 벡터스토어 로드 (테스트용 database_1000에서 불러옴 나중에 수정 필요) ###
# vectorstore = Chroma(persist_directory="./database_1000", embedding_function=embeddings)

# ## 2. 사용자 정보 기반 지역 필터링 ###
# user_name = st.session_state.get('user_name', [])
# user_age = st.session_state.get('age', [])
# visit_dates = st.session_state.get('visit_dates', [])
# visit_times = st.session_state.get('visit_times', [])
# visit_region = st.session_state.get('region', [])

# 필터 조건 구성
# region_filter = {
#     "area": {"$in": visit_region}
# }

# print(f"필터링된 지역: {visit_region}")
# print(f"필터 조건: {region_filter}")

# ## 3. 필터를 적용하여 검색기 생성 ###
# retriever = vectorstore.as_retriever(search_type="mmr",   #"mmr"
#                                      search_kwargs={"k": 8,            # K: k개의 문서 검색
#                                                     "fetch_k": 10,
#                                                     "filters":region_filter}) 


# ## 4. 프롬프트 템플릿 설정 (수정 필요: 날씨에 기반하여 대답하도록 수정) ###
# template = """
# [context]: {context}
# ---
# [질의]: {query}
# ---
# [예시]
# 선택하신 제주도 [선택한 지역]에 위치한 맛집을 추천해드리겠습니다!
# [선택한 방문 시간]에 방문할 만한 [아침식사] 맛집 찾으시는군요.  
# [visit_dates의 month]의 오전의 평균 기온은 약 00.0도입니다.
# [식당이름]의 [3월] 오전(5시-11시) 방문율은 약 00.00%로 높은 편입니다.

# 추천 이유: {context}  # 문서에서 가져온 데이터를 포함

# 추가 정보:
# ---
# 당신은 주어진 [context]와 필터 조건에 맞게 응답해야 합니다.
# 필터된 지역과 문서에 따라 맞춤형 맛집을 3~5개 추천하고, 이유를 데이터 기반으로 설명하세요.

# 데이터에 대한 설명입니다. 사용자가 요청하는 질문에서 지역 정보를 찾아 area 변수에서 필터링한 후 답변하세요.
# YM: 기준연월(1월~12월), MCT_NM: 가맹점명, MCT_TYPE: 요식관련 30개 업종, temp_05_11: 5시 11시 평균 기온, temp_12_13: 12시 13시 평균 기온, temp_14_17: 14시 17시 평균 기온, temp_18_22: 18시 22시 평균 기온, temp_23_04: 23시 4시 평균 기온, TEMP_AVG: 월(YM) 평균 기온, area: 제주도를 10개의 지역으로 구분: 동부/서부/남부/북부/산지/가파도/마라도/비양도/우도/추자도, ADDR: 가맹점 주소, RANK_CNT: 월별 업종별 이용건수 분위수 구간을 6개 구간으로 집계 시 해당 가맹점의 이용건수가 포함되는 분위수 구간 * 1:상위10%이하 2:상위10~25% 3:상위25~50% 4:상위50~75% 5:상위75~90% 6:상위90% 초과(하위10%이하) * 상위 30% 매출 가맹점 내 분위수 구간임, RANK_AMT: 월별 업종별 이용금액 분위수 구간을 6개 구간으로 집계 시 해당 가맹점의 이용금액이 포함되는 분위수 구간 * 1:상위10%이하 2:상위10~25% 3:상위25~50% 4:상위50~75% 5:상위75~90% 6:상위90% 초과(하위10%이하) * 상위 30% 매출 가맹점 내 분위수 구간임, RANK_MEAN: 월별 업종별 건당평균이용금액 분위수 구간을 6개 구간으로 집계 시 해당 가맹점의 건당평균이용금액이 포함되는 분위수 구간 * 1:상위10%이하 2:상위10~25% 3:상위25~50% 4:상위50~75% 5:상위75~90% 6:상위90% 초과(하위10%이하) * 상위 30% 매출 가맹점 내 분위수 구간임, MON_UE_CNT_RAT: 월요일 이용 건수 비중, TUE_UE_CNT_RAT: 화요일 이용 건수 비중, WED_UE_CNT_RAT: 수요일 이용 건수 비중, THU_UE_CNT_RAT: 목요일 이용 건수 비중, FRI_UE_CNT_RAT: 금요일 이용 건수 비중, SAT_UE_CNT_RAT: 토요일 이용 건수 비중, SUN_UE_CNT_RAT: 일요일 이용 건수 비중, HR_5_11_UE_CNT_RAT: 5시-11시 이용 건수 비중, HR_12_13_UE_CNT_RAT: 12시-13시 이용 건수 비중, HR_14_17_UE_CNT_RAT: 14시-17시 이용 건수 비중, HR_18_22_UE_CNT_RAT: 18시-22시 이용 건수 비중, HR_23_4_UE_CNT_RAT: 23시-4시 이용 건수 비중, LOCAL_UE_CNT_RAT: 현지인 이용 건수 비중 (고객 자택 주소가 제주도인 경우 현지인으로 정의), RC_M12_MAL_CUS_CNT_RAT: 최근 12개월 남성 회원수 비중 (기준연월 포함 최근 12개월 집계한 값), RC_M12_FME_CUS_CNT_RAT: 최근 12개월 여성 회원수 비중 (기준연월 포함 최근 12개월 집계한 값), RC_M12_AGE_UND_20_CUS_CNT_RAT: 최근 12개월 20대 이하 회원수 비중 (기준연월 포함 최근 12개월 집계한 값), RC_M12_AGE_30_CUS_CNT_RAT: 최근 12개월 30대 회원수 비중 (기준연월 포함 최근 12개월 집계한 값), RC_M12_AGE_40_CUS_CNT_RAT: 최근 12개월 40대 회원수 비중 (기준연월 포함 최근 12개월 집계한 값), RC_M12_AGE_50_CUS_CNT_RAT: 최근 12개월 40대 회원수 비중 (기준연월 포함 최근 12개월 집계한 값), RC_M12_AGE_OVR_60_CUS_CNT_RAT: 최근 12개월 60대 이상 회원수 비중 (기준연월 포함 최근 12개월 집계한 값)
# """

# # 위의 [context] 정보 내에서 [질의]에 대해 답변 [예시]와 같이 술어를 붙여서 답하세요.
# # 사용자가 구체적인 숫자를 제시하지 않았다면, 3-5개의 맛집을 추천해주세요.
# # 'visit_region'은 area 변수를 기준으로 선택되었습니다. 
# # 추천 이유는 구체적일 수록 좋습니다. 왜 사용자에게 이런 맛집을 추천했는지 비중 데이터를 근거로 설명해주세요.

# prompt = ChatPromptTemplate.from_template(template)


# ### 5. Google Gemini 모델 생성 ###
# @st.cache_resource
# def load_model():
#     system_instruction = "당신은 제주도 여행객에게 제주도 맛집을 추천하는 친절한 제주도°C 챗봇입니다. 거짓말을 할 수 없으며, 주어진 데이터를 기반으로 얘기하세요."
#     model = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
#                                    temperature=0.2,
#                                    max_tokens=5000,
#                                    system_instruction=system_instruction)
#     print("model loaded...")
#     return model
# model = load_model()


# ### 6. 검색 결과 병합 함수 ###
# def merge_pages(pages):
#     merged = "\n\n".join([page.page_content for page in pages if page.page_content])

#     # 검색된 문서 리스트 출력 (필터 확인용)
#     for page in pages:
#         print(f"검색된 문서: {page.metadata['area']}")  # 지역 필터 확인을 위해 'area' 필드 출력
    
#     return merged
    


# ## 7. LangChain 체인 구성 ###
# chain = (
#     {"query": RunnablePassthrough(),
#      "context": retriever | merge_pages,    # retriever로 검색된 문서를 merge_pages 함수에 전달
#      "user_name":RunnablePassthrough(),     # RunnablePassThrough: 값을 변경하지 않고 그대로 통과시킴
#      "user_age":RunnablePassthrough(),
#      "visit_times":RunnablePassthrough(),
#      "visit_region": RunnablePassthrough(),
#      "visit_dates": RunnablePassthrough(),
#     }
#     | prompt
#     | load_model()
#     | StrOutputParser()
# )


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
