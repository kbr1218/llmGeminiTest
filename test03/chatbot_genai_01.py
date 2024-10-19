# chatbot_genai_01.py
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GEMINI_MODEL = os.getenv('GEMINI_MODEL')

# Gemini API를 사용하는 경우, 데이터 캐싱으로 모델을 가져와도 직렬화 오류가 나지 않음 (test01.ipynb 참고)
# >> 바이너리 상태의 실제 모델을 로드하는 것이 아니라 API 연동을 위한 클래스의 인스턴스를 생성하는 것이기 때문

st.title("Gemini chatbot 01")

# 사용자의 인터랙션이 있을 때마다 인스턴스를 매번 생성하는 것보다는 참조를 전달받아 접근하는 것이 효율적 >> 리소스 캐싱 사용
@st.cache_resource
def load_model():
  genai.configure(api_key=GOOGLE_API_KEY)
  model = genai.GenerativeModel('gemini-1.5-flash')
  print("model loaded...")
  return model

model = load_model()

# 대화 이력은 google gemini API의 ChatSession 사용 (별도 관리X)
if "chat_session" not in st.session_state:    
    st.session_state["chat_session"] = model.start_chat(history=[]) 

# 대화 이력 출력
for content in st.session_state.chat_session.history:
    with st.chat_message("ai" if content.role == "model" else "user"):
        st.markdown(content.parts[0].text)

# 채팅
if prompt := st.chat_input("메시지를 입력하세요."):    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("ai"):
        # DeltaGenerator 반환
        # >> st.chat_message에 의해 생성된 출력 메시지 컨텍스트에 동적으로 데이터를 업데이트하는 place holder 역할 수행
        # >> 모델로부터 스트리밍 방식으로 반환받은 문자열을 이전에 반환받은 문자열과 결합하여 place holder로 전달하면
        # >> 출력 메시지 컨테이너에 문자열이 늘어나는 방식으로 모델의 응답 결과가 화면에 표현됨
        message_placeholder = st.empty()
        full_response = ""

        # spinner를 추가하여 작업이 진행중임을 사용자에게 알려줌
        with st.spinner("메시지 처리중입니다."):
          # ChatSession 객체의 send_message 호출 결과를 출력
          response = st.session_state.chat_session.send_message(prompt, stream=True)
          # `stream = Ture` >> 메시지 생성 요청에 대한 응답이 스트리밍 방식으로 제공되어 
          # 모델이 메시지 생성을 마치기 전에도 중간 결과를 실시간으로 수신받을 수 있음
          for chunk in response:
              full_response += chunk.text
              message_placeholder.markdown(full_response)