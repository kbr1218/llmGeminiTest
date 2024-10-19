# chatbot_02.py
import streamlit as st

# 대화 이력을 저장하고 사용자의 상호작용이 발생할 때마다 대화 이력 전체를 출력하는 코드 추가
st.title("echo-bot")


# session_state 객체를 활용하여 대화 이력을 세션으로 관리하도록 함
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for content in st.session_state.chat_history:
    with st.chat_message(content["role"]):
        st.markdown(content['message'])    

if prompt := st.chat_input("메시지를 입력하세요."):
    with st.chat_message("user"):
        st.markdown(prompt)
        # 딕셔너리를 이용하여 role과 msg 모두 저장
        st.session_state.chat_history.append({"role": "user", "message": prompt})

    with st.chat_message("ai"):                
        response = f'{prompt}... {prompt}... {prompt}...'
        st.markdown(response)
        st.session_state.chat_history.append({"role": "ai", "message": response})
