# chat.py
import streamlit as st

# 페이지 제목 설정
st.set_page_config(page_title="main", page_icon="💬", layout="wide",
                   initial_sidebar_state='expanded')

from pages.subpages import sidebar, tab_chat
from pages.subpages.modal import more

# CSS 파일 불러오기
with open('style/chat_page.css', encoding='utf-8') as css_file:
    st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)

# 사이드바
with st.sidebar:
    sidebar.show_sidebar()

# title
st.title("chat page")
# st.divider()

if st.button("더 알아보기"):
    more.show_more_modal()

# 채팅화면 출력
tab_chat.show_tab_chat()

