# chat_sea.py
import streamlit as st

# 페이지 제목 설정
st.set_page_config(page_title="제주도SEA", page_icon="🐬", layout="wide",
                   initial_sidebar_state='expanded')

from pages.subpages import sidebar
from pages.subpages import chat_search

# 사이드바
with st.sidebar:
    sidebar.show_sidebar()






### 10. Streamlit UI ###
st.subheader("🐬:blue[제주도 SEA]에게 질문하기")
st.divider()

say_hi_to_user = f"""
"""

chat_col1, search_col2 = st.columns([2, 1])
with search_col2:
    chat_search.show_search_restaurant()

    # 채팅 기록 초기화
    if st.button("채팅 기록 초기화", type='primary'):
        # st.session_state.messages_sea = [
        #     {"role": "assistant", "content": say_hi_to_user}
        # ]
        st.rerun()