# app.py
import streamlit as st

# 이미지 변수 선언
titleImgPath = 'https://raw.githubusercontent.com/kbr1218/streamlitTest/main/imgs/banner1.png'
botImgPath = 'https://raw.githubusercontent.com/kbr1218/streamlitTest/main/imgs/dolhareubang3.png'

# 페이지 제목 설정
st.set_page_config(page_title="시작 페이지", page_icon=":🍊:", layout="wide",
                   initial_sidebar_state='collapsed')

# 사이드바 가림
st.markdown(
    """
<style>
    [data-testid="collapsedControl"] {
        display: none
    }
</style>
""",
    unsafe_allow_html=True,
)

# CSS 파일 불러오기
with open('style.css', encoding='utf-8') as css_file:
    st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)

# 타이틀 이미지
titleImg = (f"""
<div class=titleImg>
    <img src="{titleImgPath}" alt="title image" width=100%>
</div>
""")
st.markdown(titleImg, unsafe_allow_html=True)

st.caption("🚀 2024 빅콘테스트 (생성형 AI 분야) 팀: 헬로빅콘")
st.markdown("<hr>", unsafe_allow_html=True)


# 말풍선
st.markdown(f"""
    <div class="chat-container">
        <img src="{botImgPath}" class="chat-icon" alt="chatbot">
        <div class="chat-bubble">
            <div class="chat-text">
                <strong class="color_orange">🏵️친절한 제주도°C:</strong><br>
                혼저옵서예! <br>
                신한카드 <strong>'제주 가맹점 이용 데이터'</strong>와 기상청 <strong>'월별 시간별 기온 데이터'</strong> 기반으로<br>
                삼춘한테 딱 맞는 맛집들 추천해드릴게<span class="color_orange">마씸 🍊</span>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)
st.write("")


# 긴 말풍선 테스트
st.markdown(f"""
    <div class="chat-container">
        <img src={botImgPath} class="chat-icon" alt="chatbot">
        <div class="chat-bubble">
            <div class="chat-text">
                우리 맛집 추천 서비스를 쓰려면 먼저 5가지 질문에 대답해줘야 함서. <br>
                왜냐면, 그 대답에 맞춰서 <strong>맛집</strong>을 추천해줄거라<span class="color_orange">마씸.</span><br><br>
                그럼 묻는 말에 대답 함 해볼까<span class="color_orange">마씸.</span> <br>
                <strong>출발 하쿠다!<strong> 
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)


# 시작하기 버튼 (or 로그인 버튼)
st.write("")
start_button = st.page_link("pages/survey.py",
                            label="[**시작하기✈️**]",
                            use_container_width=True
                            )
