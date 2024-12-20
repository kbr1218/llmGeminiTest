# survey.py: 사용자 질문 페이지
import streamlit as st
import datetime, time

# 이미지 변수 선언
botImgPath = 'https://raw.githubusercontent.com/kbr1218/streamlitTest/main/imgs/jejudoC.png'
jejuMapImgPath = 'https://raw.githubusercontent.com/kbr1218/streamlitTest/main/imgs/jejuMap_3.png'

# 페이지 제목 설정
st.set_page_config(page_title="survey", page_icon=":clipboard:", layout="wide",
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

# 페이지 내용
st.title("📋시작 하기 전에")
st.caption("🚀 2024 빅콘테스트 (생성형 AI 분야) 팀: 헬로빅콘")


##### progress bar #####
if 'percent_complete' not in st.session_state:
    st.session_state['percent_complete'] = 0

progressText = f"진행중.. {st.session_state['percent_complete']}%"
progress = st.progress(st.session_state['percent_complete'])
progress.progress(st.session_state['percent_complete'], text=progressText)

st.markdown("<hr>", unsafe_allow_html=True)
st.write(" ")

########################
##### 질문 1) 이름 #####
st.markdown(f"""
    <div class="chat-container">
        <img src="{botImgPath}" class="chat-icon" alt="chatbot">
        <div class="chat-bubble">
            <div class="chat-text">
                Hi there🖐️! 안녕하세요.<br>
                <strong class="title_text">친절한 제주°C</strong> 입니다.<br>
                5가지 질문을 하겠습니다.<br>
                먼저, 사용자의 <strong>이름</strong>을 알려주세요.   
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

if 'user_name' not in st.session_state:
    # 이름 입력 필드
    user_name = st.text_input("이름을 입력해주세요",
                              placeholder="이름을 입력해주세요", 
                              key='user_name_input',
                              label_visibility='hidden')
    if user_name:
        st.session_state['user_name'] = user_name
        st.session_state['percent_complete'] += 20         # progres bar +20
        st.rerun()

else:
    # 사용자의 대답
    st.markdown(f"""
        <div class="user-chat-container">
            <div class="chat-bubble">
                <div class="user-chat-text">
                    <span class='makebold'>{st.session_state['user_name']}</span> 입니다.
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    progress.progress(st.session_state['percent_complete'], text=progressText)



    ##########################
    ##### 질문 2) 연령대 #####
    time.sleep(0.5)  # 성별 대답 후 잠시 대기
    st.markdown(f"""
        <br>
        <div class="chat-container">
            <img src="{botImgPath}" class="chat-icon" alt="chatbot">
            <div class="chat-bubble">
                <div class="chat-text">
                    안녕하세요. <strong>{st.session_state['user_name']}</strong>님! <br>
                    다음 질문입니다. <br>
                    사용자의 <strong>연령대</strong>를 알려주세요.
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 연령대 selectbox
    if 'age' not in st.session_state:
        st.write("")
        age = st.selectbox("연령대를 선택해주세요", 
                           ("연령대를 선택해주세요", "20대 이하", "30대", "40대", "50대", "60대 이상"),
                           key="age_select", label_visibility="collapsed")
            
        if age != "연령대를 선택해주세요":              # 사용자가 나이를 선택하면
            st.session_state['age'] = age               # 선택된 값을 session_state에 저장
            st.session_state['percent_complete'] += 20  # progress bar + 20

            # 선택 후 selectbox를 숨기기 위해 'age'를 세션에 저장한 후 refresh
            st.rerun()

    else:
        # progress bar 상태 갱신
        progress.progress(st.session_state['percent_complete'], text=progressText)

        # 선택된 연령대 출력
        st.markdown(f"""
            <div class="user-chat-container">
                <div class="chat-bubble">
                    <div class="user-chat-text">
                        <span class='makebold'>{st.session_state['age']}</span> 입니다.
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        ############################
        ##### 질문 3) 방문날짜 #####
        time.sleep(0.5)                   # 연령대 대답 후 잠시 대기
        st.markdown(f"""
            <br>
            <div class="chat-container">
                <img src="{botImgPath}" class="chat-icon" alt="chatbot">
                <div class="chat-bubble">
                    <div class="chat-text">
                        <strong>{st.session_state['age']}</strong>을(를) 선택하셨습니다. <br>
                        세 번째 질문입니다. <br>
                        맛집을 언제 방문하실 계획인가요? <br>
                        <strong>맛집 방문 일자</strong>을 알려주세요.
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # 날짜 변수
        today = datetime.datetime.now()
        one_year_later = today.replace(year=today.year + 1)

        if 'visit_dates' not in st.session_state:
            # 날짜 선택 (기본값 오늘)
            visit_dates = st.date_input(
                "제주도 방문일 선택",
                value=today,                     # 날짜 기본값
                min_value=today,                 # 선택 가능 최소 날짜: 오늘
                max_value=one_year_later,        # 선택 가능 최대 날짜: 일년 후
                format="YYYY-MM-DD",
                label_visibility='collapsed'
            )
            confirm_button1 = st.button("확인")

            # 확인 버튼을 눌렀을 때만 세션에 저장하고, progress bar 갱신
            if confirm_button1:
                st.session_state['visit_dates'] = visit_dates
                st.session_state['percent_complete'] += 20
                st.rerun()

        # 사용자가 날짜 선택 후 확인 버튼을 누른 경우에만 선택한 날짜 출력
        else:
            # progress bar 상태 갱신
            progress.progress(st.session_state['percent_complete'], text=progressText)

            # 선택된 방문 날짜 출력
            st.write("")
            st.markdown(f"""
                <div class="user-chat-container">
                    <div class="chat-bubble">
                        <div class="user-chat-text">
                            <span class='makebold'>{st.session_state['visit_dates']}</span>에 방문 예정입니다.
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)


            ###############################
            ##### 질문 4) 방문 시간대 #####
            time.sleep(0.5)
            st.markdown(f"""
                <br>
                <div class="chat-container">
                    <img src="{botImgPath}" class="chat-icon" alt="chatbot">
                    <div class="chat-bubble">
                        <div class="chat-text">
                            좋습니다! <br>
                            다음 질문입니다. <br>
                            맛집을 언제 방문할 계획이신가요? <br>
                            제주도 맛집 <strong>방문 시간대</strong>를 알려주세요. <br>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # visit_times 값이 세션에 없을 때만 radio buttons 렌더링
            if 'visit_times' not in st.session_state:
                # 방문 시간대 선택 (single select)
                st.write("")

                visit_times = st.radio(
                    "방문 시간대 선택",
                    options=["아침 (05-11시)", "점심 (12-13시)", "오후 (14-17시)", "저녁 (18-22시)", "심야 (23-04시)"],
                    index=0,                        # 기본값은 첫 번째 항목으로 (아침)
                    label_visibility='collapsed'
                )
                confirm_button2 = st.button('확인')
                
                if confirm_button2:
                    st.session_state['visit_times'] = visit_times
                    st.session_state['percent_complete'] += 20
                    st.rerun()
                
            # 사용자가 방문 시간대를 선택하고 확인 버튼을 누른 경우에만 선택한 시간대 출력
            else:
                # progress bar 상태 갱신
                progress.progress(st.session_state['percent_complete'], text=progressText)
                # 선택된 방문 시간대 출력
                st.markdown(f"""
                    <div class="user-chat-container">
                            <div class="chat-bubble">
                                <div class="user-chat-text">
                                    <span class='makebold'>{st.session_state['visit_times']}</span>에 방문할 계획입니다.
                                </div>
                            </div>
                    </div>
                """, unsafe_allow_html=True)


                #############################
                ##### 질문 5) 방문 지역 #####
                time.sleep(0.5)
                st.markdown(f"""
                    <br>
                    <div class="chat-container">
                        <img src="{botImgPath}" class="chat-icon" alt="chatbot">
                        <div class="chat-bubble">
                            <div class="chat-text">
                                <strong>{st.session_state['visit_times']}</strong>을(를) 선택하셨습니다. <br>
                                마지막 질문입니다. <br>
                                제주도 <strong>어느 지역의 맛집</strong>을 찾으시나요? <br>
                                <strong>하나 이상</strong> 선택할 수 있습니다. <br>
                                <img src="{jejuMapImgPath}" class="jejuMap-icon" alt="제주도 지역 구분 지도">
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                # 'region' 값이 세션에 없을 때만 multiselect 렌더링
                if 'region' not in st.session_state:
                    # 지역 선택 (multiselect)
                    st.write("")
                    visit_region = st.multiselect(
                        "제주도 방문 지역 선택",
                        options=["동부", "서부", "남부", "북부", "우도", "비양도", "추자도", "가파도", "마라도"],
                        default=None,
                        label_visibility='collapsed'
                    )
                    confirm_button3 = st.button('확인')

                    # 사용자가 확인 버튼을 눌렀을 때만 세션에 저장하고, progress bar 갱신 및 새로고침
                    if confirm_button3 and visit_region:
                        st.session_state['region'] = visit_region
                        st.session_state['percent_complete'] += 20
                        st.rerun()
                # 사용자가 방문 지역을 선택하고 확인 버튼을 누른 경우에만 선택한 지역 출력
                else:
                    # progress bar 상태 갱신
                    progressText = f"{st.session_state['percent_complete']}% ! 모든 질문이 끝났습니다"
                    progress.progress(st.session_state['percent_complete'], text=progressText)

                    # 선택된 방문 지역 출력
                    st.markdown(f"""
                        <div class="user-chat-container">
                            <div class="chat-bubble">
                                <div class="user-chat-text">
                                    <span class='makebold'>{', '.join(st.session_state['region'])}</span>를 선택했습니다.
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                    time.sleep(0.5)
                    st.markdown(f"""
                        <br>
                        <div class="chat-container">
                            <img src="{botImgPath}" class="chat-icon" alt="chatbot">
                            <div class="chat-bubble">
                                <div class="chat-text">
                                    감사합니다.🙇‍♂️ <br>
                                    제주도 맛집을 찾기 위한 모든 질문이 끝났습니다. <br>
                                    <strong>지역별 시간대별 기온</strong>과 함께 사용자 맞춤형 맛집을 <strong>추천받고 싶다면</strong>,<br>
                                    아래 버튼을 클릭해 <strong>다음 페이지로 넘어가주세요</strong>.
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                    # 시작하기 버튼 (or 로그인 버튼)
                    st.write("")
                    start_button = st.button("**다음으로**👉",
                         type='primary',
                         use_container_width=True)
                    if start_button:
                        st.switch_page("./pages/chat.py")