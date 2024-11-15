# sidebar.py
import streamlit as st

from .widgets import weather
from .modal import edit_modal

def show_sidebar():
  # jeju_sea session_state 초기화
  if 'jeju_sea' not in st.session_state:
      st.session_state['jeju_sea'] = False

  ### 1. 사용자 정보 ###
  if 'user_name' in st.session_state:
    st.subheader(f":rainbow[{st.session_state['user_name']}]님의 제주 맛집 탐방🏝️")
  else:
    st.subheader(":rainbow[신나는] 제주 맛집 탐방🏝️")

  # 1-2. 연령대
  if 'age' in st.session_state:
      st.sidebar.markdown(f"**연령대**: {st.session_state['age']}")
  else:
      st.sidebar.warning("연령대 정보가 입력되지 않았습니다.", icon=":material/priority_high:")

  # 1-3. 방문 날짜
  if 'visit_dates' in st.session_state:
    visit_dates_str = f"{st.session_state['visit_dates']}"
    st.sidebar.markdown(f"**방문 날짜**: {visit_dates_str}")
  else:
    st.sidebar.warning("날짜 정보가 입력되지 않았습니다.", icon=":material/priority_high:")

  # 1-4. 방문 시간대
  if 'visit_times' in st.session_state:
      st.sidebar.markdown(f"**방문 시간대**: {st.session_state['visit_times']}")
  else:
      st.sidebar.warning("시간대 정보가 입력되지 않았습니다.", icon=":material/priority_high:")

  # 1-5. 방문 지역
  if 'region' in st.session_state:
      st.sidebar.markdown(f"**방문 지역**: {', '.join(st.session_state['region'])}")
  else:
      st.sidebar.warning("지역 정보가 입력되지 않았습니다.", icon=":material/priority_high:")
  
  # 수정하기 버튼
  if st.button("수정하기🖋️",
               type="secondary",
               use_container_width=True):
     edit_modal.show_edit_modal()

  # 페이지 전환 버튼
  if st.session_state['jeju_sea']:
     if st.button("제주°C로 돌아가기🍊",
                  type='primary', 
                  use_container_width=True):
        st.session_state['jeju_sea'] = False
        st.switch_page("./pages/chat.py")
        st.rerun()
  else:
    if st.button("제주도SEA 탐험하기🐬",
                type='primary',
                use_container_width=True):
      st.session_state['jeju_sea'] = True
      st.switch_page("./pages/chat_sea.py")  
      st.rerun()
      
  st.markdown("<hr>", unsafe_allow_html=True)


  ### 3. 날씨 위젯 ###
  weather.show_weather()