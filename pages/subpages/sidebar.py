# sidebar.py
import streamlit as st

from .widgets import weather
from .modal import edit_modal

def show_sidebar():
  ### 1. 사용자 정보 ###
  if 'user_name' in st.session_state:
    st.subheader(f":rainbow[{st.session_state['user_name']}]님의 제주 맛집 탐방🏝️")
  else:
    st.subheader(":rainbow[신나는] 제주 맛집 탐방🏝️")

  # 1-2. 연령대
  if 'age' in st.session_state:
      st.sidebar.markdown(f"**연령대**: {st.session_state['age']}")
  else:
      st.sidebar.warning("연령대 정보가 입력되지 않았습니다.")

  # 1-3. 방문 날짜
  if 'visit_dates' in st.session_state:
    visit_dates_str = f"{st.session_state['visit_dates']}"
    st.sidebar.markdown(f"**방문 날짜**: {visit_dates_str}")
  else:
    st.sidebar.warning("날짜 정보가 입력되지 않았습니다.")

  # 1-4. 방문 시간대
  if 'visit_times' in st.session_state:
      st.sidebar.markdown(f"**방문 시간대**: {st.session_state['visit_times']}")
  else:
      st.sidebar.warning("시간대 정보가 입력되지 않았습니다.")

  # 1-5. 방문 지역
  if 'region' in st.session_state:
      st.sidebar.markdown(f"**방문 지역**: {', '.join(st.session_state['region'])}")
  else:
      st.sidebar.warning("지역 정보가 입력되지 않았습니다.")
  
  # 수정하기 버튼
  if st.button("수정하기🖋️",
               type="secondary",
               use_container_width=True):
     edit_modal.show_edit_modal()
  
  st.markdown("<hr>", unsafe_allow_html=True)


  ### 3. 날씨 위젯 ###
  weather.show_weather()