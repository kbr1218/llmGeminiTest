# more.py
import streamlit as st
from pages.subpages.widgets import tab_map, tab_trend, tab_sights

# 좋아요와 지도 탭 모달 화면
@st.dialog("더 알아보기", width='large')
def show_more_modal(fav_restaurants):
  # tabs
  tab1, tab2, tab3 = st.tabs(['저장된 맛집', '트렌드 알아보기', '제주도 관광지 알아보기'])
  with tab1:
    tab_map.show_tab_map(fav_restaurants)
  with tab2:
    tab_trend.show_tab_trend()
  with tab3:
    tab_sights.show_tab_sight()
    