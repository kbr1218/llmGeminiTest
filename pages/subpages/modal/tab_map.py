# tab_map.py
import streamlit as st
import folium
import pandas as pd
from streamlit_folium import st_folium

# 제주도 중심 위도경도 변수 선언
LAT = 33.55
LONG = 126.55

# 데이터 불러오기
df = pd.read_csv("data/unique_restaurant.csv", encoding='cp949')

def show_tab_map(fav_restaurants):
  # 제주도 중심 지도
  m = folium.Map(location=[LAT, LONG], zoom_start=9)

  # fav_restaurants에 데이터가 있는지 확인
  if not fav_restaurants.empty:
    map_col1, map_col2 = st.columns([3, 2])

    with map_col1:
        # 선택된 맛집을 지도에 마커 추가
        for _, row in fav_restaurants.iterrows():
          # 팝업 출력창 설정
          popup = folium.Popup(row['MCT_NM'], min_width=10, max_width=100)

          folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=popup,
            icon=folium.Icon(color="red", icon='heart', prefix='fa')
          ).add_to(m)

        # folium 지도를 streamlit에 표시
        st_folium(m, width=500, height=450)
    
    with map_col2:
      st.write('저장한 맛집들은 여기:')
      st.dataframe(fav_restaurants[['MCT_NM', 'area', 'ADDR']],
                   hide_index=True)
  
  else:
    map_col1, map_col2 = st.columns([3, 1])

    with map_col1:
      popup = folium.Popup('제주도', min_width=10, max_width=100)

      folium.Marker(
        location=[33.38032, LONG],
        popup=popup,
        icon=folium.Icon(color="red", icon='heart', prefix='fa')
      ).add_to(m)
      st_folium(m, width=500, height=450)

    with map_col2:
      st.warning("저장된 맛집이 없습니다.")

