# tab_sights.py
import streamlit as st
import folium
import pandas as pd
from streamlit_folium import st_folium

### to-do: 선택한 지역에 따라 지도 중심 위경도 변경

# 제주도 중심 위도경도 변수 선언
LAT = 33.55
LONG = 126.55

df_sights = pd.read_csv('data\preprocessed\jeju_sights.csv', encoding='cp949')

def show_tab_sight():
  st.subheader('제주 관광 지도🏖️')

  # 지역 선택
  region_list = df_sights['지역'].unique().tolist()
  region_selected = st.selectbox("지역을 선택하세요:", region_list)

  # 선택한 지역의 관광지 필터링
  filtered_df = df_sights[df_sights['지역'] == region_selected]

  # 제주도 중심 지도
  m = folium.Map(location=[LAT, LONG], zoom_start=9)

  # 필터링된 관광지 데이터 지도에 표시
  if not filtered_df.empty:
    for _, row in filtered_df.iterrows():
      # 관광지 이름과 주소를 팝업으로 설정
      popup = folium.Popup(f"{row['AREA_NM']}", max_width=250)

      # 마커 추가
      folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=popup,
        icon=folium.Icon(color="blue", icon='info-sign')
      ).add_to(m)

    # Folium 지도를 Streamlit에 표시
    st_folium(m, height=400, use_container_width=True)

    # 관광지 목록 표로 출력
    st.write("**📍 선택된 지역의 관광지 목록**")
    st.dataframe(filtered_df[['AREA_NM', 'ADDR']], hide_index=True)
    
  else:
    st.warning("선택된 지역에 관광지가 없습니다.")