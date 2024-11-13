# tab_sights.py
import streamlit as st
import folium
import pandas as pd
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster

### to-do: 선택한 지역에 따라 지도 중심 위경도 변경

# 제주도 중심 위도경도 변수 선언
LAT = 33.55
LONG = 126.55

df_sights = pd.read_csv('data\preprocessed\jeju_sights.csv', encoding='cp949')

def show_tab_sight():
  st.subheader('제주 관광 지도🏖️')

  # 지역 선택
  region_list = ['지역을 선택해주세요'] + sorted(df_sights['지역'].unique().tolist())
  region_selected = st.selectbox("지역을 선택하세요:", region_list, index=0)

  if region_selected == '지역을 선택해주세요':
    # 기본 지도 출력
    m = folium.Map(location=[33.38032, LONG], zoom_start=9)

    popup = folium.Popup('제주도', min_width=10, max_width=50)
    folium.Marker(
      location=[33.38032, LONG],
      popup=popup,
      icon=folium.Icon(color="red", icon='heart', prefix='fa')
    ).add_to(m)
    st_folium(m, height=400, use_container_width=True)

  else:
    # 선택한 지역의 관광지 필터링
    filtered_df = df_sights[df_sights['지역'] == region_selected]

    # 지도 중심 좌표 계산 (필터링된 데이터의 평균 위도/경도)
    if not filtered_df.empty:
      center_lat = filtered_df['latitude'].mean()
      center_long = filtered_df['longitude'].mean()
    else:
      center_lat, center_long = LAT, LONG  # 관광지가 없을 경우 기본 중심 사용

    # 제주도 중심 지도
    m = folium.Map(location=[center_lat, center_long], zoom_start=11)

    # MarkerCluster 객체 생성
    marker_cluster = MarkerCluster().add_to(m)

    # 필터링된 관광지 데이터 지도에 표시
    for _, row in filtered_df.iterrows():
      # 관광지 이름과 주소를 팝업으로 설정
      popup = folium.Popup(f"{row['관광지명']}", max_width=250)

      # 마커 추가
      folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=popup,
        icon=folium.Icon(color="red", icon='map-marked-alt', prefix='fa')
      ).add_to(marker_cluster)

    # Folium 지도를 Streamlit에 표시
    st_folium(m, height=400, use_container_width=True)

    # 관광지 목록 표로 출력
    st.write(f"**📍{region_selected}의 관광지**")
    st.dataframe(filtered_df[['관광지명', '주소']], hide_index=True, use_container_width=True)