# tab_trend.py
import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import time
import altair as alt

# 제주도 중심 위도경도 변수 선언
LAT = 33.55
LONG = 126.7

# 데이터프레임 가져오기
df_month = pd.read_csv("data/preprocessed/rank_by_month_type.csv", encoding='cp949')
df_local = pd.read_csv("data/preprocessed/local_over_80.csv", encoding='cp949')


def show_tab_trend():
  ###################################
  #### 01. 월별 타입별 방문객 순위 ####
  with st.container(border=True):
    # 컬럼 레이아웃 설정
    trend_col1, trend_col2 = st.columns([3, 2])  # 화면을 3:1로 나눔
    # type 종류
    graph_type_options = sorted(df_month['MCT_TYPE'].unique())

    with trend_col2:
      # multiselect
      graph_selected_type = st.multiselect('Select Type (최대 5개까지 선택 가능합니다)',
                                            graph_type_options,
                                            default=[graph_type_options[0]], # 기본값으로 하나 선택
                                            key='graph1'
                                            )
                
      # 선택한 MCT_TYPE이 5개를 넘으면 경고 메시지 표시
      if len(graph_selected_type) > 5:
        st.warning("⚠️ 카테고리는 최대 5개까지 적용됩니다.", icon=":material/warning:")
        graph_selected_type = graph_selected_type[:5]  # max 5
                  
      st.caption("""1: 상위 10% 이하<br>2: 상위 10\~25%<br>3: 상위 25\~50%<br>4: 상위 50\~75%<br> 
                  5: 상위 75\~90%<br>6: 상위 90% 초과 (하위 10% 이하)""", unsafe_allow_html=True)

    with trend_col1:
      # title
      st.subheader("📈제주도 맛집 월별 이용 건수 순위 알아보기")
      st.caption("월별/메뉴 TYPE별 이용 건수의 평균값입니다.")
      if not graph_selected_type:
        graph_selected_type = [graph_type_options[0]]  # 선택된 type이 없으면 기본값으로 첫 번째 타입 그래프 출력

      graph_choosen_types = df_month[df_month['MCT_TYPE'].isin(graph_selected_type)]

      # 그래프 출력
      chart = alt.Chart(graph_choosen_types).mark_line(point=True).encode(
        x=alt.X("MONTH:O", title="월"),
        y=alt.Y('RANK_CNT:Q', title='월 평균 이용 건수 순위 평균'),
                color='MCT_TYPE:N',                                    # 각 타입별 색상 구분
                tooltip=['MONTH', 'MCT_TYPE', 'RANK_CNT']
      )
      st.altair_chart(chart, use_container_width=True)      

  # 약간의 마진 남기기
  st.markdown("<br>", unsafe_allow_html=True)

  #######################################
  #### 02. 현지인 비중 상위 80% 식당 ####
  # 변수 초기화
  if 'map_selected_ym' not in st.session_state:
    st.session_state.map_selected_ym = None
  if 'map_selected_type' not in st.session_state:
    st.session_state.map_selected_type = None
  if 'random_seed' not in st.session_state:
    st.session_state.random_seed = int(time.time())

  with st.container(border=True):
    # title
    st.subheader("🗺️현지인이 선택한 맛집 알아보기")
    st.caption("현지인 이용 비중이 80% 이상인 매장 기준이며, 랜덤으로 5개씩 선정됩니다.")

    # selectbox
    map_ym_options = sorted(df_local['MONTH'].unique())
    map_type_options = sorted(df_local['MCT_TYPE'].unique())

    # 첫 번째 값을 기본값으로 설정
    select_col1, select_col2 = st.columns(2)
    with select_col1:
      map_selected_ym = st.selectbox('Select Month', map_ym_options, key='map1', index=None, placeholder="Select Month")
    with select_col2:
      map_selected_type = st.selectbox('Select Type', map_type_options, key='map2', index=None, placeholder="Select Type")
          
    # MONTH/TYPE이 변경된 경우에만 random_seed 갱신
    if (map_selected_ym != st.session_state.map_selected_ym) or (map_selected_type != st.session_state.map_selected_type):
      st.session_state.random_seed = int(time.time())
      st.session_state.map_selected_ym = map_selected_ym
      st.session_state.map_selected_type = map_selected_type
          
      # 아이콘 설정
    icon_mapping = {
            '커피': 'coffee',
            '디저트/간식': 'ice-cream',
            '베이커리': 'bread-slice',
            '세계 요리': 'globe',
            '맥주/요리주점': 'beer',
            '치킨': 'drumstick-bite',
            '패스트푸드/간단한 식사': 'hamburger',
            '피자': 'pizza-slice'
    }
    icon_name = icon_mapping.get(st.session_state.map_selected_type, 'utensils')


    # 선택한 MONTH과 TYPE_NAME에 따라 데이터 필터링
    filtered_data = df_local[(df_local['MONTH'] == map_selected_ym) & (df_local['MCT_TYPE'] == map_selected_type)]

    # 필터링된 데이터가 없을 경우
    if map_selected_ym is None and map_selected_type is None:
      st.info("월과 Type을 선택해주세요.", icon=":material/info:")
    elif map_selected_ym is None:
      st.info("월을 선택해주세요.", icon=":material/info:")
    elif map_selected_type is None:
      st.info("Type을 선택해주세요.", icon=":material/info:")
    elif filtered_data.empty:
      st.warning("해당하는 매장이 없습니다.", icon=":material/warning:")
    else:
      # 현지인 비중 백분율로 변환
      filtered_data.loc[:, 'LOCAL_UE_CNT_RAT'] = filtered_data['LOCAL_UE_CNT_RAT'].astype(str) + '%'

      # 데이터는 최대 5개로 제한 & 랜덤으로 추출
      filtered_data = filtered_data.sample(n=min(5, len(filtered_data)), random_state=st.session_state.random_seed)

      # 지도 초기화
      m = folium.Map(location=[LAT, LONG], zoom_start=9)

      # 필터링된 위치에 마커 추가
      for _, row in filtered_data.iterrows():
        # popup에 매장명+주소+이용비중 출력
        iframe = row['MCT_NM']+": <strong>"+row['LOCAL_UE_CNT_RAT']+"</strong>"
        popup = folium.Popup(iframe, min_width=50, max_width=300)
        folium.Marker(
              location=[row['latitude'], row['longitude']],
              popup=popup,
              icon=folium.Icon(color="orange", icon=icon_name, prefix='fa')
        ).add_to(m)
          
      # folium 지도를 streamlit에 표시
      st_folium(m, width="100%", height=400)

      # 선택한 식당 정보를 테이블로 출력
      st.write("**📍매장 정보**")
      filtered_data_display = filtered_data[['MCT_NM', 'ADDR', 'LOCAL_UE_CNT_RAT']].rename(
        columns={
              'MCT_NM': '매장명',
              'ADDR': '주소',
              'LOCAL_UE_CNT_RAT': '현지인 이용 비중'
            }
      ).reset_index(drop=True)
      st.table(filtered_data_display)