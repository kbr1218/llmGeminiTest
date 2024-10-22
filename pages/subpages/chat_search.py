# chat_search.py
import pandas as pd
import streamlit as st

# 데이터 불러오기
df = pd.read_csv("data/unique_restaurant.csv", encoding='cp949')

# favs session_state 초기화
if 'favs' not in st.session_state:
    st.session_state.favs = []

# 검색 함수 정의
def search():
    search_query = st.session_state.search_query

    if search_query:
        # 검색어를 포함하는 데이터 필터링
        search_result = df[df['MCT_NM'].str.contains(search_query, case=False, na=False)]
        
        # 검색 결과가 있을 때만 출력
        if not search_result.empty:
            st.write(f"검색 결과: **{len(search_result)}**개")

            # 'area'를 기준으로 그룹화
            grouped_result = search_result.groupby('area')

            # 검색 결과 출력
            for area, group in grouped_result:
                # 지역 출력 (동부, 서부, 남부 등)
                st.write(f"**{area}**")

                for idx, row in group.iterrows():
                    col1, col2 = st.columns([8, 1])

                    with col1:
                        st.write(f"""**{row['MCT_NM']}**  
                                 : {row['ADDR']}""")  # 식당 이름과 주소 출력

                    # 좋아요 버튼
                    with col2:
                        # 고유한 키 생성
                        unique_key = f"like_{idx}_{row['MCT_NM']}_{row['latitude']}_{row['longitude']}"
                        if st.button("❤️", key=unique_key):
                            if {'MCT_NM': row['MCT_NM'], 'ADDR': row['ADDR']} not in st.session_state.favs:
                                st.session_state.favs.append({
                                    'MCT_NM': row['MCT_NM'],
                                    'ADDR': row['ADDR'],
                                    'lat': row['latitude'],
                                    'long': row['longitude']
                                })
                                st.success(f"{row['MCT_NM']}을(를) 좋아요 목록에 추가했습니다.")
                # 그룹간 구분선 추가
                st.divider()
        else:
            st.write("검색 결과가 없습니다.")
    else:
        st.write("검색어를 입력해주세요.")

# 메인 함수
def show_search_restaurant():
    with st.container():  # 검색과 검색 결과를 같은 컨테이너에 포함
      st.caption("제주도℃에게 추천받은 맛집을 검색하고 저장하세요.")

      if st.text_input("검색하기",
                      key='search_query',
                      label_visibility='collapsed'):
        search()
