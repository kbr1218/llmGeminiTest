# chat_search.py
import pandas as pd
import streamlit as st
from pages.subpages.modal import more

# 데이터 불러오기
df = pd.read_csv("data/preprocessed/unique_restaurant.csv", encoding='cp949')

# 검색 함수 정의
def search(query):
    filtered_df = df[df['MCT_NM'].str.contains(query, case=False, na=False)]
    return filtered_df[['MCT_NM', 'area', 'ADDR', 'latitude', 'longitude']]

# 메인 함수
def show_search_restaurant():
    with st.container():
      st.caption("제주℃에게 추천받은 맛집을 검색하고 저장하세요.")

      query = st.text_input("검색하기",
                            key='search_query',
                            label_visibility='collapsed')
      
      # favs session_state 초기화
      if 'favs' not in st.session_state:
        st.session_state.favs = []
      
      if query:
          # 검색 결과 반환
          search_results = search(query)

          st.caption(f"검색 결과: {len(search_results)} 개")

          if not search_results.empty:
              # 체크박스 추가
              for idx, row in search_results.iterrows():
                  # 체크박스 초기 상태 설정
                  is_checked = idx in st.session_state.favs
                  checkbox = st.checkbox(f"""**{row['MCT_NM']}** ({row['area']})  
                                         {row['ADDR']}""",
                                         value=is_checked,
                                         key=idx
                                         )

                  # checkbox 상태에 따라 favs 업데이트
                  if checkbox and idx not in st.session_state.favs:
                      st.session_state.favs.append(idx)
                      st.info(f"**{row['MCT_NM']}**이 저장되었습니다.")
                  elif not checkbox and idx in st.session_state.favs:
                      st.session_state.favs.remove(idx)
                      st.warning(f"**{row['MCT_NM']}**이 삭제되었습니다.")
          else:
             st.error("검색 결과가 없습니다.")
      
      # 버튼
      if st.button("지도로 확인하기"):
        fav_restaurants = df.loc[st.session_state.favs, ['MCT_NM', 'area', 'ADDR', 'latitude', 'longitude']]
        more.show_more_modal(fav_restaurants)
    
