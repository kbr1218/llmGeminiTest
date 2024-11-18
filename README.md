# RAG-Powered Chatbot using Gemini API and Streamlit🍊
**: LLM 활용 제주도 맛집 추천 대화형 AI 서비스 "친절한 제주°C" (w/ 따뜻한 제주°SEA)**

<br>

## 🙋🏻‍♀️ 프로젝트 소개
**🍊제주°C**: 기상청 제주도 **시간대별 지역별 평균 기온 데이터**와 신한카드 **제주 가맹점 이용 데이터** 기반 제주도 맛집 추천 서비스  
**🌊제주° SEA**: 제주도 바다의 **월별 지역별 평균 수온 데이터**를 활용한 해수욕장 추천과 수온에 따른 **적절한 물놀이 복장 안내** 및 해수욕장 **근처 맛집** 추천 서비스

<br>

## 🛠 시스템 아키텍처
<div style="text-align: left;">
  <div style="margin: ; text-align: left;" "text-align: left;">
    <p><strong>service interface: &nbsp;</strong>
      <img src="https://img.shields.io/badge/streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white">
      <img src="https://img.shields.io/badge/javascript-F7DF1E?style=flat&logo=javascript&logoColor=white">
    </p>
    <p><strong>python server: &nbsp;</strong>
      <img src="https://img.shields.io/badge/python-3776AB?style=flat&logo=python&logoColor=white">
      <img src="https://img.shields.io/badge/langchain-1C3C3C?style=flat&logo=langchain&logoColor=white">
    </p>
    <p><strong>LLM: &nbsp;</strong>
      <img src="https://img.shields.io/badge/googlegemini-8E75B2?style=flat&logo=googlegemini&logoColor=white">
    </p>
    <p><strong>vectorDB: &nbsp;</strong>
      <img src="https://img.shields.io/badge/huggingface-FFD21E?style=flat&logo=huggingface&logoColor=white">
      <img src="https://img.shields.io/badge/ChromaDB-f76144?style=flat&logo=Java&logoColor=white">     </p>
  </div>
</div>

<br>

## 📝 주요 기능
1. **제주도 맛집 추천 챗봇 제주°C**
   * 사용자가 사전에 입력한 정보(연령대, 맛집 방문 일자, 방문 시간, 방문 지역)에 맞춰 맛집 추천
   <br>
2. **제주도 해수욕장 추천 챗봇 제주°SEA**
   * 제주도 월별 지역별 수온에 따른 해수욕장 추천
   * 적절한 물놀이 복장 안내
   * 해수욕장 근처 맛집 추천
   <br>
3. **맛집 검색 및 저장**
   * 챗봇에게 추천받은 맛집을 검색해서 저장
   * 저장한 맛집을 지도에서 확인
   <br>
4. **통계 확인**
   * 월별 업종별 맛집 이용 건수 순위 확인
   * 현지인 방문 비중 80% 이상 맛집 확인
   <br>
4. **지역별 제주 관광 지도**
   * 비짓제주에 등록된 관광지를 지역별로 확인 

<br>

## 🖥 서비스 화면
- 초기 화면
<div align="center">
    <img src="/img/0_로그인.png" width="200">
</div>
<br><br>

- 사전 질문
<div align="center">
</div>
<br><br>

- 제주°C & 제주°SEA 챗봇
<div align="center">
</div>
<br><br>

- 부가 기능 (맛집 검색 및 저장/통계/제주 관광 지도)
<div align="center">
</div>
<br><br>

<br>

## 📂 데이터 출처
- [신한카드 제공 제주 가맹점 데이터](https://github.com/greenjhp/shcard_bigcontest2024_llm)
- [기상청 시간대별 지역별 평균 기온 데이터](https://data.kma.go.kr/cmmn/main.do)
- [제주 바다 월별 지역별 수온 데이터](https://m.badatime.com/)
- [비짓제주 관광지 데이터](https://www.visitjeju.net/kr)

<br>

## ✅ 실행 방법
1. 필요한 패키지 설치  
   `poetry install` | `pip install -r requirements.txt`
2. .env 환경변수 설정  
   : `.env_example` 참고  

4. streamlit 실행  
   `streamlit run app.py`
