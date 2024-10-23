# setup_vectorstore.py

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# 0. metadata 설명
metadata_descriptions = {
  'YM': '기준연월(1월~12월)',
  'MCT_NM': '가맹점명',
  'MCT_TYPE': '요식관련 30개 업종',
  'temp_05_11': '5시 11시 평균 기온',
  'temp_12_13': '12시 13시 평균 기온',
  'temp_14_17': '14시 17시 평균 기온',
  'temp_18_22': '18시 22시 평균 기온',
  'temp_23_04': '23시 4시 평균 기온',
  'TEMP_AVG': '월(YM) 평균 기온',
  'area': '제주도를 10개의 지역으로 구분: 동부/서부/남부/북부/산지/가파도/마라도/비양도/우도/추자도',
  'ADDR': '가맹점 주소',
  'RANK_CNT': '월별 업종별 이용건수 분위수 구간을 6개 구간으로 집계 시 해당 가맹점의 이용건수가 포함되는 분위수 구간 * 1:상위10%이하 2:상위10~25% 3:상위25~50% 4:상위50~75% 5:상위75~90% 6:상위90% 초과(하위10%이하) * 상위 30% 매출 가맹점 내 분위수 구간임',
  'RANK_AMT': '월별 업종별 이용금액 분위수 구간을 6개 구간으로 집계 시 해당 가맹점의 이용금액이 포함되는 분위수 구간 * 1:상위10%이하 2:상위10~25% 3:상위25~50% 4:상위50~75% 5:상위75~90% 6:상위90% 초과(하위10%이하) * 상위 30% 매출 가맹점 내 분위수 구간임',
  'RANK_MEAN': '월별 업종별 건당평균이용금액 분위수 구간을 6개 구간으로 집계 시 해당 가맹점의 건당평균이용금액이 포함되는 분위수 구간 * 1:상위10%이하 2:상위10~25% 3:상위25~50% 4:상위50~75% 5:상위75~90% 6:상위90% 초과(하위10%이하) * 상위 30% 매출 가맹점 내 분위수 구간임',
  'MON_UE_CNT_RAT': '월요일 이용 건수 비중',
  'TUE_UE_CNT_RAT': '화요일 이용 건수 비중',
  'WED_UE_CNT_RAT': '수요일 이용 건수 비중',
  'THU_UE_CNT_RAT': '목요일 이용 건수 비중',
  'FRI_UE_CNT_RAT': '금요일 이용 건수 비중',
  'SAT_UE_CNT_RAT': '토요일 이용 건수 비중',
  'SUN_UE_CNT_RAT': '일요일 이용 건수 비중',
  'HR_5_11_UE_CNT_RAT':	'5시-11시 이용 건수 비중',
  'HR_12_13_UE_CNT_RAT': '12시-13시 이용 건수 비중',
  'HR_14_17_UE_CNT_RAT': '14시-17시 이용 건수 비중',
  'HR_18_22_UE_CNT_RAT': '18시-22시 이용 건수 비중',
  'HR_23_4_UE_CNT_RAT': '23시-4시 이용 건수 비중',
  'LOCAL_UE_CNT_RAT': '현지인 이용 건수 비중 (고객 자택 주소가 제주도인 경우 현지인으로 정의)',
  'RC_M12_MAL_CUS_CNT_RAT': '최근 12개월 남성 회원수 비중 (기준연월 포함 최근 12개월 집계한 값)',
  'RC_M12_FME_CUS_CNT_RAT': '최근 12개월 여성 회원수 비중 (기준연월 포함 최근 12개월 집계한 값)',
  'RC_M12_AGE_UND_20_CUS_CNT_RAT': '최근 12개월 20대 이하 회원수 비중 (기준연월 포함 최근 12개월 집계한 값)',
  'RC_M12_AGE_30_CUS_CNT_RAT': '최근 12개월 30대 회원수 비중 (기준연월 포함 최근 12개월 집계한 값)',
  'RC_M12_AGE_40_CUS_CNT_RAT': '최근 12개월 40대 회원수 비중 (기준연월 포함 최근 12개월 집계한 값)',
  'RC_M12_AGE_50_CUS_CNT_RAT': '최근 12개월 40대 회원수 비중 (기준연월 포함 최근 12개월 집계한 값)',
  'RC_M12_AGE_OVR_60_CUS_CNT_RAT': '최근 12개월 60대 이상 회원수 비중 (기준연월 포함 최근 12개월 집계한 값)'
}


# 1. CSV 파일에서 문서 로드
loader = CSVLoader(file_path="data\sample_without_meta.csv",
                   encoding="cp949",
                   csv_args={
                    'fieldnames': list(metadata_descriptions.keys()),
                    'delimiter': ','
                  },
                  metadata_columns=list(metadata_descriptions.keys()))



# 2. 로드한 문서에 메타데이터 설명 추가
pages = loader.load()
for page in pages:
  descriptions = {key: metadata_descriptions[key] for key in page.metadata.keys() if key in metadata_descriptions}
  page.metadata['descriptions'] = ", ".join([f"{k}: {v}" for k, v in descriptions.items()])
  # page.metadata['descriptions'] = {key: metadata_descriptions[key] for key in page.metadata.keys() if key in metadata_descriptions}

print(f"문서 수: {len(pages)}")



# 2. HuggingFace 임베딩 생성
embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask",
                                   encode_kwargs={'normalize_embeddings':True}) # 임베딩을 정규화하여 모든 벡터가 같은 범위의 갖도록 함

# 3. Chroma 벡터스토어에 문서 저장
vectorstore = Chroma.from_documents(pages, embeddings, persist_directory="./database_1000")
vectorstore.persist()
print("벡터스토어 저장 완료!")