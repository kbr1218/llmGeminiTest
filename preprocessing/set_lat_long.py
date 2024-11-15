# set_lat_long.py
import pandas as pd
import requests
import time

# API 키를 여기에 입력하세요
API_KEY = '여기에 API 키'

# CSV 파일 읽기
df = pd.read_csv('여기에 파일 경로')

# 위도와 경도를 저장할 빈 리스트 생성
latitudes = []
longitudes = []

def get_lat_lon(address):
    """주소를 입력받아 위도와 경도를 반환하는 함수"""
    url = f'https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={API_KEY}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['results']:
            lat = data['results'][0]['geometry']['location']['lat']
            lon = data['results'][0]['geometry']['location']['lng']
            return lat, lon
    return None, None

# 주소를 반복하여 위도와 경도를 구함
for address in df['ADDR']:
    lat, lon = get_lat_lon(address)
    latitudes.append(lat)
    longitudes.append(lon)
    time.sleep(1)  # API 호출을 제한하기 위해 1초 대기

# 위도와 경도를 데이터프레임에 추가
df['latitude'] = latitudes
df['longitude'] = longitudes

# 새로운 CSV 파일로 저장
df.to_csv('여기에 파일 경로', index=False, encoding='utf-8-sig')

print("주소 변환 완료! '000000_lat_lon.csv' 파일을 확인하세요.")
