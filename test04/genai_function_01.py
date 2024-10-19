# genai_function_01.py
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GEMINI_MODEL = os.getenv('GEMINI_MODEL')

# Google Gemini API에서 제공하는 Function Calling 자동 호출 파라미터를 활용하여 함수 호출하기
def get_price(product: str) -> int:
  # docstring으로 함수의 기능과 매개변수를 기술
  """제품의 가격을 알려주는 함수
  
  Args:
      theme: 제품명
  """
  return 1000

def get_temperature(city: str) -> float:
  """도시의 온도를 알려주는 함수
  
  Args:
      genre: 도시명
  """
  return 20.5

# tools에 함수를 파라미터로 전달하고 서울의 온도를 물으면
# Gemini는 get_temprature라는 메서드의 결과를 바탕으로 답변을 생성함
model = genai.GenerativeModel(GEMINI_MODEL, tools=[get_price, get_temperature])

# `enable_automatic_funciton_calling=True`로 수행할 때
# >> 간편하지만 멀티턴 대화에서는 잘 작동하지 않는 경우가 있고
# >> 함수 호출 과정을 세밀하게 제어할 수 없는 단점이 있음
chat_session = model.start_chat(enable_automatic_function_calling=True)
response = chat_session.send_message("서울의 온도는?")
print(response.text)

# docstring 외에도 함수 이름, 매개변수, 반환타입까지 고려하므로 함수명/타입 어노테이션을 꼼꼼히 기술해야 함