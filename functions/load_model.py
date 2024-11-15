# load_model.py
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

### 환경변수 로드 ###
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

### Gemini model ###
def load_gemini(system_instruction):
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        max_tokens=5000,
        system_instruction=system_instruction
    )
    print(">>>>>>> model loaded...")
    return model
