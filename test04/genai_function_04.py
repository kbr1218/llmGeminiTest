# genai_function_04.py
# 불필요한 모델 호출을 줄이기 위해 2단계로 함수 호출을 줄이기
# 2단계 함수 호출을 사용한다고 해서 text(user) >> function_call >> function_response >> text(model)
# 데이터 구조를 벗어날 순 없음. 따라서 2단계 수행 후 3, 4단계의 데이터 구조를 임의로 생성해야 함
import google.generativeai as genai
import google.ai.generativelanguage as glm

# 1. 특정 제품에 대한 재고 확인
# 2. 특정 제품에 대한 가격 확인
# 3. 특정 제품에 대한 주문 신청

# 데이터 저장
prod_database = { 
    "갤럭시 S24": {"재고": 10, "가격": 1_700_000}, 
    "갤럭시 S23": {"재고": 5, "가격": 1_300_000}, 
    "갤럭시 S22": {"재고": 3, "가격": 1_100_000}, 
} 

# 여기부터 함수 정의
def is_product_available(product_name: str)-> bool: 
    """특정 제품의 재고가 있는지 확인한다.

    Args:
        product_name: 제품명
    """

    if product_name in prod_database: 
        if prod_database[product_name]["재고"] > 0: 
            # False로 변경
            return False,f"현재 {product_name} 재고가 있습니다."
    return False, f"현재 {product_name} 재고가 없습니다."

def get_product_price(product_name: str)-> int: 
    """제품의 가격을 가져온다.

    Args:
        product_name: 제품명
    """
    if product_name in prod_database: 
        return True, prod_database[product_name]["가격"] 
    return True, None

def place_order(product_name: str, address: str)-> str: 
    """제품 주문결과를 반환한다.
    Args:
        product_name: 제품명
        address: 배송지
    """
    if is_product_available(product_name): 
        prod_database[product_name]["재고"] -= 1 
        return True, "주문 완료" 
    else: 
        return True, "재고 부족으로 주문 불가" 

# 함수를 딕셔너리에 저장
function_repoistory = {     
    "is_product_available": is_product_available, 
    "get_product_price": get_product_price, 
    "place_order": place_order 
} 

# 유니코드 문자열로 나오는 결과를 한글로 다시 바꾸는 함수
def correct_response(response): 
    part = response.candidates[0].content.parts[0] 
    if part.function_call: 
        for k, v in part.function_call.args.items(): 
            byte_v = bytes(v, "utf-8").decode("unicode_escape") 
            corrected_v = bytes(byte_v, "latin1").decode("utf-8") 
            part.function_call.args.update({k:  corrected_v}) 

# 3, 4 단계를 실행시키는 함수
def make_fc_history(chat_session, part, answer):
    # history에 넣어야 하는 Content 객체 정보 생성
    content = glm.Content(parts=[part], role="user")        
    chat_session.history.append(content)

    response = glm.GenerateContentResponse({
        "candidates": [{"content": {"role": "model", "parts": [{"text": answer}]}}]
    })
    chat_session.history.append(response.candidates[0].content)
    return response

# 모델 설정
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash", 
    tools=function_repoistory.values()  # 함수가 저장된 딕셔너리
)

chat_session = model.start_chat(history=[])
queries = ["갤럭시 S24 판매 중인가요?", "가격은 어떻게 되나요?", "서울시 종로구 종로1가 1번지로 배송부탁드립니다"]             

for query in queries: 
    # 1) 사용자가 text 타입으로 질문
    print(f"\n사용자: {query}")     
    response = chat_session.send_message(query) 
    correct_response(response) 
    part = response.candidates[0].content.parts[0] 

    # 2) 모델이 함수 호출 방법에 대해 function_call 타입으로 알려줌
    if part.function_call: 
        function_call =  part.function_call 
        function_name = function_call.name 
        function_args = {k: v for k, v in function_call.args.items()} 

        is_required_3_4, function_result = function_repoistory[function_name](**function_args) 

        # 3) 사용자는 function_response 타입으로 함수 호출 결과를 전달
        part = glm.Part(
              function_response=glm.FunctionResponse(
                  name=function_name, 
                  response={
                      "content": function_result
                    }
                  )
                )
        response = chat_session.send_message(part)

        if is_required_3_4:
            response = chat_session.send_message(part)
        else:
            # 3, 4단계가 필요없는 경우 make_fc_history 함수를 호출해 3, 4단계에 해당하는 구조를 chat_session의 history 객체에 삽입되게 함
            response = make_fc_history(chat_session, part, function_result)


        # 4) 모델은 text 타입으로 최종 응답을 반환
        print(f"모델: {response.candidates[0].content.parts[0].text}")