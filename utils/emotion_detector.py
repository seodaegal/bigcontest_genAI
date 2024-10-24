from utils.config import model

def detect_emotion_and_context(question):
    """
    Detects whether the given question refers to specific elements related to restaurant data.
    If the question covers all the predefined items, it returns '1', otherwise, it returns '2'.

    Parameters:
    - question (str): The input question to analyze.
    - model (GenerativeModel): The pre-configured Generative AI model.

    Returns:
    - type (str): '1' if the question refers to all the predefined items, '2' otherwise.
    """

    
    prompt = f"""
    '{question}' 이 질문이 묻는 항목이 다음 리스트에 모두 포함되어있으면 1, 리스트에 포함되어 있지 않는 항목을 묻고있으면 2를 출력해주세요.
    [개설일자, 판매음식종류, 지역(_시 _동), 이용건수(상위 _%), 이용금액(상위 _%), 건당평균이용금액(상위 _%),
    월요일이용건수비중,화요일이용건수비중,수요일이용건수비중,목요일이용건수비중,금요일이용건수비중,토요일이용건수비중,일요일이용건수비중,
    5시부터11시이용건수비중,12시부터13시이용건수비중,14시부터17시이용건수비중,18시부터22시이용건수비중,23시부터4시이용건수비중,
    현지인이용비중,최근12개월남성회원수비중,최근12개월여성회원수비중,20대이하고객수비중,30대고객수비중,40대고객수비중,50대고객수비중,60대이상고객수비중] 

    만약 1이면 밑에다 정보를 각 항목마다 항목이 높아야 되는지 낮아야되는지, 판매음식종류는 주어진 리스트 중 하나, 지역같은 정확한 정보가 있는지 정리하세요.: {question} 질문이 묻는 정보가 아니라면 항목에 '해당없음'이라고 적어주세요.\n개설일자:,판매음식종류:이중 하나['가정식', '단품요리 전문', '커피', '베이커리', '일식', '치킨', '중식', '분식', '햄버거', '양식', '맥주/요리주점', '아이스크림/빙수', '피자', '샌드위치/토스트', '차', '꼬치구이', '기타세계요리', '구내식당/푸드코트', '떡/한과', '도시락', '도너츠', '주스', '동남아/인도음식', '패밀리 레스토랑', '기사식당', '야식', '스테이크', '포장마차', '부페', '민속주점']중 하나,지역:(_시 _동),이용건수:(상위 _%),이용금액:(상위 _%), 건당평균이용금액:(상위 _%),월요일이용건수비중:,화요일이용건수비중:,수요일이용건수비중:,목요일이용건수비중:,금요일이용건수비중:,토요일이용건수비중:,일요일이용건수비중:,현지인이용비중:,20대이하고객수비중:,30대고객수비중:,40대고객수비중:,50대고객수비중:,60대이상고객수비중:
    만약 2인 경우 줄 바꿈없이 그냥 2만 출력하세요.
    """
    response = model.generate_content(prompt)

    if hasattr(response, 'candidates') and response.candidates:
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if hasattr(part, 'text'):
                    type=part.text
    
    return type# 감정과 상황을 텍스트로 리턴