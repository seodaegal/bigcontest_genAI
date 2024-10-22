import sys
import os

# bigcontest_genAI 디렉토리를 Python 모듈 경로에 추가
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from utils.config import config  # config.yaml에서 경로와 설정을 가져옴
import pandas as pd
import re

def filter_and_recommend_restaurants(input_address, purpose_choice, categorized_data_file, output_filename):
    """
    지역 정보와 목적에 따라 레스토랑을 필터링하고 CSV 파일로 저장하는 함수.
    
    Parameters:
    input_address (str): 사용자가 입력한 지역 정보 (예: '서귀포시 남원읍').
    purpose_choice (str): 사용자가 선택한 목적 (예: '상관없음', '식사', '카페/디저트').
    categorized_data_file (str): 목적에 따른 카테고리 데이터를 포함하는 CSV 파일 경로.
    output_filename (str): 필터링된 결과를 저장할 CSV 파일명.
    
    Returns:
    pd.DataFrame: 필터링된 레스토랑 정보가 포함된 DataFrame.
    """
    # Step 1: 지역 기반 필터링 (config.yaml에서 경로 가져옴)
    data = pd.read_csv(config['data']['restaurant_info_data_csv'])  # config.yaml에서 경로 사용
    filtered_by_address = data[data['address_map'].str.contains(input_address, case=False, na=False, regex=False)]
    
    # Step 2: 목적에 따른 필터링 (상관없음, 식사, 카페/디저트)
    categorized_restaurants_df = pd.read_csv(categorized_data_file)
    
    if purpose_choice == '상관없음':
        # 지역 필터링만 한 데이터를 저장하고 반환
        filtered_by_address.to_csv(output_filename, index=False)
        return filtered_by_address
    
    # 선택된 목적에 따른 카테고리 필터링
    filtered_categories = categorized_restaurants_df[categorized_restaurants_df['purpose'] == purpose_choice]['category'].values[0]
    
    # 정규 표현식을 사용하여 쉼표로 구분된 카테고리 추출 (쉼표가 포함된 경우도 포함)
    category_list = re.findall(r"'([^']*)'", filtered_categories)
    category_list = [cat.strip() for cat in category_list]
    
    # 카테고리가 매칭되는지 확인하는 함수 (쉼표 포함된 문자열을 전체로 비교)
    def category_match(restaurant_categories, category_list):
        restaurant_category = restaurant_categories.lower().strip()
        return any(restaurant_category == cat.lower().strip() for cat in category_list)
    
    # 카테고리를 기준으로 필터링
    final_filtered_data = filtered_by_address[filtered_by_address['category'].apply(lambda x: category_match(x, category_list))]
    
    # 필터링된 데이터를 CSV 파일로 저장
    final_filtered_data.to_csv(output_filename, index=False)    
    return final_filtered_data

# 사용 예시
if __name__ == "__main__":
    # 사용자 입력 예시
    input_address = '제주시 (제주특별자치도 북부)'  # 사용자가 입력한 지역 정보
    purpose_choice = '식사'  # 목적 선택 (예: '상관없음', '식사', '카페/디저트')
    
    # 수정된 파일 경로
    categorized_data_file = '../data/categorized_restaurants.csv'  # 업로드된 경로 사용
    output_filename = 'map_purpose_filtered_example.csv'  # 결과를 저장할 파일 이름

    # 레스토랑 필터링 및 추천
    filtered_recommendations = filter_and_recommend_restaurants(input_address, purpose_choice, categorized_data_file, output_filename)

    if filtered_recommendations is not None:
        print(f"Recommended restaurants saved to {output_filename}")
