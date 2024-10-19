# 질문에다 식당을 text1로 변환, 지금은 사용 안함

import os
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import google.generativeai as genai

# Gemini configuration
GOOGLE_API_KEY = "AIzaSyALTtMw0gouvDX08pmc0JqaWq_BhXm6qSU"  # Replace with your actual Google API key
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# Load CSV data
data_path = '/root/shcard_bigcontest2024_llm/data'
csv_file_path = "JEJU_MCT_DATA_modified.csv"
df = pd.read_csv(os.path.join(data_path, csv_file_path))

# Create restaurant_data
restaurant_data = []

for index, row in df.iterrows():
    restaurant_info = f"""
    {row['가맹점명']}은(는) {row['가맹점업종']}으로, 주소는 {row['가맹점주소']}입니다. 
    이용건수 구간: {row['이용건수구간']}, 이용금액 구간: {row['이용금액구간']}, 
    건당 평균 이용 금액 구간: {row['건당평균이용금액구간']}.

    요일별 이용 건수 비중:
    월요일: {row['월요일이용건수']}%, 화요일: {row['화요일이용건수']}%, 
    수요일: {row['수요일이용건수']}%, 목요일: {row['목요일이용건수']}%, 
    금요일: {row['금요일이용건수']}%, 토요일: {row['토요일이용건수']}%, 
    일요일: {row['일요일이용건수']}%.

    시간대별 이용 건수 비중:
    오전 5시부터 11시까지: {row['5시~11시이용건수비중']}%, 
    오후 12시부터 1시까지: {row['12시~13시이용건수비중']}%, 
    오후 2시부터 5시까지: {row['14시~17시이용건수비중']}%, 
    오후 6시부터 10시까지: {row['18시~22시이용건수비중']}%, 
    오후 11시부터 오전 4시까지: {row['23시~4시이용건수비중']}%.

    현지인 이용 건수 비중: {row['현지인이용건수비중']}%.

    최근 12개월 성별 회원수 비중:
    남성: {row['최근12개월남성회원수비중']}%, 
    여성: {row['최근12개월여성회원수비중']}%.

    연령대별 회원수 비중:
    20대 이하: {row['최근12개월20대이하회원수비중']}%, 
    30대: {row['최근12개월30대회원수비중']}%, 
    40대: {row['최근12개월40대회원수비중']}%, 
    50대: {row['최근12개월50대회원수비중']}%, 
    60대 이상: {row['최근12개월60대이상회원수비중']}%.

    영업시간: {row['영업시간']}.

    """
    restaurant_data.append(restaurant_info.strip())

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hugging Face embedding model
model_name = "jhgan/ko-sroberta-multitask"
tokenizer = AutoTokenizer.from_pretrained(model_name)
embedding_model = AutoModel.from_pretrained(model_name).to(device)

# Embed text
def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        embeddings = embedding_model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.squeeze().cpu().numpy()

# Main function to generate response using Gemini
def generate_response_with_text(question, restaurant_data, model, embed_text):
    # Generate embeddings for all restaurant descriptions
    embeddings = np.array([embed_text(restaurant) for restaurant in restaurant_data])

    # Embed the question
    query_embedding = embed_text(question).reshape(1, -1)

    # Here, you can implement a similarity search based on embeddings
    # For example, using cosine similarity to find the closest match
    similarities = np.dot(embeddings, query_embedding.T).squeeze()
    top_indices = similarities.argsort()[-3:][::-1]  # Get top 3 results

    # Prepare the reference information
    reference_info = "\n".join([restaurant_data[i] for i in top_indices])
    prompt = f"질문: {question}\n참고할 정보:\n{reference_info}\n응답:"

    response = model.generate_content(prompt)
    return response

# Example usage
if __name__ == "__main__":
    question = "제주도에서 월요일 이용건수가 제일 많은 식당 추천해줘"
    response = generate_response_with_text(question, restaurant_data, model, embed_text)
    print(response)