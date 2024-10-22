import torch
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
from utils.config import model, tokenizer, embedding_model, device

def text2faiss(user_input, faiss_index_path, embeddings_path, df):
    faiss_index = faiss.read_index(faiss_index_path)
    embeddings = np.load(embeddings_path)
    inputs = tokenizer(user_input, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        user_embedding = embedding_model(**inputs).last_hidden_state.mean(dim=1).cpu().squeeze().numpy().astype('float32')

    # FAISS로 유사한 식당 300개 추출
    distances, indices = faiss_index.search(np.array([user_embedding]), 300)
    top_300_restaurants = df.iloc[indices[0]]

    return top_300_restaurants

def recommend_restaurant_from_subset(user_input, top_300_restaurants):
    # 전체 식당 설명을 한 번에 생성 (레스토랑 이름과 요약 정보를 사용)
    all_descriptions = "\n\n".join(
        [
            f"{restaurant['restaurant_name']}: {restaurant['text2']} (영업 시간: {restaurant['business_hours']})"
            for idx, restaurant in top_300_restaurants.iterrows()
        ]
    )

    # Gemini 모델을 위한 메시지 구성

    messages = [
        {
            "role": "model",
            "parts": [f"너는 사용자의 취향과 감정을 기반으로 제주도 맛집을 추천하는 챗봇이야. 사용자에게 맞는 식당을 대화하는 방식으로 추천해줘."]
        },
        {
            "role": "user",
            "parts": [f"{top_300_restaurants}는 식당 이름과 해당 식당에 대한 정보가 들어있는 데이터프래임이야. 사용자가 '{user_input}'라고 말했을 때 이 데이터프래임을 참고해서, 여기 있는 식당들 중에서 어떤 식당을 추천할지 3개를 골라주고, 그 이유를 설명해줘. 영업 시간을 유저가 원하는 영업 시간과 맞을 수 있게 잘 고려해주고 영업 시간을 그대로 출력해줘. 추천 식당은 최대한 겹치지 않게 해줘."]
        }
    ]

    # Gemini 1.5 Flash로 응답 생성
    response = model.generate_content(messages)
    response_text = response.text.strip()

    return response_text