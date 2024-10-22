from utils.config import model, df, text2_df, config 
from utils.sql_utils import convert_question_to_sql, execute_sql_query_on_df
from utils.faiss_utils import load_faiss_index, embed_text
from utils.emotion_detector import detect_emotion_and_context
from utils.response_generator import generate_response_with_faiss, generate_gemini_response_from_results
from utils.text2 import text2faiss, recommend_restaurant_from_subset
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os


def main():
    question = "제주시 노형동에 있는 단품요리 전문점 중 이용건수가 상위 10%에 속하고 현지인 이용 비중이 가장 높은 곳은?"  # Example user question

    
    which_csv = detect_emotion_and_context(question)
    print("이 질문은" + which_csv)

    if int(which_csv) == 1:
        # Convert question to SQL
        sql_query = convert_question_to_sql(question)
        print(f"Generated SQL Query: {sql_query}")  # Debugging: Print the generated SQL query

        # Execute SQL query on DataFrame
        sql_results = execute_sql_query_on_df(sql_query, df)

        if sql_results.empty:
            print("SQL query failed or returned no results. Falling back to FAISS.")

            embeddings_path = config['faiss']['embeddings_path'] 

            embeddings = np.load(embeddings_path)
            response = generate_response_with_faiss(question, df, embeddings, model, embed_text)
            print(response)
        else:
            response = generate_gemini_response_from_results(sql_results, question)
            print(response)

    elif int(which_csv) == 2: 

        index_path = config['faiss']['text2_faiss_index']
        embeddings_path = config['faiss']['text2_embeddings']


        top_300 = text2faiss(question, index_path, embeddings_path, text2_df)
        response = recommend_restaurant_from_subset(question, top_300)
        print(response)

    else:
        print("Error in classifying question type")

if __name__ == "__main__":
    main()
