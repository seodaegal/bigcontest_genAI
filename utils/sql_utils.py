import pandasql as ps
import pandas as pd
from utils.config import model

# Function to convert question to SQL
def convert_question_to_sql(question):
    prompt = f"다음 질문에서 묻는 정보를 각 항목마다 항목이 높아야 되는지 낮아야되는지 카페나 지역같은 정확한 정보가 있는지 정리하세요.: {question} 질문이 묻는 정보가 아니라면 항목에 '해당없음'이라고 적어주세요.\n개설일자:,판매음식종류:(단품요리 전문, 치킨, 카페, 일식,...),지역:(_시 _동),이용건수:(상위 _%),이용금액:(상위 _%), 건당평균이용금액:(상위 _%),월요일이용건수비중:,화요일이용건수비중:,수요일이용건수비중:,목요일이용건수비중:,금요일이용건수비중:,토요일이용건수비중:,일요일이용건수비중:,현지인이용비중:,20대이하고객수비중:,30대고객수비중:,40대고객수비중:,50대고객수비중:,60대이상고객수비중:"
    response = model.generate_content(prompt)

    if hasattr(response, 'candidates') and response.candidates:
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if hasattr(part, 'text'):
                    response= part.text
    print(response)
    
    sql_prompt= f'너는 텍스트가 주어지면 sql query를 짜는 모델이야. 다음 칼럼명들과 칼럼들이 높아야되는지, 낮아야되는지에, 혹은 상관없음이면 그냥 두는 sql query를 생성해줘. 비중이나 건수 같은 수치가 아니라 개설일자, 판매음식종류, 지역 같은 칼럼에 대해서 물으면 그거는 구체적으로 LIKE을 써줘. {response}. 쿼리를 돌릴 테이블 명은 df이고 칼럼명은 위에서 알려주는 칼럼명과 동일하게 한국어로 되어있고 모두 띄어쓰기가 안되어있어 backticks이나 double quote 빠트리지마. 쿼리에 대한 설명은 주지말고 쿼리만 줘.'
    to_sql= model.generate_content(sql_prompt)
    # Extracting the SQL query from the response
    sql_query = to_sql.candidates[0].content.parts[0].text
    
    # Removing the ```sql and ``` markers
    sql_query_cleaned = sql_query.strip().strip("```sql").strip("```").strip()
    return sql_query_cleaned

# Function to execute SQL query on DataFrame using Pandas
def execute_sql_query_on_df(query, df):
    try:
        # Use pandasql to execute the query on the DataFrame
        result_df = ps.sqldf(query, locals())
        return result_df
    except Exception as e:
        print(f"Error executing query: {e}")
        return pd.DataFrame()
    
