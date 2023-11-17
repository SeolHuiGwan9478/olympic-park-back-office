import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

model = SentenceTransformer('jhgan/ko-sroberta-multitask')

df = pd.read_csv('Chatbot/한국체육산업개발주식회사_자주 묻는 질문정보_20230925.csv',encoding='cp949')

# df['제목'] = df['제목'].map(lambda x: list(model.encode(x)))
# df['내용'] = df['내용'].map(lambda x: list(model.encode(x)))

df['embedding'] = pd.Series([[]] * len(df))
df['embedding'] = df['제목'].map(lambda x: list(model.encode(x)))

data = {'제목' : df['제목'],
        '내용' : df['내용'],
        'embedding' : df['embedding']
}

df = pd.DataFrame(data)

embedding = model.encode('미사경정공원 족구장 대여')

df.to_csv('Chatbot/prep.csv',index=False)

df = pd.read_csv("Chatbot/prep.csv")

a = json.loads(df['embedding'][30])

print(cosine_similarity([a],[embedding]))

# print(df.dtypes)

# text = '드론비행을 해도 될까요?'

# embedding = model.encode(text)
# df['embedding'] = pd.to_numeric(df['embedding'])
# df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())