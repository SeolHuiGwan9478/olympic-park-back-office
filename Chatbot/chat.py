import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

# @st.cache(allow_output_mutation=True)
def cached_model():
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    return model

# @st.cache(allow_output_mutation=True)
def get_dataset():
    df = pd.read_csv('Chatbot/prep.csv')
    df['embedding'] = df['embedding'].apply(json.loads)
    return df

model = cached_model()
df = get_dataset()

while(True):
    text = input()
    # 종료 조건
    if text == 'quit':
        break
    embedding = model.encode(text)

    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]

    print(answer['내용'])