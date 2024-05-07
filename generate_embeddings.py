import pandas as pd
from openai import OpenAI

openai_client = OpenAI(
  api_key=""
)

def generate_embeddings(df: pd.DataFrame) -> pd.DataFrame:
  df['embeddings'] = df.description.apply(lambda x: openai_client.embeddings.create(input=x, model="text-embedding-3-small").data[0].embedding)

  df.to_csv("bikes_with_embeddings.csv", index=False)

  return df