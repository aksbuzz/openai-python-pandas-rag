import numpy as np
import pandas as pd

from openai import OpenAI
from scipy.spatial.distance import cosine

max_len = 1800
openai_client = OpenAI(
  api_key=""
)

def create_context(question: str, df: pd.DataFrame):
  q_embeddings = openai_client.embeddings.create(input=question, model="text-embedding-3-small").data[0].embedding
  df['distances'] = df['embeddings'].apply(lambda x: cosine(x, q_embeddings))

  returns = []
  cur_len = 0

  for i, row in df.sort_values('distances', ascending=True).iterrows():
    cur_len += row['n_tokens'] + 4

    if cur_len > max_len:
      break

    returns.append(row['description'])

  return "\n\n###\n\n".join(returns)

def answer_question(question: str, df: pd.DataFrame, debug=False):
  context = create_context(question, df)
  print(context)

  if debug:
    print("Context:\n" + context)
    print("\n\n")

  try:
    response = openai_client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "system", "content": "Answer the question based on the context below, and be descriptive but not too much, and if the question can't be answered based on the context, say \"I don't know\"\n\n"},
        {"role": "user", f"content": f"Context: {context}\n\n---\n\nQuestion: {question}\nAnswer:"}
      ],
      temperature=0,
      max_tokens=300,
    )
    return response.choices[0].message.content
  except Exception as e:
    print(e)

def main():
  df = pd.read_csv("bikes_with_embeddings.csv")
  df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
  df.head()

  answer = answer_question("Which bike is manufactured by nHill?", df)
  print(answer)

if __name__ == "__main__":
  main()