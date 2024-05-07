import pandas as pd
import tiktoken
from split_into_many import split_into_many

tokenizer = tiktoken.get_encoding("cl100k_base")

def load_and_chunk(csv: str) -> pd.DataFrame:
  df = pd.read_csv(csv)
  df.columns = ["model", "description"]

  df['n_tokens'] = df.description.apply(lambda x: len(tokenizer.encode(x)))

  shortened = []

  for row in df.iterrows():
    if row[1]['description'] is None:
      continue
      
    if row[1]['n_tokens'] > 300:
      shortened += split_into_many(row[1]['description'], 300)
      
    else:
      shortened.append(row[1]['description'])
      
  df = pd.DataFrame(shortened, columns=["description"])
  df['n_tokens'] = df.description.apply(lambda x: len(tokenizer.encode(x)))

  return df