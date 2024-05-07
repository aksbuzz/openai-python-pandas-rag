import tiktoken

tokenizer = tiktoken.get_encoding("cl100k_base")

# Function to split text into chunks of max_tokens
def split_into_many(text: str, max_tokens: int) -> list:
  sentences = text.split(". ")

  n_tokens = [len(tokenizer.encode(" " + s)) for s in sentences]
  
  chunks = []
  tokens_so_far = 0
  chunk = []
  
  for (sentence, token) in zip(sentences, n_tokens):
    if tokens_so_far + token > max_tokens:
      chunks.append(". ".join(chunk) + ".")
      chunk = []
      tokens_so_far = 0
    
    if token > max_tokens:
      continue

    chunk.append(sentence)
    tokens_so_far += token + 1

  return chunks