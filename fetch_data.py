import requests
import pandas as pd

url = "https://raw.githubusercontent.com/bsbodden/redis_vss_getting_started/main/data/bikes.json"

def fetchData():
  response = requests.get(url)
  bikes = response.json()

  # Convert files to dataframe and write to csv
  df = pd.DataFrame(bikes, columns=["model", "description"])
  df["description"] = df.model + ' ' + df.description
  df.to_csv("bikes.csv", index=False)

