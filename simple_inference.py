import time
from tqdm.auto import tqdm
import requests
import pandas as pd

INFERENCE_URL = "http://localhost:8000/model/inference"

examples = pd.read_csv("data/future_unseen_examples.csv")

inference_times = []
for row in tqdm(examples.to_dict(orient="records")):
    row = examples.iloc[0]
    data = row.to_dict()

    start = time.time()
    response = requests.post(INFERENCE_URL, json=data)
    end = time.time()
    inference_times.append(end - start)

print(f"Average inference time: {sum(inference_times) / len(inference_times)} seconds")