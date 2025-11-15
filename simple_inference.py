import time
from tqdm.auto import tqdm
import requests
import pandas as pd

FULL_INFERENCE_URL = "http://localhost:8000/inference/full"

examples = pd.read_csv("data/future_unseen_examples.csv")

inference_times = []
for row in tqdm(examples.to_dict(orient="records")):
    response = requests.post(FULL_INFERENCE_URL, json=row)
    inference_times.append(response.elapsed.total_seconds())

print(f"Average inference time: {sum(inference_times) / len(inference_times)} seconds")