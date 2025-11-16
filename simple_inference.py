from typing import List
from tqdm.auto import tqdm

import requests
import pandas as pd

HOST = "http://localhost:8000"

PRODUCTION_INFERENCE_URL = f"{HOST}/inference/production/full"
DEV_INFERENCE_URL = f"{HOST}/inference/dev/full"


def print_results(inference_times: List[float], label: str):
    average_time = sum(inference_times) / len(inference_times)
    print(f"Average {label} inference time: {average_time} seconds")


def test_endpoint(url: str, examples: pd.DataFrame):
    inference_times = []
    for row in tqdm(examples.to_dict(orient="records")):
        response = requests.post(url, json=row)
        inference_times.append(response.elapsed.total_seconds())

    label = "production" if "production" in url else "dev"
    print_results(inference_times, label)


def main():
    examples = pd.read_csv("data/future_unseen_examples.csv")

    for url in [PRODUCTION_INFERENCE_URL, DEV_INFERENCE_URL]:
        test_endpoint(url, examples)


if __name__ == "__main__":
    main()
