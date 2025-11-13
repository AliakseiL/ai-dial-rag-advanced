import json

import requests

DIAL_EMBEDDINGS = 'https://ai-proxy.lab.epam.com/openai/deployments/{model}/embeddings'


# ---
# https://dialx.ai/dial_api#operation/sendEmbeddingsRequest
# ---
# Implement DialEmbeddingsClient:
# - constructor should apply deployment name and api key
# - create method `get_embeddings` that will generate embeddings for input list (don't forget about dimensions)
#   with Embedding model and return back a dict with indexed embeddings (key is index from input list and value vector list)

class DialEmbeddingsClient:
    def __init__(self, deployment_name: str, api_key: str):
        self.deployment_name = deployment_name
        self.api_key = api_key
        self.url = DIAL_EMBEDDINGS.format(model=self.deployment_name)
        self.headers = {
            'Content-Type': 'application/json',
            'Api-Key': self.api_key
        }

    def get_embeddings(self, input_list, dimensions):
        payload = {
            "input": input_list,
            "model": self.deployment_name,
            "dimensions": dimensions
        }
        response = requests.post(self.url, headers=self.headers, data=json.dumps(payload))
        response.raise_for_status()
        response_data = response.json()

        embeddings_dict = {}
        for item in response_data.get("data", []):
            index = item["index"]
            embedding_vector = item["embedding"]
            embeddings_dict[index] = embedding_vector

        return embeddings_dict


# Hint:
#  Response JSON:
#  {
#     "data": [
#         {
#             "embedding": [
#                 0.19686688482761383,
#                 ...
#             ],
#             "index": 0,
#             "object": "embedding"
#         }
#     ],
#     ...
#  }
