from huggingface_hub import InferenceClient
import os

client = InferenceClient(api_key=os.environ.get("HF_TOKEN"))
print(client.whoami())
