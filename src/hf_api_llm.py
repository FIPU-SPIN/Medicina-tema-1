import os
import requests
from dotenv import load_dotenv

load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

def generate_definition(chunk, term):
    prompt = f"Please provide a clear, concise medical definition in Croatian for the term '{term}' based on the following text:\n\n{chunk}\n\nDefinition:"
    
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 200}}
    
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    
    output = response.json()
    # personaplex vraća listu sa 'generated_text'
    return output[0]['generated_text']
