import os
import json
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

class MedicalDefinitionGenerator:
    def __init__(self):
        
        self.client = InferenceClient(api_key=os.environ.get("HF_TOKEN"), bill_to="CPUI")
        self.model = "Qwen/Qwen3-235B-A22B-Instruct-2507"
        self.fallback_model = "tiiuae/falcon-7b-instruct"
        
        print(f"Using primary model: {self.model}")
        print(f"Fallback model: {self.fallback_model}")
    
    def generate_definition(self, chunk, term):
        prompt = f"""Based on the following text, write a clear, concise medical definition in English. 
                     Do not include any additional information beyond the definition. 
                     Do not translate medical terms.
                     Do not use bullet points or lists. 
                     Keep it brief and to the point.

Pojam: {term}

Tekst:
{chunk}

Definicija:"""

        messages = [
            {
                "role": "system",
                "content": "You are a doctor with specialization in urology. Your task is to provide precise, clinically verified definitions of urological terms and findings."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=200,
                temperature=0.2,
            )
            
            if hasattr(completion, 'choices') and len(completion.choices) > 0:
                return completion.choices[0].message["content"].strip()
            else:
                raise Exception("No response content received from model")
        
        except Exception as e:
            print(f"⚠️ Primary model not available ({self.model}): {e}")
            print(f"⚠️ Using fallback model: {self.fallback_model}")
           
            completion = self.client.chat.completions.create(
                model=self.fallback_model,
                messages=messages,
                max_tokens=200,
                temperature=0.2,
            )
            
            if hasattr(completion, 'choices') and len(completion.choices) > 0:
                return completion.choices[0].message["content"].strip()
            else:
                raise Exception("No response content received from fallback model")


_generator = MedicalDefinitionGenerator()

def generate_definition(chunk, term):
    return _generator.generate_definition(chunk, term)

if __name__ == "__main__":
    test_chunk = "Benign prostatic hyperplasia (BPH) je stanje kod starijih muškaraca gdje se prostata povećava i može uzrokovati probleme s mokrenjem."
    term = "Benign prostatic hyperplasia"
    definition = generate_definition(test_chunk, term)
    print("Definicija:\n", definition)