from hf_api_llm import generate_definition

test_chunk = "Benign prostatic hyperplasia (BPH) je stanje kod starijih muškaraca gdje se prostata povećava i može uzrokovati probleme s mokrenjem."
term = "Benign prostatic hyperplasia"

definition = generate_definition(test_chunk, term)
print("Definicija:\n", definition)
