from ollama import chat

prompt = """
Ti si liječnik specijalist urologije.
Odgovaraj isključivo na hrvatskom jeziku.
Koristi standardnu medicinsku terminologiju.
Ne koristi kolokvijalne izraze.
Ako nisi siguran u termin, nemoj ga izmišljati.

Napiši kratku, stručno točnu definiciju pojma:

Benigna hiperplazija prostate
"""

response = chat(
    model="qwen2.5:7b",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print(response["message"]["content"])

