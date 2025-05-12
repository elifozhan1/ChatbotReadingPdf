import requests

def ask_ai(context, question):
    prompt = f"""
    Aşağıdaki belgeye göre soruyu yanıtla. Yanıtı soruyla aynı dilde ver:

    Belge:
    {context}

    Soru:
    {question}

    Cevap:
    """
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]
