import ollama

def generate(prompt, model="qwen3:8b", system=None):
    messages = []

    if system:
        messages.append({"role": "system", "content": system})

    messages.append({"role": "user", "content": prompt})

    response = ollama.chat(
        model = model,
        messages = messages,
        options = {
            "temperature": 0.3
        }
    )

    return response["message"]["content"]

def generate_stream(prompt, model="llama3:8b", system=None):
    messages = []

    if system:
        messages.append({"role": "system", "content": system})

    messages.append({"role": "user", "content": prompt})

    response = ollama.chat(
        model = model,
        messages = messages,
        stream= True,
        options = {
            "temperature": 0.3
        }
    )

    for chunk in response:
        yield chunk["message"]["content"]