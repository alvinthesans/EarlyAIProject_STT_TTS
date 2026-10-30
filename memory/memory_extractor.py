import ollama
import json
def extract_memory(user_input, ai_response):
    prompt = f"""
    Extract important long-term memory.
    
    Rules:
    - Only store useful long-term info
    - Ignore small talk
    - Return STRICT JSON.
    - Do not include explanation.
    
    Schema:
    {{
      "facts": [],
      "preferences": [],
      "decisions": []
    }}
    
    Text to look for:
    User: {user_input}
    Assistant: {ai_response}
    """

    response = ollama.chat(
        model="llama3:8b",
        messages=[{"role": "user", "content": prompt}],
        format="json"
    )
    data = json.loads(response["message"]["content"])

    if not data or not any(data.values()):
        return {}
    return data

#Turn json data into natural text chunks
def flatten_memory(memory_json):
    results = []

    for category, items in memory_json.items():
        for item in items:
            results.append(f"{category}: {item}")

    return results