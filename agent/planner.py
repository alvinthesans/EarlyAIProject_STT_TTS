from llm.ollama_client import generate

def is_planning_request(query):
    query_lower = query.lower()

    keywords = ["step by step",
    "steps to",
    "how do i start",
    "how to start",
    "roadmap",
    "full plan",
    "detailed plan",
    "strategy"]

    if any(k in query_lower for k in keywords):
        return True

    if len(query_lower.split()) < 5:
        return False

    return llm_intent_check(query_lower)

def llm_intent_check(query):
    prompt = f"""
    You are a strict classifier.

    Only return "YES" if the user EXPLICITLY asks for:
    - step-by-step instructions
    - a full plan
    - a roadmap
    - a detailed strategy

    If the user is asking:
    - general questions
    - explanations
    - information

    RETURN "NO".

    Examples:
    "How do I start a business step by step?" → YES
    "Give me a business plan" → YES
    "What do I need to open a bar?" → NO
    "Explain how bars work" → NO
    
    Query:
    {query}
    
    Answer ONLY "YES" or "NO".
    """

    response = generate(prompt).strip().upper()
    return response.startswith("YES")

def generate_plan(user_input, memory_context, search_context, history):
    prompt = f"""
    You are a business planning assistant.
    
    Instructions:
    - Use memory if relevant
    - Be practical and concise
    - Format responses using Markdown (headings, bullet points, lists)
    
    Memory Context:
    {memory_context}
    
    History(Previous messages if applicable):
    {history}

    User Request:
    {user_input}
    
    Search Context(If applicable):
    {search_context}

    Create a clear step-by-step plan.

    Format to follow:
    Goal:
    <goal>

    Steps:
    1. ...
    2. ...
    3. ...

    Tips (if applicable):
    - ...
    """

    return prompt

def format_plan_memory(plan_text):
    return f"PLAN:\n{plan_text}"