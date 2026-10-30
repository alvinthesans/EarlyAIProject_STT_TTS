from ddgs import DDGS

from llm.ollama_client import generate

seen_links = []
def search_web(query, max_results=5):
    with DDGS() as ddgs:
        global seen_links
        result1 = list(ddgs.text(query, max_results=max_results))
        seen_links = {item['href'] for item in result1}

        content = {item['body'] for item in result1}

    prompt = f"""
    Decide whether the searched content are accurate with minimal contradicting info.
    Also consider if the information match the query or question.
    If the content is trustworthy and match the query, answer "YES", else answer "NO"
    RETURN ONLY "YES" OR "NO"
    
    Searched Content:
    {content}
    
    Query:
    {query}
    """

    response = generate(prompt).strip().upper()

    if response == "YES":
        seen_links.clear()
        return "\n".join(content)
    else:
        result2 = search_web_depth2(query)
        finalResult = {*result2, *content}
        return "\n".join(finalResult)

def search_web_depth2(query, max_results = 8):
    with DDGS() as ddgs:
        global seen_links
        result2 = list(ddgs.text(query, max_results=max_results, timelimit='m'))
        new_info = [item for item in result2 if item["href"] not in seen_links]

        content = {item['body'] for item in new_info}
        return content

def should_search(query):
    keywords = ["latest", "market", "price", "trend", "cost", "competitor", "search"]

    query_lower = query.lower()

    if any(k in query.lower() for k in keywords):
        return True

    return llm_intent_check(query_lower)


def llm_intent_check(query):
    prompt = f"""
    Classify the user query.

    Return ONLY "YES" or "NO".

    If the request ask for:
    - Perform a search for factual information
    - Requires information from the internet
    - Needs evidence to prove points

    RETURN "YES", else return "NO"

    Query:
    {query}
    """

    response = generate(prompt).strip().upper()
    return "YES" in response

def summarize_search(results):
    prompt = f"""
    Summarize the key facts from the following information:
    {results}
    
    Keep it concise and factual, DO NOT INCLUDE YOUR THINKING PROCESS
    """

    return generate(prompt)

