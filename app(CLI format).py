from llm.ollama_client import generate, generate_stream
from memory.memory_extractor import extract_memory, flatten_memory
from memory.vector_store import add_memory, load_memory
from retrieval.retriever import retrieve_context
from agent.planner import is_planning_request, generate_plan, format_plan_memory
from tools.web_search import search_web, summarize_search
from tools.web_search import should_search
from voice.pipeline import voice_input
from voice.audio_queue import start_audio_worker
from voice.stream_speaker import stream_and_speak
from voice.control import reset_stop_flag
import re
import threading

CHAT_HISTORY_LIMIT = 6
chat_history = []

def build_prompt(user_input, memory_context, search_context):
    history_text = "\n".join(chat_history)

    prompt = f"""
    You are a business assistant.
    
    Conversation History:
    {history_text}
    
    Relevant Memory:
    {memory_context}
    
    Web info:
    {search_context}
    
    Current Question:
    {user_input}
    
    Instructions:
    - Use memory if relevant
    - Be practical and concise
    - If unsure, say "I don't know"
    """

    return prompt

def update_chat_history(user_input, response):
    global chat_history

    chat_history.append(f"User: {user_input}")
    chat_history.append(f"Assistant: {response}")

    # Keep only last N messages ,e.g. 6
    if len(chat_history) > CHAT_HISTORY_LIMIT:
        chat_history = chat_history[-CHAT_HISTORY_LIMIT:]

def clean_response(text):
    text = text.strip()

    # remove space before punctuation
    text = re.sub(r"\s+([.,!?])", r"\1", text)

    # normalize spacing
    text = re.sub(r"\s+", " ", text)

    # fix newlines
    text = re.sub(r"\n+", "\n", text)

    return text

def chat():
    print("AI Assistant (type 'exit' to quit)\n")

    while True:
        search_context = ""

        #Temporary without UI buttons
        mode = input("Type or Speak? (t/s): ").lower().strip()
        if mode == "s":
            user_input = voice_input()
        else:
            user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        memory_context = retrieve_context(user_input)

        if should_search(user_input):
            raw_results = search_web(user_input)
            search_context = summarize_search(raw_results)

        #Plan mode
        if is_planning_request(user_input):
            prompt = generate_plan(user_input, memory_context, search_context)
        else:
            #Normal pipeline/response
            prompt = build_prompt(user_input, memory_context, search_context)

        user_voice = input("🔊 Speak response? (y/n): ").lower().strip()

        if user_voice == "y":
            print("\nAI:", end="")
            reset_stop_flag()

            generator = generate_stream(prompt=prompt)
            response_container = {"text": ""}

            def run_speech():
                reset_stop_flag()
                response_container["text"] = stream_and_speak(generator)

            speech_thread = threading.Thread(target=run_speech)
            speech_thread.start()

            from voice.control import stop_speaking
            skipped = False

            while speech_thread.is_alive():
                user_skip = input("⏹ Type 'skip' to stop speech: ").lower().strip()

                if user_skip == "skip":
                    stop_speaking()
                    skipped = True
                    break

            speech_thread.join()
            response = clean_response(response_container["text"])

            if skipped:
                print("\n⏹ Speech stopped.")

            print("\nAI:", response)
        else:
            response = generate(prompt=prompt)
            print("\nAI:", response)

        update_chat_history(user_input, response)

        memory_json = extract_memory(user_input, response)

        if memory_json:
            memory_chunks = flatten_memory(memory_json)

            for chunk in memory_chunks:
                add_memory(chunk)


if __name__ == "__main__":
    start_audio_worker()
    load_memory()
    chat()