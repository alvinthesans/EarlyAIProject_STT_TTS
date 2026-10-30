import os
import shutil
import threading
import uuid
import json
import re
import gradio as gr

# ---- your modules ----
from llm.ollama_client import generate_stream
from memory.memory_extractor import extract_memory, flatten_memory
from memory.vector_store import add_memory, load_memory
from retrieval.retriever import retrieve_context
from agent.planner import is_planning_request, generate_plan
from tools.web_search import search_web, summarize_search, should_search
from voice.pipeline_ui import voice_input
from voice.control import reset_stop_flag, stop_flag
from voice.tts import generate_tts_audio
# ---- config ----
SESSION_FILE = "memData/sessions.json"
user_avatar = "img/user_avatar.png"
bot_avatar = "img/bot_avatar.png"

sessions_cache = []
# ---- session utils ----
def load_sessions():
    global sessions_cache

    if sessions_cache:
        return sessions_cache

    try:
        with open(SESSION_FILE, "r") as f:
            sessions_cache = json.load(f)["sessions"]

    except:
        sessions_cache = []

    return sessions_cache


def save_session(session):
    global sessions_cache

    sessions_cache = [
        s for s in sessions_cache
        if s["id"] != session["id"]
    ]

    sessions_cache.append(session)

    with open(SESSION_FILE, "w") as f:
        json.dump(
            {"sessions": sessions_cache},
            f,
            indent=2
        )


def group_sessions():
    sessions = load_sessions()
    grouped = {}

    for s in sessions:
        folder = s.get("folder", "General")
        grouped.setdefault(folder, []).append(s)

    return grouped


def rename_chat(new_title):
    global current_session

    if not new_title.strip():
        return gr.update()

    current_session["title"] = new_title.strip()
    save_session(current_session)

    return refresh_chat_list(current_session["folder"])


def delete_chat():
    global current_session
    global sessions_cache

    sessions_cache = [
        s for s in sessions_cache
        if s["id"] != current_session["id"]
    ]

    with open(SESSION_FILE, "w") as f:
        json.dump({"sessions": sessions_cache}, f, indent=2)

    current_session = {
        "id": str(uuid.uuid4()),
        "title": "New Chat",
        "folder": "General",
        "messages": []
    }

    grouped = group_sessions()

    titles = [
        s["title"]
        for s in grouped.get("General", [])
    ]

    return [], gr.update(choices=titles, value=None)


def refresh_chat_list(folder):
    grouped = group_sessions()
    sessions = grouped.get(folder, [])
    titles = [s["title"] for s in sessions]

    return gr.update(choices=titles)


# ---- helpers ----
def format_history_for_prompt(history):
    return "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in history
    )


# Auto filing system
def classify_folder(user_input):
    prompt = f"""
    Classify this request into ONE of these folders:

    - My Ideas (business ideas, starting something, plans)
    - Research Notes (facts, regulations, market info)
    - General (everything else)

    Only return EXACTLY one of:
    My Ideas
    Research Notes
    General

    Input:
    {user_input}

    Folder:
    """

    try:
        from llm.ollama_client import generate
        result = generate(prompt).strip()

        # safety normalize
        if "idea" in result.lower():
            return "My Ideas"
        elif "research" in result.lower():
            return "Research Notes"
        else:
            return "General"

    except:
        return "General"


def async_folder_classification(user_input, session_id):

    def run():
        folder = classify_folder(user_input)

        sessions = load_sessions()

        for s in sessions:
            if s["id"] == session_id:
                s["folder"] = folder

        with open(SESSION_FILE, "w") as f:
            json.dump({"sessions": sessions}, f, indent=2)

    threading.Thread(target=run).start()


# Create titles for new chat after first input
def generate_chat_title(user_input):
    prompt = f"""
    Create a SHORT title (max 6 words) for this conversation.

    Rules:
    - No punctuation at the end
    - No quotes
    - Keep it simple and clear

    Example:
    "I want to open a cocktail bar in KL" → Cocktail Bar Business Idea

    Input:
    {user_input}

    Title:
    """

    try:
        from llm.ollama_client import generate
        title = generate(prompt).strip()

        # clean safety
        title = title.replace('"', '').strip()
        return title[:50]

    except:
        # fallback
        return user_input[:40]


def async_title_generation(user_input, session_id):

    def run():
        title = generate_chat_title(user_input)

        sessions = load_sessions()

        for s in sessions:
            if s["id"] == session_id:
                s["title"] = title

        with open(SESSION_FILE, "w") as f:
            json.dump({"sessions": sessions}, f, indent=2)

    threading.Thread(target=run).start()


def clean_response(text):
    text = text.strip()
    text = re.sub(r"\s+([.,!?])", r"\1", text)

    if "Goal:" in text and "Steps:" in text:
        text = re.sub(r"\bGoal:\s*", "\n\n## 🎯 Goal\n", text)
        text = re.sub(r"\bSteps?:\s*", "\n\n## 🪜 Steps\n", text)

    text = re.sub(r"(?<=\n)(\d+\.)\s*", r"\n\1 ", text)
    text = re.sub(r"\n[-o]\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def run_agent(user_input, history):
    memory_context = retrieve_context(user_input)

    search_context = ""
    if should_search(user_input):
        raw = search_web(user_input)
        search_context = summarize_search(raw)

    if is_planning_request(user_input):
        return generate_plan(user_input, memory_context, search_context, history)

    return f"""
    You are a business assistant.

    Conversation History:
    {format_history_for_prompt(history)}

    Relevant Memory:
    {memory_context}

    Web info:
    {search_context}

    Question:
    {user_input}

    Instructions:
    - Be practical and clear
    - Use markdown when useful
    - Only use structured plans when explicitly asked
    """


# ---- speech ----
def extract_speakable_text(text):
    # remove markdown
    text = re.sub(r"[*#`_>-]", "", text)

    # fix ranges
    text = re.sub(r"(\d+)-(\d+)", r"\1 to \2", text)

    # collapse whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def smart_expand(text):
    return text


def handle_speech(text, speak):
    if not speak:
        return None

    if stop_flag.is_set():
        return None

    text = extract_speakable_text(text)
    text = smart_expand(text)

    if text.strip().endswith("▌"):
        return None

    if len(text.split()) < 6:
        return None

    return generate_tts_audio(text)


# ---- sidebar logic ----
def update_chat_list(folder):
    grouped = group_sessions()
    sessions = grouped.get(folder, [])
    return gr.update(choices=[s["title"] for s in sessions])


def load_chat(title, folder):
    global current_session

    grouped = group_sessions()
    sessions = grouped.get(folder, [])

    for s in sessions:
        if s["title"] == title:
            current_session = s
            return s["messages"], s["title"]

    return [], ""


def new_chat():
    global current_session

    current_session = {
        "id": str(uuid.uuid4()),
        "title": "New Chat",
        "folder": "General",
        "messages": []
    }

    return [], gr.update(value="General"), gr.update(choices=[])


# ---- main pipeline ----
def process_input(user_input, history, speak, folder):
    if not user_input:
        yield "", history, gr.skip()
        return

    reset_stop_flag()
    current_session["folder"] = folder

    history.append({"role": "user", "content": user_input})
    yield "", history, gr.skip()

    if not current_session["messages"]:
        async_title_generation(user_input, current_session["id"])

    if not current_session["messages"] and current_session["folder"] == "General":
        async_folder_classification(user_input, current_session["id"])

    history.append({"role": "assistant", "content": "⏳ Thinking..."})
    yield "", history, gr.skip()

    prompt = run_agent(user_input, history)

    reset_stop_flag()

    history[-1]["content"] = ""
    full_text = ""

    for chunk in generate_stream(prompt=prompt):
        if stop_flag.is_set():
            history[-1]["content"] += "\n\n⏹ Stopped"
            break

        full_text += chunk

        history[-1]["content"] = clean_response(full_text) + "▌"
        yield "", history, gr.skip()

    history[-1]["content"] = clean_response(full_text)
    full_text = full_text.replace("▌", "")

    if not full_text.strip():
        history[-1]["content"] = "Response stopped."

    audio_path = handle_speech(full_text, speak)

    memory_json = extract_memory(user_input, full_text)
    if memory_json:
        for chunk in flatten_memory(memory_json):
            add_memory(chunk)

    current_session["messages"] = history
    save_session(current_session)

    yield "", history, gr.update(value=audio_path)


def voice_chat(audio, history, speak, folder):
    if not audio:
        yield "", history, gr.skip()
        return

    text = voice_input(audio)

    yield from process_input(
        text,
        history,
        speak,
        folder
    )


def stop_all():
    stop_flag.set()

    return gr.update(
        value=None
    )


# ---- UI ----
with gr.Blocks() as demo:
    gr.Markdown("""
    ## 🤖 AI Business Assistant for Moon ❤️
    A mother day's gift created in 2026!
    """)

    with gr.Row():
        # sidebar
        with gr.Column(scale=1):
            gr.Markdown("### 📂 Chats")

            folder_dropdown = gr.Dropdown(
                ["General", "My Ideas", "Research Notes"],
                value="General",
                label="Folder"
            )

            chat_list = gr.Radio(choices=[], label="Chats")

            rename_box = gr.Textbox(
                placeholder="Rename chat...",
                label="✏️ Rename"
            )

            rename_btn = gr.Button("Apply")

            delete_btn = gr.Button("🗑 Delete Chat")

            new_chat_btn = gr.Button("➕ New Chat")

        # main
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                height=450,
                avatar_images=(user_avatar, bot_avatar),
                layout="bubble"
            )
            voice_status = gr.Markdown("")

            with gr.Row():
                msg = gr.Textbox(scale=4)
                speak_toggle = gr.Checkbox(value=True, label="🔊 Speak")

                audio_output = gr.Audio(
                    autoplay=True,
                    elem_id="hidden-audio"
                )

            with gr.Row():
                send_btn = gr.Button("Send")
                mic_input = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="🎤 Speak"
                )

                stop_btn = gr.Button("⏹ Stop")

            with gr.Row():
                gr.Button("💡 Help me start a business").click(
                    lambda: "I want to start a business, help me plan it",
                    None,
                    msg
                )

                gr.Button("📊 Check my idea").click(
                    lambda: "Is this business idea good:",
                    None,
                    msg
                )

    folder_dropdown.change(update_chat_list, folder_dropdown, chat_list)

    chat_list.change(
        load_chat,
        [chat_list, folder_dropdown],
        [chatbot, rename_box]
    )

    send_btn.click(
        process_input,
        [msg, chatbot, speak_toggle, folder_dropdown],
        [msg, chatbot, audio_output]
    )

    msg.submit(
        process_input,
        [msg, chatbot, speak_toggle, folder_dropdown],
        [msg, chatbot, audio_output]
    )

    new_chat_btn.click(
        new_chat,
        None,
        [chatbot, folder_dropdown, chat_list]
    )

    rename_btn.click(
        rename_chat,
        rename_box,
        chat_list
    )

    mic_input.stop_recording(
        voice_chat,
        [mic_input, chatbot, speak_toggle, folder_dropdown],
        [msg, chatbot, audio_output]
    )

    delete_btn.click(
        delete_chat,
        None,
        [chatbot, chat_list]
    )

    stop_btn.click(
        stop_all,
        None,
        audio_output
    )
    demo.load(update_chat_list, folder_dropdown, chat_list)

# ---- start ----
if __name__ == "__main__":
    load_memory()
    if os.path.exists("temp_audio"):
        shutil.rmtree("temp_audio")

    os.makedirs("temp_audio", exist_ok=True)

    from memory.vector_store import get_model
    from voice.stt import get_model as create_stt
    from voice.tts import get_tts

    current_session = {
        "id": str(uuid.uuid4()),
        "title": "New Chat",
        "folder": "General",
        "messages": []
    }

    get_model()
    create_stt()
    get_tts()

    demo.queue().launch(theme=gr.themes.Soft(), server_name="0.0.0.0", css="""
    .bubble {
        max-width: 95% !important;
    }

    .message {
        line-height: 1.6;
        font-size: 16px;
        padding: 12px 16px;
    }

    .markdown h2 {
        margin-top: 10px;
    }
    
    #hidden-audio {
    height: 0px !important;
    min-height: 0px !important;
    opacity: 0;
    overflow: hidden;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
    }
    """)

