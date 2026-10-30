import time
from voice.audio_queue import audio_queue

def stream_and_speak(generator):
    buffer = ""
    full_text = ""

    for chunk in generator:
        print(chunk, end="", flush=True)
        buffer += chunk
        full_text += chunk

        if "." in buffer or "\n" in buffer or len(buffer) > 150:
            text_to_speak = buffer.strip()
            text_to_speak = clean_text(text_to_speak)
            buffer = ""

            #filter junk
            if text_to_speak and len(text_to_speak) > 5 :
                audio_queue.put(text_to_speak)

    if buffer.strip():
        if len(buffer.strip()) > 5:
            audio_queue.put(buffer.strip())
        full_text += buffer

    return full_text.strip()

def clean_text(text):
    text = text.strip()

    if text in [".", ",", "-", ""]:
        return ""

    return text