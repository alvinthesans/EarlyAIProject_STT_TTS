import queue
import threading
from voice.tts import speak

audio_queue = queue.Queue()

def audio_worker():
    while True:
        text = audio_queue.get()

        if text is None:
            break

        speak(text)
        audio_queue.task_done()


def start_audio_worker():
    thread = threading.Thread(target=audio_worker, daemon=True)
    thread.start()