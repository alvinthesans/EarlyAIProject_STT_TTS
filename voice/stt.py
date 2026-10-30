
# Load only once to prevent memory explode
model = None

def get_model():
    global model
    if model is None:
        from faster_whisper import WhisperModel
        model = WhisperModel("small", compute_type="int8")
    return model

def transcribe(audio_path):
    segments, _ = get_model().transcribe(audio_path)

    text = ""
    for segment in segments:
        text += segment.text + " "

    return text.strip()