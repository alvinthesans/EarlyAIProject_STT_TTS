from voice.recorder import record_audio
from voice.stt import transcribe

def voice_input():
    audio_file = record_audio()
    text = transcribe(audio_file)

    print("You(Spoken):", text)

    return text
