import os
import uuid

import sounddevice as sd
from voice.control import stop_flag
import numpy as np
import time


#Load only once, do not duplicate
tts = None


def get_tts():
    import torch

    _original_torch_load = torch.load

    def custom_torch_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _original_torch_load(*args, **kwargs)

    torch.load = custom_torch_load

    global tts
    if tts is None:
        from TTS.api import TTS
        tts = TTS(model_name="tts_models/en/ljspeech/fast_pitch")
    return tts

#local version
def speak(text):
    print("Speaking...")
    audio = get_tts().tts(text, speaker="p240")

    audio = np.array(audio, dtype=np.float32)
    samplerate = 22050

    if len(audio.shape) > 1:
        audio = audio.squeeze()

    sd.play(audio, samplerate=samplerate)

    while sd.get_stream() and sd.get_stream().active:
        if stop_flag.is_set():
            sd.stop()
            return
        time.sleep(0.05)

#Web version
def generate_tts_audio(text):
    os.makedirs("temp_audio", exist_ok=True)

    filename = f"temp_audio/{uuid.uuid4()}.wav"

    get_tts().tts_to_file(
        text=text,
        speed=1.15,
        file_path=filename
    )

    return filename
