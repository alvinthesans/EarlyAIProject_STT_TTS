from faster_whisper import WhisperModel
import tempfile
import soundfile as sf
import numpy as np
import webrtcvad
import collections

model = None

def get_model():
    global model

    if model is None:
        model = WhisperModel("small", compute_type="int8")

    return model


def trim_silence(audio_path, samplerate=16000):
    # remove pauses and silence to allow accurate transcribe later
    audio, sr = sf.read(audio_path)

    #convert to mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    #convert audio to 16-bit for formatting purpose
    audio = (audio * 32768).astype(np.int16)

    #filter background noice voice detector
    vad = webrtcvad.Vad(2)

    #set the acceptad frames for audio file
    frame_duration = 30
    frame_size = int(sr * frame_duration / 1000)

    voiced_frames = []

    #slice the audio into frames
    for i in range(0, len(audio) - frame_size, frame_size):
        frame = audio[i:i + frame_size]

        try:
            speech = vad.is_speech(frame.tobytes(), sr)

            if speech:
                voiced_frames.append(frame)

        except:
            continue

    if not voiced_frames:
        return audio_path

    trimmed = np.concatenate(voiced_frames)

    temp_file = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    )

    sf.write(temp_file.name, trimmed, sr)

    return temp_file.name


def voice_input(audio_path):
    cleaned_audio = trim_silence(audio_path)

    result, info = get_model().transcribe(cleaned_audio)
    text = ""

    for seg in result:
        text += seg.text

    return text