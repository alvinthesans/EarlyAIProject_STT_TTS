import sounddevice as sd
import numpy as np
import webrtcvad
import collections
import scipy.io.wavfile as wav

def record_audio(filename="input.wav", samplerate=16000):
    # record audio with the web permission of microphone
    vad = webrtcvad.Vad(2)

    frame_duration = 30
    frame_size = int(samplerate * frame_duration / 1000)

    ring_buffer = collections.deque(maxlen=10)
    triggered = False

    frames = []

    def is_speech(frame):
        return vad.is_speech(frame.tobytes(), samplerate)

    print("🎤 Listening... (speak now)")

    with sd.InputStream(samplerate=samplerate, channels=1, dtype='int16') as stream:
        while True:
            audio_chunk, _ = stream.read(frame_size)

            speech = is_speech(audio_chunk)

            if not triggered:
                ring_buffer.append((audio_chunk, speech))

                if sum(1 for f, s in ring_buffer if s) > 0.6 * ring_buffer.maxlen:
                    triggered = True
                    print("🟢 Speech detected")

                    for f, _ in ring_buffer:
                        frames.append(f)

                    ring_buffer.clear()

            else:
                frames.append(audio_chunk)
                ring_buffer.append((audio_chunk, speech))

                if sum(1 for f, s in ring_buffer if not s) > 0.8 * ring_buffer.maxlen:
                    print("🔴 Silence detected, stopping")
                    break

    audio = np.concatenate(frames, axis=0)
    wav.write(filename, samplerate, audio)

    return filename