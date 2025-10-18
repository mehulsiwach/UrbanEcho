import numpy as np
import librosa
import tensorflow_hub as hub

YAMNET_MODEL = hub.load("https://tfhub.dev/google/yamnet/1")

def extract_features(audio_path):
    wav, sr = librosa.load(audio_path, sr=16000, mono=True)
    waveform = wav.astype(np.float32)
    _, embeddings, _ = YAMNET_MODEL(waveform)
    return np.mean(embeddings.numpy(), axis=0)
