# app/utils/yamnet_utils.py
import tempfile
import os
import numpy as np
import librosa
import tensorflow_hub as hub

# Load YAMNet once globally
YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"
yamnet = hub.load(YAMNET_HANDLE)

def wavfile_to_mean_embedding(file_like_or_path):
    """
    Accepts either a path string or a file-like object (Flask upload).
    Returns a 1024-d numpy embedding (mean over frames).
    """
    created_tmp = False
    if hasattr(file_like_or_path, "read"):
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(file_like_or_path.read())
        tmp.flush()
        tmp.close()
        path = tmp.name
        created_tmp = True
    else:
        path = file_like_or_path

    wav, sr = librosa.load(path, sr=16000, mono=True)
    waveform = wav.astype(np.float32)
    scores, embeddings, spec = yamnet(waveform)
    emb = np.mean(embeddings.numpy(), axis=0)

    if created_tmp:
        try:
            os.unlink(path)
        except:
            pass

    return emb
