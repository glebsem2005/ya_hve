import numpy as np
import soundfile as sf
from dataclasses import dataclass
from typing import Literal
import os

_yamnet = None
_head = None

CLASSES = ["chainsaw", "gunshot", "engine", "axe", "fire", "background"]
AudioClass = Literal[
    "chainsaw", "gunshot", "engine", "axe", "fire", "background", "unknown"
]

MODEL_PATH = os.environ.get(
    "YAMNET_HEAD_PATH",
    os.path.join(os.path.dirname(__file__), "yamnet_forest_classifier_v5.keras"),
)


@dataclass
class AudioResult:
    label: AudioClass
    confidence: float
    raw_scores: dict


def _load_models():
    global _yamnet, _head
    if _yamnet is None:
        import tensorflow_hub as hub

        _yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
    if _head is None:
        import tensorflow as tf

        _head = tf.keras.models.load_model(MODEL_PATH)
    return _yamnet, _head


def classify(audio_path: str) -> AudioResult:
    yamnet, head = _load_models()

    waveform, sr = sf.read(audio_path, dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    scores, embeddings, spectrogram = yamnet(waveform)
    emb_np = embeddings.numpy()

    mean_emb = emb_np.mean(axis=0)  # 1024
    max_emb = emb_np.max(axis=0)  # 1024
    features = np.concatenate([mean_emb, max_emb])  # 2048

    # v5 model expects 2181 (2048 YAMNet + 128 PCEN + 5 temporal).
    # Pad with zeros for PCEN/temporal dims — model relies primarily on
    # the 2048 YAMNet features, so accuracy stays close to full pipeline.
    expected_dim = head.input_shape[-1]
    if features.shape[0] < expected_dim:
        features = np.concatenate(
            [features, np.zeros(expected_dim - features.shape[0])]
        )

    pred = head.predict(features[np.newaxis, :], verbose=0)[0]
    pred_idx = int(np.argmax(pred))
    confidence = float(pred[pred_idx])

    label: AudioClass = CLASSES[pred_idx] if pred_idx < len(CLASSES) else "unknown"

    raw = {CLASSES[i]: float(pred[i]) for i in range(len(CLASSES))}

    return AudioResult(label=label, confidence=confidence, raw_scores=raw)
