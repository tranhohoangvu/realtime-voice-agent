import os
import urllib.request
import numpy as np
import onnxruntime as ort
from typing import Tuple, Optional

SILERO_VAD_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "assets")
MODEL_PATH = os.path.join(MODEL_DIR, "silero_vad.onnx")


class SileroVAD:
    """
    Silero VAD v5 wrapper using ONNX Runtime.
    Processes 16kHz audio chunks (recommended 512 samples = 32ms) with near-zero latency (<5ms).
    """

    def __init__(self, threshold: float = 0.5, sample_rate: int = 16000):
        self.threshold = threshold
        self.sample_rate = sample_rate
        self._ensure_model_exists()

        # Session options for fast CPU inference
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=["CPUExecutionProvider"])
        self.reset_states()

    def _ensure_model_exists(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
            print(f"[SileroVAD] Downloading Silero VAD v5 ONNX model (~2MB) to {MODEL_PATH}...")
            urllib.request.urlretrieve(SILERO_VAD_URL, MODEL_PATH)
            print("[SileroVAD] Download complete.")

    def reset_states(self):
        """Reset internal recurrent states for a new audio stream."""
        # For Silero VAD v5, state shape is (2, 1, 128)
        self._state = np.zeros((2, 1, 128), dtype=np.float32)

    def process_chunk(self, audio_chunk: np.ndarray) -> Tuple[float, bool]:
        """
        Process a chunk of audio.
        :param audio_chunk: 1D float32 numpy array normalized to [-1.0, 1.0], 512 samples at 16kHz
        :return: (speech_probability, is_speech)
        """
        if audio_chunk.ndim == 1:
            audio_chunk = np.expand_dims(audio_chunk, axis=0)  # Shape: (1, samples)

        inputs = {
            "input": audio_chunk.astype(np.float32),
            "state": self._state,
            "sr": np.array(self.sample_rate, dtype=np.int64)
        }

        out, new_state = self.session.run(None, inputs)
        self._state = new_state
        prob = float(out[0][0])
        is_speech = prob >= self.threshold

        return prob, is_speech

    @staticmethod
    def pcm16_to_float32(pcm_bytes: bytes) -> np.ndarray:
        """Convert raw 16-bit PCM bytes to float32 numpy array normalized to [-1.0, 1.0]."""
        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        return audio_int16.astype(np.float32) / 32768.0
