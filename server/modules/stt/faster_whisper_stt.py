import os
import io
import numpy as np
import soundfile as sf
from typing import Optional
from faster_whisper import WhisperModel


class FasterWhisperSTT:
    """
    Local Speech-to-Text using Faster-Whisper (CTranslate2).
    Runs 100% on local CPU with int8 quantization. Free and private.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8"
    ):
        print(f"[FasterWhisperSTT] Loading whisper model '{model_size}' on {device} ({compute_type})...")
        self.model = WhisperModel(
            model_size_or_path=model_size,
            device=device,
            compute_type=compute_type
        )
        print("[FasterWhisperSTT] Model loaded successfully.")

    def transcribe_audio_array(self, audio_array: np.ndarray, sample_rate: int = 16000, language: Optional[str] = None) -> str:
        """
        Transcribe 1D float32 audio array.
        """
        if len(audio_array) == 0:
            return ""

        # faster-whisper accepts 1D float32 numpy array directly
        segments, _ = self.model.transcribe(
            audio_array,
            beam_size=1,
            language=language,
            condition_on_previous_text=False
        )
        text = " ".join([seg.text.strip() for seg in segments if seg.text])
        return text.strip()

    def transcribe_pcm_bytes(self, pcm_bytes: bytes, sample_rate: int = 16000, language: Optional[str] = None) -> str:
        """
        Transcribe raw 16-bit 16kHz PCM bytes.
        """
        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        return self.transcribe_audio_array(audio_float32, sample_rate=sample_rate, language=language)
