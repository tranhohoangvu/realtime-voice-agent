import time
from typing import Dict, Any, Optional


class LatencyTracker:
    """
    Profiles end-to-end and component latencies:
    - VAD speech end detection
    - STT transcription duration
    - LLM Time-to-First-Token (TTFT)
    - TTS Time-to-First-Audio (TTFA)
    - Total Time-to-First-Audio-Byte (TTFAB)
    """

    def __init__(self):
        self.speech_ended_at: Optional[float] = None
        self.stt_started_at: Optional[float] = None
        self.stt_ended_at: Optional[float] = None
        self.llm_started_at: Optional[float] = None
        self.llm_first_token_at: Optional[float] = None
        self.tts_started_at: Optional[float] = None
        self.tts_first_byte_at: Optional[float] = None

    def mark_speech_ended(self):
        self.speech_ended_at = time.perf_counter()

    def mark_stt_started(self):
        self.stt_started_at = time.perf_counter()

    def mark_stt_ended(self):
        self.stt_ended_at = time.perf_counter()

    def mark_llm_started(self):
        self.llm_started_at = time.perf_counter()

    def mark_llm_first_token(self):
        if not self.llm_first_token_at:
            self.llm_first_token_at = time.perf_counter()

    def mark_tts_started(self):
        self.tts_started_at = time.perf_counter()

    def mark_tts_first_byte(self):
        if not self.tts_first_byte_at:
            self.tts_first_byte_at = time.perf_counter()

    def get_metrics(self) -> Dict[str, Any]:
        """Calculates latency in milliseconds."""
        metrics = {}

        if self.stt_started_at and self.stt_ended_at:
            metrics["stt_duration_ms"] = round((self.stt_ended_at - self.stt_started_at) * 1000, 1)

        if self.llm_started_at and self.llm_first_token_at:
            metrics["llm_ttft_ms"] = round((self.llm_first_token_at - self.llm_started_at) * 1000, 1)

        if self.tts_started_at and self.tts_first_byte_at:
            metrics["tts_ttfa_ms"] = round((self.tts_first_byte_at - self.tts_started_at) * 1000, 1)

        if self.speech_ended_at and self.tts_first_byte_at:
            metrics["total_ttfab_ms"] = round((self.tts_first_byte_at - self.speech_ended_at) * 1000, 1)

        return metrics
