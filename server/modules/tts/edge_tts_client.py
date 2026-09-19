import edge_tts
from typing import AsyncGenerator


class EdgeTTSClient:
    """
    100% Free Text-to-Speech using edge-tts (Microsoft Azure Neural voices).
    Zero API key required. High quality and low-latency audio streaming.
    """

    def __init__(self, voice: str = "vi-VN-HoaiMyNeural"):
        self.voice = voice

    async def stream_audio_chunks(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Synthesize text and yield audio bytes chunks as they arrive.
        """
        if not text.strip():
            return

        communicate = edge_tts.Communicate(text=text, voice=self.voice)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]

    async def synthesize_all(self, text: str) -> bytes:
        """Synthesize text and return full audio byte buffer."""
        communicate = edge_tts.Communicate(text=text, voice=self.voice)
        audio_data = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.extend(chunk["data"])
        return bytes(audio_data)
