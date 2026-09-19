from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True

    # LLM Settings
    LLM_PROVIDER: str = "groq"  # 'groq' or 'gemini'
    GROQ_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "qwen/qwen3.8-27b"
    GEMINI_MODEL: str = "gemini-3.6-flash"

    # STT Settings
    STT_PROVIDER: str = "faster-whisper"
    WHISPER_MODEL_SIZE: str = "base"  # 'tiny', 'base', 'small'
    WHISPER_DEVICE: str = "cpu"
    WHISPER_COMPUTE_TYPE: str = "int8"

    # TTS Settings
    TTS_PROVIDER: str = "edge-tts"
    TTS_VOICE_EN: str = "en-US-EmmaNeural"
    TTS_VOICE_VI: str = "vi-VN-HoaiMyNeural"

    # VAD Settings
    VAD_CONFIDENCE_THRESHOLD: float = 0.35
    SILENCE_DURATION_MS: int = 700  # Silence window to trigger end of utterance

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
