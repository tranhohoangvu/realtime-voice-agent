import os
import asyncio
from typing import AsyncGenerator, List, Dict
from groq import AsyncGroq


class GroqLLM:
    """
    Ultra-low latency LLM inference using Groq (Llama-3.3-70B).
    TTFT (Time-to-First-Token) ~ 100-150ms.
    """

    def __init__(self, api_key: str = None, model: str = "qwen/qwen3.8-27b"):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.model = model
        self.client = AsyncGroq(api_key=self.api_key) if self.api_key else None

    async def stream_response(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str = "You are a concise, helpful Voice AI assistant. Keep responses under 2 sentences."
    ) -> AsyncGenerator[str, None]:
        """
        Stream tokens from Groq. If API key is not configured, provides a mock response for testing.
        """
        if not self.client or self.api_key == "your_groq_api_key_here":
            mock_reply = "Xin chào bạn! Tôi là Voice AI Agent hoạt động thời gian thực. Hệ thống xử lý âm thanh đã kết nối thành công."
            for word in mock_reply.split(" "):
                await asyncio.sleep(0.05)
                yield word + " "
            return

        chat_messages = [{"role": "system", "content": system_prompt}] + messages
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=chat_messages,
            stream=True,
            temperature=0.6,
            max_tokens=150
        )

        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content
