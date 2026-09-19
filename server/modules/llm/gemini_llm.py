import os
import asyncio
from typing import AsyncGenerator, List, Dict
from google import genai


class GeminiLLM:
    """
    Streaming LLM using Google Gemini 2.0 Flash via the new google-genai SDK.
    TTFT is very low, ideal for real-time voice conversational agents.
    """

    def __init__(self, api_key: str = None, model: str = "gemini-3.6-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model = model
        self.client = genai.Client(api_key=self.api_key) if self.api_key and self.api_key != "your_gemini_api_key_here" else None

    async def stream_response(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str = "You are a concise, friendly voice AI assistant. Always respond in 1-2 short sentences."
    ) -> AsyncGenerator[str, None]:
        if not self.client:
            mock_reply = "Xin chào! Đây là phản hồi giả lập từ Gemini. Hãy điền GEMINI_API_KEY vào file .env để kích hoạt AI trực tiếp."
            for word in mock_reply.split(" "):
                await asyncio.sleep(0.05)
                yield word + " "
            return

        # Combine conversation history
        prompt_parts = [f"System: {system_prompt}\n"]
        for msg in messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            prompt_parts.append(f"{role}: {msg['content']}\n")
        prompt_parts.append("Assistant: ")
        full_prompt = "".join(prompt_parts)

        # Run stream in executor since genai client streaming has sync chunks iterator
        loop = asyncio.get_running_loop()
        response_stream = await loop.run_in_executor(
            None,
            lambda: self.client.models.generate_content_stream(
                model=self.model,
                contents=full_prompt
            )
        )

        for chunk in response_stream:
            if chunk.text:
                yield chunk.text
