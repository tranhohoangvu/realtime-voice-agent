import re
from typing import AsyncGenerator


class ClauseChunker:
    """
    Sentence & Clause Chunker for Voice AI:
    Splits LLM token streams into natural speech chunks on punctuation boundaries.
    Allows TTS to begin synthesizing after the first few words instead of waiting for full LLM response.
    """

    # Primary delimiters (end of sentence) and secondary delimiters (natural pauses)
    DELIMITERS = re.compile(r'([.!?;:\n,])')

    def __init__(self, min_chunk_words: int = 3, max_chunk_words: int = 15):
        self.min_chunk_words = min_chunk_words
        self.max_chunk_words = max_chunk_words
        self._buffer = ""

    async def process_token_stream(self, token_stream: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
        """
        Takes an async token generator from LLM and yields speech-ready clauses.
        """
        self._buffer = ""

        async for token in token_stream:
            self._buffer += token

            # Split buffer by delimiters while preserving delimiters
            parts = self.DELIMITERS.split(self._buffer)

            # If we have at least one delimiter matched (parts length >= 3)
            if len(parts) >= 3:
                # Reassemble the completed clause (part + delimiter)
                clause = parts[0] + parts[1]
                word_count = len(clause.split())

                if word_count >= self.min_chunk_words or parts[1] in ".!?\n":
                    yield clause.strip()
                    # Remaining buffer
                    self._buffer = "".join(parts[2:])

        # Yield any remaining text in buffer at end of stream
        remainder = self._buffer.strip()
        if remainder:
            yield remainder
