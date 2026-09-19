import asyncio
import time
import numpy as np
from typing import Optional, Callable, Awaitable, List, Dict
from server.config import settings
from server.modules.vad.silero_vad import SileroVAD
from server.modules.stt.faster_whisper_stt import FasterWhisperSTT
from server.modules.llm.groq_llm import GroqLLM
from server.modules.llm.gemini_llm import GeminiLLM
from server.modules.tts.edge_tts_client import EdgeTTSClient
from server.core.chunker import ClauseChunker
from server.core.state_machine import StateMachine, AgentState
from server.telemetry.latency_tracker import LatencyTracker


class VoicePipeline:
    """
    Core Full-Duplex Real-Time Voice Pipeline:
    - Audio Ingestion -> Silero VAD
    - Turn Detection & Barge-in Cancellation
    - Faster-Whisper STT
    - Groq / Gemini LLM Streaming
    - Clause Chunker -> Edge-TTS Streaming
    """

    def __init__(
        self,
        send_text_event: Callable[[Dict], Awaitable[None]],
        send_audio_chunk: Callable[[bytes, int], Awaitable[None]],
        stt_model: Optional[FasterWhisperSTT] = None
    ):
        self.send_text_event = send_text_event
        self.send_audio_chunk = send_audio_chunk

        # State and turn management
        self.state_machine = StateMachine(on_state_change=self._on_state_change)
        self.generation_id = 0

        # Components
        self.vad = SileroVAD(threshold=settings.VAD_CONFIDENCE_THRESHOLD)
        self.stt = stt_model if stt_model else FasterWhisperSTT(
            model_size=settings.WHISPER_MODEL_SIZE,
            device=settings.WHISPER_DEVICE,
            compute_type=settings.WHISPER_COMPUTE_TYPE
        )

        if settings.LLM_PROVIDER == "gemini":
            self.llm = GeminiLLM(api_key=settings.GEMINI_API_KEY, model=settings.GEMINI_MODEL)
        else:
            self.llm = GroqLLM(api_key=settings.GROQ_API_KEY, model=settings.GROQ_MODEL)

        self.tts = EdgeTTSClient(voice=settings.TTS_VOICE_VI)
        self.chunker = ClauseChunker()

        # Audio stream byte buffers
        self.audio_buffer = bytearray()
        self.vad_stream_buffer = bytearray()
        self.silence_chunks_count = 0
        self.speech_chunks_count = 0
        self.is_speech_active = False

        # 500ms of silence at 32ms/chunk ~= 16 chunks
        self.max_silence_chunks = int(settings.SILENCE_DURATION_MS / 32)

        # Active tasks (can be cancelled on barge-in)
        self.active_generation_task: Optional[asyncio.Task] = None
        self.conversation_history: List[Dict[str, str]] = []
        self.latency_tracker: Optional[LatencyTracker] = None

    async def _on_state_change(self, old_state: AgentState, new_state: AgentState):
        await self.send_text_event({
            "type": "state",
            "state": new_state.value,
            "old_state": old_state.value
        })

    async def process_audio_chunk(self, chunk_bytes: bytes):
        """
        Receives raw 16-bit PCM bytes (any packet size) and slices them into exact 1024-byte (512-sample) windows.
        """
        self.vad_stream_buffer.extend(chunk_bytes)
        CHUNK_SIZE = 1024  # 512 samples * 2 bytes/sample (16-bit)

        while len(self.vad_stream_buffer) >= CHUNK_SIZE:
            exact_chunk = bytes(self.vad_stream_buffer[:CHUNK_SIZE])
            del self.vad_stream_buffer[:CHUNK_SIZE]
            await self._process_single_vad_window(exact_chunk)

    async def _process_single_vad_window(self, chunk_bytes: bytes):
        float_chunk = SileroVAD.pcm16_to_float32(chunk_bytes)
        prob, _ = self.vad.process_chunk(float_chunk)

        # Audio energy RMS calculation
        energy = float(np.sqrt(np.mean(float_chunk ** 2)))

        # Hybrid Speech Detection: Speech if audio energy exceeds background noise (>0.015) or VAD prob is high
        is_speech = bool((energy >= 0.015) or (prob >= 0.2))

        self._diag_counter = getattr(self, "_diag_counter", 0) + 1
        if self._diag_counter % 25 == 0 or is_speech:
            print(f"[Audio In] Energy (RMS): {energy:.4f} | VAD prob: {prob:.3f} | Speech: {is_speech} | State: {self.state_machine.current_state.value}")

        # 1. Check for BARGE-IN: User speaks while bot is speaking or thinking
        if is_speech and (self.state_machine.current_state in [AgentState.SPEAKING, AgentState.THINKING]):
            self.speech_chunks_count += 1
            if self.speech_chunks_count >= 2:  # Confirmed speech (approx 64ms)
                print(f"[Pipeline] ⚡ BARGE-IN detected (prob={prob:.2f})! Interrupting current response.")
                await self.interrupt()

        # 2. Track speech start and continuation
        if is_speech:
            self.speech_chunks_count += 1
            self.silence_chunks_count = 0
            if not self.is_speech_active and self.speech_chunks_count >= 2:
                self.is_speech_active = True
                self.audio_buffer.clear()
                print(f"[Pipeline] 🗣️ User started speaking (prob={prob:.2f})")
                await self.state_machine.transition_to(AgentState.LISTENING)

            if self.is_speech_active:
                self.audio_buffer.extend(chunk_bytes)
        else:
            self.speech_chunks_count = 0
            if self.is_speech_active:
                self.audio_buffer.extend(chunk_bytes)
                self.silence_chunks_count += 1

                # 3. Speech End Detection (turn finished)
                if self.silence_chunks_count >= self.max_silence_chunks:
                    self.is_speech_active = False
                    self.silence_chunks_count = 0
                    print(f"[Pipeline] 🤫 User finished speaking ({len(self.audio_buffer)} audio bytes)")
                    await self._on_user_turn_complete()

    async def interrupt(self):
        """Handle Barge-in: immediately cancel audio generation and stop client playback."""
        self.generation_id += 1
        if self.active_generation_task and not self.active_generation_task.done():
            self.active_generation_task.cancel()
            self.active_generation_task = None

        await self.state_machine.transition_to(AgentState.INTERRUPTED)
        await self.send_text_event({
            "type": "interrupt",
            "generation_id": self.generation_id
        })
        self.audio_buffer.clear()
        self.is_speech_active = True
        await self.state_machine.transition_to(AgentState.LISTENING)

    async def _on_user_turn_complete(self):
        """User finished speaking. Kick off STT -> LLM -> TTS pipeline."""
        if len(self.audio_buffer) < 3200:  # Less than 100ms of audio, ignore
            self.audio_buffer.clear()
            await self.state_machine.transition_to(AgentState.IDLE)
            return

        user_pcm = bytes(self.audio_buffer)
        self.audio_buffer.clear()

        self.latency_tracker = LatencyTracker()
        self.latency_tracker.mark_speech_ended()

        self.generation_id += 1
        gen_id = self.generation_id

        # Launch response generator as cancellable background task
        self.active_generation_task = asyncio.create_task(
            self._generate_response(user_pcm, gen_id)
        )

    async def _generate_response(self, user_pcm: bytes, current_gen_id: int):
        try:
            await self.state_machine.transition_to(AgentState.THINKING)

            # STT
            self.latency_tracker.mark_stt_started()
            loop = asyncio.get_running_loop()
            transcription = await loop.run_in_executor(
                None,
                lambda: self.stt.transcribe_pcm_bytes(user_pcm, language=None)
            )
            self.latency_tracker.mark_stt_ended()

            if not transcription or not transcription.strip():
                print("[Pipeline] No speech transcribed. Returning to IDLE.")
                await self.state_machine.transition_to(AgentState.IDLE)
                return

            print(f"[User]: {transcription}")
            await self.send_text_event({
                "type": "transcript",
                "role": "user",
                "text": transcription
            })

            # Check if cancelled before LLM
            if current_gen_id != self.generation_id:
                return

            self.conversation_history.append({"role": "user", "content": transcription})

            # LLM Stream
            self.latency_tracker.mark_llm_started()
            token_gen = self.llm.stream_response(self.conversation_history[-6:])

            async def timed_tokens():
                async for tok in token_gen:
                    self.latency_tracker.mark_llm_first_token()
                    yield tok

            clause_stream = self.chunker.process_token_stream(timed_tokens())

            # TTS Streaming per clause
            full_bot_response = []
            self.latency_tracker.mark_tts_started()

            async for clause in clause_stream:
                if current_gen_id != self.generation_id:
                    return  # Discard if barge-in happened

                full_bot_response.append(clause)
                await self.send_text_event({
                    "type": "transcript",
                    "role": "assistant_clause",
                    "text": clause
                })

                await self.state_machine.transition_to(AgentState.SPEAKING)

                # Synthesize clause to audio chunks
                async for audio_chunk in self.tts.stream_audio_chunks(clause):
                    if current_gen_id != self.generation_id:
                        return
                    self.latency_tracker.mark_tts_first_byte()
                    await self.send_audio_chunk(audio_chunk, current_gen_id)

            # End of response
            if current_gen_id == self.generation_id:
                complete_text = " ".join(full_bot_response)
                self.conversation_history.append({"role": "assistant", "content": complete_text})
                metrics = self.latency_tracker.get_metrics()
                print(f"[Metrics]: {metrics}")
                await self.send_text_event({
                    "type": "metrics",
                    "metrics": metrics
                })
                await self.state_machine.transition_to(AgentState.IDLE)

        except asyncio.CancelledError:
            print(f"[Pipeline] Task for gen_id={current_gen_id} cancelled.")
        except Exception as e:
            print(f"[Pipeline Error]: {e}")
            await self.state_machine.transition_to(AgentState.IDLE)
