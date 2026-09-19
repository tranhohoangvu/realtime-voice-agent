import json
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from server.config import settings
from server.core.pipeline import VoicePipeline
from server.modules.stt.faster_whisper_stt import FasterWhisperSTT

shared_stt_model: FasterWhisperSTT = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global shared_stt_model
    print("[Server] Pre-loading Faster-Whisper model into memory...")
    shared_stt_model = FasterWhisperSTT(
        model_size=settings.WHISPER_MODEL_SIZE,
        device=settings.WHISPER_DEVICE,
        compute_type=settings.WHISPER_COMPUTE_TYPE
    )
    print("[Server] Faster-Whisper ready. Server started.")
    yield
    print("[Server] Shutting down...")


app = FastAPI(title="Realtime Voice Agent", lifespan=lifespan)

# Client static files path
CLIENT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "client")


@app.get("/")
async def get_index():
    return FileResponse(os.path.join(CLIENT_DIR, "index.html"))


@app.websocket("/ws/voice")
async def websocket_voice_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[WebSocket] Client connected.")

    async def send_text_event(event_dict: dict):
        try:
            await websocket.send_text(json.dumps(event_dict))
        except Exception:
            pass

    async def send_audio_chunk(chunk_bytes: bytes, gen_id: int):
        try:
            # Header: 4 bytes big-endian for generation_id, followed by raw MP3 audio bytes
            header = gen_id.to_bytes(4, byteorder="big")
            await websocket.send_bytes(header + chunk_bytes)
        except Exception:
            pass

    pipeline = VoicePipeline(
        send_text_event=send_text_event,
        send_audio_chunk=send_audio_chunk,
        stt_model=shared_stt_model
    )

    chunk_count = 0
    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message and message["bytes"]:
                chunk_count += 1
                if chunk_count == 1 or chunk_count % 100 == 0:
                    print(f"[WebSocket] Received audio packet #{chunk_count} ({len(message['bytes'])} bytes)")
                await pipeline.process_audio_chunk(message["bytes"])
            elif "text" in message and message["text"]:
                data = json.loads(message["text"])
                if data.get("type") == "interrupt":
                    await pipeline.interrupt()
                elif data.get("type") == "reset":
                    pipeline.conversation_history.clear()
                    await pipeline.state_machine.transition_to(pipeline.state_machine.current_state.IDLE)
    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected.")
    except Exception as e:
        print(f"[WebSocket Error]: {e}")
    finally:
        if pipeline.active_generation_task and not pipeline.active_generation_task.done():
            pipeline.active_generation_task.cancel()


# Mount static assets if client/src exists
app.mount("/src", StaticFiles(directory=os.path.join(CLIENT_DIR, "src")), name="src")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.main:app", host=settings.HOST, port=settings.PORT, reload=False)
