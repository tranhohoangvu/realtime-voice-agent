<div align="center">

# 🎙️ Ultra-Low Latency Realtime Voice AI Agent (Full-Duplex)

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Cost](https://img.shields.io/badge/Cost-100%25%20Free%20%2F%20Local-success)](https://github.com/tranhohoangvu/realtime-voice-agent)

An end-to-end, full-duplex conversational voice agent engineered for ultra-low latency (**TTFAB < 500ms**) with instantaneous **barge-in** interruption support. Built on a **100% free / open-source local-first stack** requiring zero paid APIs.

[Architecture](#-system-architecture) • [Key Features](#-key-features) • [Tech Stack](#-tech-stack) • [Quickstart](#-quickstart) • [Roadmap](#-roadmap)

</div>

---

## ⚡ System Architecture

```
[User Browser / Audio Input]
         │ (Native 44.1k/48k Mic Stream)
         ▼
 ┌────────────────────────────────────────┐
 │ AudioWorklet / Web Audio Resampler     │ ➔ Downsampled to 16kHz PCM
 └──────────────────┬─────────────────────┘
                    │ (WebSocket Binary Chunks, 512 samples / 32ms)
                    ▼
 ┌─────────────────────────────────────────────────────────────┐
 │                Realtime Voice Agent Engine                  │
 │                                                             │
 │  ┌─────────────────────────┐      ┌──────────────────────┐  │
 │  │ Silero VAD + Energy RMS │─────►│ Interrupt Controller │  │
 │  │ (Hybrid Turn Detector)  │      │ (Instant Task Cancel)│  │
 │  └──────────┬──────────────┘      └──────────▲───────────┘  │
 │             │ (Voice Active Chunks)          │ (Barge-in)   │
 │             ▼                                │              │
 │  ┌─────────────────────────┐                 │              │
 │  │ Faster-Whisper (int8)   │                 │              │
 │  │ (Local CPU Transcription│                 │              │
 │  └──────────┬──────────────┘                 │              │
 │             │ (Transcribed Text Utterance)   │              │
 │             ▼                                │              │
 │  ┌─────────────────────────┐                 │              │
 │  │ Streaming LLM           │─────────────────┤              │
 │  │ (Groq Qwen / Gemini)    │ (Cancel Signal) │              │
 │  └──────────┬──────────────┘                 │              │
 │             │ (Token Streams)                │              │
 │             ▼                                │              │
 │  ┌─────────────────────────┐                 │              │
 │  │ Sentence/Clause Chunker │                 │              │
 │  │ (Grammar Boundary Split)│                 │              │
 │  └──────────┬──────────────┘                 │              │
 │             │ (Synthesizable Clauses)        │              │
 │             ▼                                │              │
 │  ┌─────────────────────────┐                 │              │
 │  │ Streaming Edge-TTS      │─────────────────┘              │
 │  │ (Microsoft Azure Voice) │                                │
 │  └──────────┬──────────────┘                                │
 │             │ (Tagged MP3 Audio Chunks with generation_id)  │
 └─────────────┼───────────────────────────────────────────────┘
               ▼
 [User Browser / Audio Playback Queue]
 (Plays seamlessly, instantly purges queue upon receiving INTERRUPT)
```

---

## 🌟 Key Features

- **Full-Duplex Bidirectional Audio:** Continuous micro-chunk streaming via WebSocket. Both user and AI can speak simultaneously.
- **Natural Barge-In Interruption (< 150ms):** When the user speaks while the bot is speaking or thinking, the pipeline immediately cancels the running `asyncio.Task`, sends an interrupt event, purges the browser playback queue, and pivots to `LISTENING`.
- **Streaming Clause Chunker:** Instead of waiting for full LLM completion (which adds 2-3s delay), tokens are segmented on punctuation boundaries (`,`, `.`, `?`, `!`, `\n`) and piped to TTS clause-by-clause.
- **Zero-Cost Production Stack:**
  - **VAD:** Silero VAD v5 (ONNX Runtime CPU inference < 5ms).
  - **STT:** Faster-Whisper `base` model (CTranslate2 int8 on CPU).
  - **LLM:** Groq (`qwen/qwen3.8-27b`, ~100-150ms TTFT) or Google Gemini Flash (`gemini-3.6-flash`).
  - **TTS:** Edge-TTS (free cloud neural voices for English and Vietnamese).
- **Client-Side Hardware Resampling:** Automatically converts diverse browser hardware sample rates (44.1kHz / 48kHz / 96kHz) to strict 16kHz mono PCM frames before transmission.
- **Granular Latency Telemetry:** Real-time breakdown of STT processing time, LLM Time-to-First-Token (TTFT), TTS Time-to-First-Audio (TTFA), and Total Time-to-First-Audio-Byte (TTFAB).

---

## 🛠️ Tech Stack

| Component | Technology | Purpose / Highlights |
| :--- | :--- | :--- |
| **Backend** | FastAPI + Asyncio WebSockets | High-concurrency event-driven pipeline |
| **VAD** | Silero VAD v5 (ONNX) + Energy RMS Gating | Ultra-fast speech activity detection (<5ms) |
| **STT** | Faster-Whisper (CTranslate2) | 100% local CPU speech recognition with int8 quantization |
| **LLM** | Groq Cloud / Google Gemini API | High-throughput streaming inference |
| **TTS** | Edge-TTS | High-fidelity neural voices with zero API key requirement |
| **Frontend** | Vanilla JS + Web Audio API | Low-overhead streaming client & visualizer |

---

## 🚀 Quickstart

### 1. Prerequisites
- **Python 3.10+** (Tested on Python 3.12)
- Microphone and speakers/headphones

### 2. Installation

Clone the repository:
```bash
git clone https://github.com/tranhohoangvu/realtime-voice-agent.git
cd realtime-voice-agent
```

Create and activate virtual environment:
```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

Copy `.env.example` to `.env`:
```powershell
cp .env.example .env
```

Add your free API keys in `.env`:
```ini
# LLM Provider ('groq' or 'gemini')
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_free_groq_key_here
GEMINI_API_KEY=your_gemini_key_here

# STT & TTS Options
WHISPER_MODEL_SIZE=base
WHISPER_DEVICE=cpu
TTS_PROVIDER=edge-tts
TTS_VOICE_EN=en-US-EmmaNeural
TTS_VOICE_VI=vi-VN-HoaiMyNeural
```

> **Note:** Get free keys without credit cards at [Groq Console](https://console.groq.com) or [Google AI Studio](https://aistudio.google.com).

### 4. Run the Server

```powershell
python -m uvicorn server.main:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser at **[http://localhost:8000](http://localhost:8000)** and click **"Start Talking"**.

---

## 📂 Project Structure

```
realtime-voice-agent/
├── server/
│   ├── main.py                     # FastAPI WebSocket entry point
│   ├── config.py                   # Pydantic Settings
│   ├── core/
│   │   ├── pipeline.py             # VoicePipeline orchestrator (VAD -> STT -> LLM -> TTS)
│   │   ├── state_machine.py        # Conversational state manager & Barge-in
│   │   └── chunker.py              # Clause/Sentence streaming chunker
│   ├── modules/
│   │   ├── vad/silero_vad.py       # Silero VAD v5 ONNX runtime wrapper
│   │   ├── stt/faster_whisper_stt.py # Faster-Whisper local STT wrapper
│   │   ├── llm/
│   │   │   ├── groq_llm.py         # Groq streaming client
│   │   │   └── gemini_llm.py       # Gemini Flash streaming client
│   │   └── tts/edge_tts_client.py  # Edge-TTS streaming client
│   └── telemetry/
│       └── latency_tracker.py      # Granular latency breakdown tracker
│
├── client/
│   ├── index.html                  # Interactive Web UI & Voice Orb
│   └── src/
│       └── app.js                  # Web Audio resampling & playback queue
│
├── PLAN.md                         # Detailed project roadmap & technical plan
├── requirements.txt                # Python dependencies
└── .env.example                    # Environment variable template
```

---

## 📊 Roadmap

- [x] **Milestone 1:** Minimal Working Pipeline (VAD + Faster-Whisper + Groq + Edge-TTS)
- [x] **Milestone 2:** Clause-based Token Chunking for streaming synthesis
- [x] **Milestone 3:** Full-duplex State Machine with instant Barge-in interruption
- [x] **Milestone 4:** Client-side 16kHz resampler and Web Audio Visualizer
- [ ] **Milestone 5:** Automated latency benchmarking suite (`benchmarks/`)
- [ ] **Milestone 6:** WebRTC / LiveKit transport layer migration

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.
