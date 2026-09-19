# 🎙️ Project Plan: Ultra-Low Latency Realtime Voice AI Agent (Full-Duplex)

## 1. Tổng Quan Dự Án (Project Overview)
Xây dựng một hệ thống Voice AI tương tác giọng nói hai chiều thời gian thực (**Full-Duplex Speech-to-Speech**) với mục tiêu độ trễ cực thấp (**TTFAB < 500ms** - Time to First Audio Byte). Hệ thống hỗ trợ tính năng **Barge-in (ngắt lời tự nhiên)**: khi bot đang nói, nếu người dùng lên tiếng, bot sẽ lập tức dừng phát âm thanh, huỷ luồng sinh câu trả lời cũ và lắng nghe người dùng.

### Các Mục Tiêu Kỹ Thuật Chính (Key KPIs)
- **TTFAB (Time-to-First-Audio-Byte):** $\le 500\text{ ms}$ (tính từ khi user dừng nói đến khi nghe thấy âm thanh đầu tiên từ bot).
- **Interruption Latency:** $\le 150\text{ ms}$ (thời gian từ lúc user bắt đầu ngắt lời đến khi client ngừng hoàn toàn âm thanh cũ).
- **Concurrency & Resource Efficiency:** Kiến trúc Asyncio Non-blocking, quản lý luồng bằng hàng đợi và state machine không có race-conditions.
- **Audio Quality:** Âm thanh tự nhiên, không giật cục (không bị pops/clicks khi ngắt câu).

---

## 2. Kiến Trúc Hệ Thống (System Architecture)

```
[User Browser / Client Mic]
         │ (Opus / PCM 16kHz via WebRTC or WebSocket)
         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Voice Agent Engine                       │
│                                                             │
│  ┌─────────────────────────┐      ┌──────────────────────┐  │
│  │ Silero VAD (Client/Edge)│─────►│ Interrupt Controller │  │
│  │ (Phát hiện tiếng nói)   │      │ (Hủy task LLM & TTS) │  │
│  └──────────┬──────────────┘      └──────────▲───────────┘  │
│             │ (Voice Active Chunks)          │ (Barge-in)   │
│             ▼                                │              │
│  ┌─────────────────────────┐                 │              │
│  │ Streaming STT           │                 │              │
│  │ (Faster-Whisper / Cloud)│                 │              │
│  └──────────┬──────────────┘                 │              │
│             │ (Finalized Text / Utterance)   │              │
│             ▼                                │              │
│  ┌─────────────────────────┐                 │              │
│  │ Streaming LLM           │─────────────────┤              │
│  │ (Groq / vLLM / Gemini)  │ (Cancel Signal) │              │
│  └──────────┬──────────────┘                 │              │
│             │ (Token Streams -> Clause Chunker)             │
│             ▼                                │              │
│  ┌─────────────────────────┐                 │              │
│  │ Streaming TTS           │─────────────────┘              │
│  │ (Kokoro-82M / Cartesia) │                                │
│  └──────────┬──────────────┘                                │
│             │ (Audio Chunks 24kHz)                          │
└─────────────┼───────────────────────────────────────────────┘
              ▼
[User Browser / Audio Output]
```

---

## 3. Lựa Chọn Công Nghệ (Technology Stack)

| Thành phần | Lựa chọn đề xuất | Lý do kỹ thuật |
| :--- | :--- | :--- |
| **Giao thức truyền tải** | **WebSocket** (Phase 1) ➔ **WebRTC / LiveKit** (Phase 2) | WebSocket triển khai nhanh, dễ kiểm soát buffer byte; WebRTC cho độ trễ truyền mạng jitter tối ưu nhất. |
| **VAD (Voice Activity)** | **Silero VAD v5** (ONNX Runtime) | Siêu nhẹ (~2MB), chạy cực nhanh (<5ms/chunk 30ms), độ chính xác nhận diện tiếng người rất cao. |
| **STT (Speech-to-Text)** | **Faster-Whisper** (Local) / **Deepgram Nova-2** (Cloud fallback) | Faster-Whisper dùng CTranslate2 tối ưu VRAM/CPU; Deepgram Nova-2 có latency streaming ~150-200ms. |
| **LLM Inference** | **Groq Llama-3.3-70B** hoặc **Local vLLM (Qwen2.5-3B)** | Time-to-First-Token (TTFT) đạt 100-150ms. |
| **TTS (Text-to-Speech)** | **Kokoro-82M** (Local ONNX) hoặc **Cartesia Sonic** / **ElevenLabs Flash** | Kokoro-82M là open-weight TTS nhẹ nhất hiện nay đạt chất lượng giọng tự nhiên với RTF < 0.1; Cartesia có latency ~100ms. |
| **Backend Framework** | **FastAPI + Asyncio Tasks + Pydantic** | Hỗ trợ Native Async WebSocket, quản lý tác vụ nền và State machine mượt mà. |
| **Client UI** | **Next.js / React + Web Audio API** | Trực quan hóa âm thanh (Audio Visualizer), quản lý AudioWorklet để capture và playback audio không bị block UI. |

---

## 4. Cấu Trúc Thư Mục Dự Án (Project Structure)

```
realtime-voice-agent/
├── PLAN.md                     # Tài liệu kế hoạch & thiết kế chi tiết (file này)
├── README.md                   # Hướng dẫn cài đặt, chạy demo và benchmark
├── docker-compose.yml          # Setup container (Backend + Redis/LiveKit nếu cần)
├── requirements.txt            # Python dependencies
├── .env.example                # Cấu hình API keys, ports, model options
│
├── server/                     # Backend Voice Engine (FastAPI)
│   ├── main.py                 # Entry point, WebSocket endpoints
│   ├── config.py               # Settings & Environment variables
│   │
│   ├── core/                   # Các module lõi xử lý thời gian thực
│   │   ├── pipeline.py         # VoicePipeline: kết nối VAD -> STT -> LLM -> TTS
│   │   ├── state_machine.py    # State Manager (IDLE, LISTENING, THINKING, SPEAKING, INTERRUPTED)
│   │   ├── chunker.py          # Sentence/Clause chunker (chia nhỏ token LLM thành cụm từ cho TTS)
│   │   └── audio_processor.py  # Resampling, PCM/WAV conversion, volume normalizer
│   │
│   ├── modules/                # Wrapper cho các AI models
│   │   ├── vad/                # Silero VAD implementation
│   │   │   └── silero_vad.py
│   │   ├── stt/                # Speech-to-Text implementations
│   │   │   ├── base.py
│   │   │   ├── faster_whisper_stt.py
│   │   │   └── deepgram_stt.py
│   │   ├── llm/                # Streaming LLM clients
│   │   │   ├── base.py
│   │   │   ├── groq_llm.py
│   │   │   └── vllm_client.py
│   │   └── tts/                # Streaming Text-to-Speech
│   │       ├── base.py
│   │       ├── kokoro_tts.py
│   │       └── cartesia_tts.py
│   │
│   └── telemetry/              # Metrics & Latency Profiler
│       ├── latency_tracker.py  # Đo đạc chi tiết từng chặng (VAD->STT->LLM->TTS)
│       └── logger.py
│
├── client/                     # Web Frontend (React / Vite hoặc Vanilla JS)
│   ├── index.html              # Giao diện demo với Audio Visualizer
│   ├── src/
│   │   ├── audio-worklet.js    # AudioWorkletProcessor thu âm PCM 16kHz mượt mà
│   │   ├── audio-player.js     # Queue buffer playback với tính năng instant-stop khi bị ngắt lời
│   │   ├── websocket-client.js # Quản lý kết nối streaming 2 chiều
│   │   └── app.js              # UI controller
│
└── benchmarks/                 # Bộ script đo đạc & đánh giá kỹ thuật
    ├── benchmark_latency.py    # Đo TTFAB tự động bằng audio test signals
    ├── test_barge_in.py        # Giả lập kịch bản user nói chen ngang và đo thời gian dừng TTS
    └── report_generator.py     # Tạo biểu đồ phân rã độ trễ (Latency Breakdown Chart)
```

---

## 5. Lộ Trình Triển Khai (Phased Roadmap)

### 🔹 Giai đoạn 1: Minimal Streaming Pipeline (Tuần 1)
- [x] Thiết lập môi trường Python, cài đặt `faster-whisper`, `groq`, `silero-vad` và `fastapi`.
- [x] Xây dựng WebSocket server nhận audio stream thô (PCM 16-bit 16kHz) từ client.
- [x] Tích hợp Silero VAD + Hybrid Energy VAD để phát hiện khi nào người dùng bắt đầu nói và kết thúc câu nói.
- [x] Kết nối tuần tự: Audio End -> STT -> LLM Streaming -> TTS audio chunk trả về client.
- [x] Đo đạc baseline latency lần đầu (LatencyTracker).

### 🔹 Giai đoạn 2: Sentence Chunker & TTS Streaming (Tuần 2)
- [x] Viết `Sentence/Clause Chunker`: Thay vì chờ LLM sinh xong cả câu dài mới gửi qua TTS, gom token theo cụm dấu câu (`,`, `.`, `?`, `!`) để đẩy ngay sang TTS.
- [x] Tích hợp Edge-TTS streaming audio bytes ngay lập tức (không cần API key, hỗ trợ giọng EN & VI).
- [x] Client xây dựng `AudioPlayer` với hàng đợi (queue) để phát âm thanh liền mạch, không giật tiếng giữa các chunks.

### 🔹 Giai đoạn 3: State Machine & Barge-in (Ngắt lời tức thì) (Tuần 3)
- [x] Xây dựng `ConversationStateMachine` với các trạng thái: `IDLE`, `LISTENING`, `THINKING`, `SPEAKING`, `INTERRUPTED`.
- [x] Khi bot đang ở trạng thái `SPEAKING`: Nếu VAD kích hoạt tín hiệu tiếng nói người dùng:
  1. Gửi ngay cờ `INTERRUPT` về Client để dừng ngay lập tức Audio Context đang phát.
  2. Cancel `asyncio.Task` của LLM generator và TTS synthesis đang chạy nền.
  3. Reset audio buffer, tăng generation_id để loại bỏ chunk trễ và chuyển trạng thái về `LISTENING`.

### 🔹 Giai đoạn 4: Web UI, Telemetry & Benchmark Evals (Tuần 4)
- [x] Xây dựng Web Audio Visualizer (Voice Orb hiển thị trạng thái và nhấp nháy theo âm lượng micro).
- [x] Tích hợp bộ Downsampler 16kHz trên client chuyển đổi mọi tần số phần cứng (44.1k/48k) chuẩn hóa sang 16kHz.
- [x] Viết module `LatencyTracker`: Lưu log và tính toán chi tiết:
  - STT Processing Time (ms)
  - LLM Time-to-First-Token (ms)
  - TTS Time-to-First-Audio (ms)
  - Total TTFAB (ms)
- [ ] Viết script giả lập tự động phát audio test file và đo đạc biểu đồ phân rã độ trễ (`benchmarks/`).
- [ ] Đóng gói Docker Compose.

---

## 6. Các Thách Thức Kỹ Thuật & Giải Pháp (Pitfalls & Mitigations)

1. **Acoustic Echo (Tiếng vang từ loa của bot lọt vào mic của user):**
   - *Giải pháp:* Kích hoạt browser-level `echoCancellation: true` trong `navigator.mediaDevices.getUserMedia`. Ở phía server, dùng timestamp hoặc cơ chế mute VAD tạm thời trong 100ms đầu bot nói.
2. **LLM Chunking quá ngắn hoặc quá dài:**
   - Nếu chunk chỉ 1-2 từ: TTS đọc không có ngữ điệu tự nhiên.
   - Nếu chunk cả câu 20 từ: Làm tăng TTFAB lên hơn 1 giây.
   - *Giải pháp:* Tách câu theo dynamic regex (dấu phẩy, liên từ "and", "nhưng", hoặc độ dài tối thiểu 5-8 từ).
3. **Race Condition khi hủy tác vụ:**
   - Khi ngắt lời, task cũ chưa kịp cancel có thể vẫn đẩy audio chunk vào queue.
   - *Giải pháp:* Dùng `generation_id` (UUID hoặc int tăng dần). Mọi audio chunk sinh ra đều mang ID này, nếu ID cũ hơn ID hiện tại thì lập tức drop.
