// Web Audio & WebSocket Client for Realtime Voice Agent
class VoiceClient {
    constructor() {
        this.ws = null;
        this.audioCtx = null;
        this.mediaStream = null;
        this.processorNode = null;
        this.isConnected = false;
        this.isRecording = false;

        // Audio playback queue
        this.playbackQueue = [];
        this.isPlaying = false;
        this.currentSource = null;
        this.currentGenerationId = 0;

        // Resampling & buffering for 16kHz VAD chunks (512 samples = 32ms)
        this.sampleBuffer16k = [];

        // UI Callbacks
        this.onStateChange = null;
        this.onTranscript = null;
        this.onMetrics = null;
        this.onVisualizer = null;
    }

    async connect() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            return;
        }

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/voice`;

        return new Promise((resolve, reject) => {
            this.ws = new WebSocket(wsUrl);
            this.ws.binaryType = 'arraybuffer';

            this.ws.onopen = () => {
                console.log('[WebSocket] Connected successfully to', wsUrl);
                this.isConnected = true;
                if (this.onStateChange) this.onStateChange('CONNECTED');
                resolve();
            };

            this.ws.onmessage = async (event) => {
                if (typeof event.data === 'string') {
                    const msg = JSON.parse(event.data);
                    this.handleJsonMessage(msg);
                } else if (event.data instanceof ArrayBuffer) {
                    this.handleAudioChunk(event.data);
                }
            };

            this.ws.onclose = () => {
                console.log('[WebSocket] Disconnected.');
                this.isConnected = false;
                this.stopRecording();
                if (this.onStateChange) this.onStateChange('DISCONNECTED');
            };

            this.ws.onerror = (err) => {
                console.error('[WebSocket Error]:', err);
                reject(err);
            };
        });
    }

    handleJsonMessage(msg) {
        if (msg.type === 'state') {
            if (this.onStateChange) this.onStateChange(msg.state);
        } else if (msg.type === 'transcript') {
            if (this.onTranscript) this.onTranscript(msg.role, msg.text);
        } else if (msg.type === 'metrics') {
            if (this.onMetrics) this.onMetrics(msg.metrics);
        } else if (msg.type === 'interrupt') {
            console.log('[Barge-in] Interrupt received! Generation ID:', msg.generation_id);
            this.currentGenerationId = msg.generation_id;
            this.stopCurrentAudio();
            if (this.onStateChange) this.onStateChange('INTERRUPTED');
        }
    }

    async handleAudioChunk(arrayBuffer) {
        const view = new DataView(arrayBuffer);
        const chunkGenId = view.getUint32(0, false); // 4 bytes big endian
        const audioBytes = arrayBuffer.slice(4);

        if (chunkGenId < this.currentGenerationId) {
            // Stale audio chunk from prior interrupted response
            return;
        }

        this.playbackQueue.push({ genId: chunkGenId, data: audioBytes });
        if (!this.isPlaying) {
            this.playNextChunk();
        }
    }

    async playNextChunk() {
        if (this.playbackQueue.length === 0) {
            this.isPlaying = false;
            return;
        }

        const item = this.playbackQueue.shift();
        if (item.genId < this.currentGenerationId) {
            return this.playNextChunk();
        }

        this.isPlaying = true;
        try {
            if (!this.audioCtx) {
                this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            }
            if (this.audioCtx.state === 'suspended') {
                await this.audioCtx.resume();
            }

            const audioBuffer = await this.audioCtx.decodeAudioData(item.data.slice(0));
            if (item.genId < this.currentGenerationId) {
                return this.playNextChunk();
            }

            const source = this.audioCtx.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(this.audioCtx.destination);
            this.currentSource = source;

            source.onended = () => {
                this.currentSource = null;
                this.playNextChunk();
            };

            source.start();
        } catch (e) {
            console.error('Audio playback error:', e);
            this.playNextChunk();
        }
    }

    stopCurrentAudio() {
        this.playbackQueue = [];
        if (this.currentSource) {
            try {
                this.currentSource.stop();
            } catch (e) {}
            this.currentSource = null;
        }
        this.isPlaying = false;
    }

    // Downsample input float32 array to 16kHz
    downsampleBuffer(inputBuffer, inputSampleRate, outputSampleRate = 16000) {
        if (inputSampleRate === outputSampleRate) {
            return inputBuffer;
        }
        const sampleRateRatio = inputSampleRate / outputSampleRate;
        const newLength = Math.round(inputBuffer.length / sampleRateRatio);
        const result = new Float32Array(newLength);
        let offsetResult = 0;
        let offsetBuffer = 0;

        while (offsetResult < result.length) {
            const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
            let accum = 0, count = 0;
            for (let i = offsetBuffer; i < nextOffsetBuffer && i < inputBuffer.length; i++) {
                accum += inputBuffer[i];
                count++;
            }
            result[offsetResult] = count > 0 ? accum / count : 0;
            offsetResult++;
            offsetBuffer = nextOffsetBuffer;
        }
        return result;
    }

    async startRecording() {
        if (!this.isConnected) {
            await this.connect();
        }

        if (!this.audioCtx) {
            this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        }
        if (this.audioCtx.state === 'suspended') {
            await this.audioCtx.resume();
        }

        this.mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            }
        });

        const inputSampleRate = this.audioCtx.sampleRate;
        console.log(`[AudioContext] Native hardware sample rate: ${inputSampleRate}Hz (will downsample to 16000Hz)`);

        const source = this.audioCtx.createMediaStreamSource(this.mediaStream);
        const bufferSize = 2048;
        this.processorNode = this.audioCtx.createScriptProcessor(bufferSize, 1, 1);
        this.sampleBuffer16k = [];

        this.processorNode.onaudioprocess = (e) => {
            if (!this.isRecording || !this.ws || this.ws.readyState !== WebSocket.OPEN) return;

            const inputData = e.inputBuffer.getChannelData(0);

            // Compute volume for visualizer
            let sum = 0;
            for (let i = 0; i < inputData.length; i++) {
                sum += inputData[i] * inputData[i];
            }
            const rms = Math.sqrt(sum / inputData.length);
            if (this.onVisualizer) this.onVisualizer(rms);

            // Downsample to 16kHz
            const downsampled = this.downsampleBuffer(inputData, inputSampleRate, 16000);
            for (let i = 0; i < downsampled.length; i++) {
                this.sampleBuffer16k.push(downsampled[i]);
            }

            // Send in exact chunks of 512 samples (16-bit PCM = 1024 bytes)
            const CHUNK_SIZE = 512;
            while (this.sampleBuffer16k.length >= CHUNK_SIZE) {
                const chunk = this.sampleBuffer16k.splice(0, CHUNK_SIZE);
                const pcm16 = new Int16Array(CHUNK_SIZE);
                for (let i = 0; i < CHUNK_SIZE; i++) {
                    let s = Math.max(-1, Math.min(1, chunk[i]));
                    pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                }
                this.ws.send(pcm16.buffer);
            }
        };

        // Connect processor via mute gain to destination to keep Chrome audio graph alive without echo
        this.muteGain = this.audioCtx.createGain();
        this.muteGain.gain.value = 0;

        source.connect(this.processorNode);
        this.processorNode.connect(this.muteGain);
        this.muteGain.connect(this.audioCtx.destination);
        this.isRecording = true;
        console.log('[Microphone] Audio capture started successfully! Context state:', this.audioCtx.state);
    }

    stopRecording() {
        this.isRecording = false;
        this.sampleBuffer16k = [];
        if (this.processorNode) {
            this.processorNode.disconnect();
            this.processorNode = null;
        }
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(t => t.stop());
            this.mediaStream = null;
        }
    }
}

window.VoiceClient = VoiceClient;
