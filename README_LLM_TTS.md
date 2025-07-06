# LLM-Powered Real-Time TTS System

This system combines a Large Language Model (LLM) running in Docker with a Text-to-Speech (TTS) client for real-time conversational AI with ultra-low latency.

## Features

- **Real-time streaming**: LLM tokens are streamed via WebSocket and immediately converted to speech
- **Natural speech patterns**: The LLM is prompted to speak naturally with filler words and pauses
- **Low latency**: Optimized for <200ms latency from query to first audio
- **Instant stop**: Press 's' to immediately stop both LLM generation and audio playback
- **Latency tracking**: Monitors and reports latency at each stage

## Architecture

```
User Input → TTS Client → WebSocket → LLM Docker Container
                ↓                           ↓
            Audio Output ← TTS Engine ← Token Stream
```

## Setup

### 1. Prerequisites

- Docker and Docker Compose (with NVIDIA GPU support for optimal performance)
- Python 3.11
- CUDA-capable GPU (optional but recommended)
- `speaker_ref.wav` file in the project root

### 2. Build and Run LLM Container

```bash
# Build and start the LLM server
docker-compose up -d

# Check logs
docker-compose logs -f llm-server
```

The LLM server will:
- Download the model on first run (cached for subsequent runs)
- Start WebSocket server on port 8765
- Wait for connections

### 3. Install TTS Dependencies

```bash
# Use your Python 3.11 environment
.venv-p311/Scripts/activate  # Windows
# or
source .venv-p311/bin/activate  # Linux/Mac

# Install additional dependencies
pip install websockets asyncio
```

### 4. Run the TTS Client

```bash
.venv-p311/Scripts/python.exe tts_llm_client.py
```

## Usage

1. Type your query when prompted
2. The system will:
   - Send query to LLM via WebSocket
   - Stream LLM tokens as they're generated
   - Convert complete sentences to speech in real-time
   - Play audio with minimal latency

3. Press 's' at any time to stop the current response
4. Type 'quit' to exit

## Latency Metrics

The system tracks and reports:
- **LLM first token latency**: Time from query to first LLM token
- **Total latency**: Time from query to first audio output
- **LLM stats**: Total generation time, tokens generated, tokens/sec

## Configuration

### LLM Server (`llm_server.py`)
- `MODEL_NAME`: Change to use different models (default: microsoft/phi-2)
- `SYSTEM_PROMPT`: Modify for different speaking styles

### TTS Client (`tts_llm_client.py`)
- `LLM_WEBSOCKET_URL`: WebSocket server address
- `SPEAKER_WAV_PATH`: Path to voice reference file
- TTS parameters for quality/latency tradeoff

## Troubleshooting

1. **Model download fails**: Ensure you have enough disk space and internet connection
2. **CUDA out of memory**: Use a smaller model or CPU inference
3. **High latency**: Check GPU availability and network connection
4. **Audio issues**: Ensure `speaker_ref.wav` exists and is valid

## Alternative Models

For older transformers versions, you can use:
- `microsoft/phi-2` (default, works with transformers 4.35+)
- `meta-llama/Llama-2-7b-chat-hf` (requires access approval)
- `mistralai/Mistral-7B-Instruct-v0.1`

## Development

To modify the natural speech patterns, edit the `SYSTEM_PROMPT` in `llm_server.py`.

To adjust the sentence detection for TTS chunking, modify the `sentence_endings` list in `tts_llm_client.py`. 