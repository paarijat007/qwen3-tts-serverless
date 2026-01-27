# AI Podcast Generator 🎙️

Generate full podcast episodes with realistic multi-voice conversations on ANY topic using AI. Powered by [Modal](https://modal.com), [Grok AI](https://x.ai), and [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign).

## Features

- **🎭 Multi-Voice Podcasts**: 3 distinct AI voices (Host, Expert, Comedian) with natural expressiveness
- **🚀 Parallel Generation**: All audio segments generated simultaneously for speed
- **🎬 Audio & Video**: Create both audio podcasts (WAV) and video with waveform visualization (MP4)
- **💬 Natural Dialogue**: Uses Grok AI to generate realistic, conversational scripts with filler words, interruptions, and natural pacing
- **⚡ Fast**: Full 2-3 minute podcast in under 90 seconds
- **🌐 Web Interface**: Beautiful, responsive web UI included
- **💰 Cost Optimized**: L4 GPU, parallel processing, auto-scaling

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/paarijat007/qwen3-tts-serverless.git
cd qwen3-tts-serverless
pip install modal
modal setup

# 2. Add your Grok API key
modal secret create grok-api-key GROK_API_KEY=your-xai-key

# 3. Deploy
modal deploy app.py

# 4. Open the URL Modal gives you
# Example: https://yourname--aiden-webrtc-tts-web.modal.run
```

Generate podcasts on any topic: "Why everyone's ex looks exactly the same", "The psychology of procrastination", etc.

## How It Works

```
User Topic → Grok AI Script → Parse Segments → Parallel TTS → Assembly → WAV/MP4
```

1. **Script Generation**: Grok AI (`grok-4-1-fast-non-reasoning`) creates natural dialogue between 3 personas (12-15 exchanges)
2. **Parsing**: Script is split into speaker segments (HOST/EXPERT/COMEDIAN), emotion tags removed
3. **Parallel Voice Synthesis**: All segments generated simultaneously using `asyncio.gather()` - this is the key speedup!
4. **Assembly**: Audio segments concatenated with 0.7s natural pauses between speakers
5. **Output**: Return as WAV audio or MP4 video (with ffmpeg waveform visualization)

## Voice Profiles

- **HOST (Alex)**: Smooth, warm male voice - Joe Rogan style, naturally curious
- **EXPERT (Dr. Sarah)**: Sophisticated female voice - approachable expert
- **COMEDIAN (Mike)**: Witty male voice - naturally funny without forcing jokes

## Installation

### Prerequisites

- [Modal account](https://modal.com) - Free tier available (requires credit card for GPU access)
- [Grok API key](https://x.ai) - Get from x.ai (~$5/1M tokens)
- Python 3.11+

### Setup (5 minutes)

1. **Clone the repository**:
```bash
git clone https://github.com/paarijat007/qwen3-tts-serverless.git
cd qwen3-tts-serverless
```

2. **Install and configure Modal**:
```bash
pip install modal
modal setup  # Follow prompts to authenticate
```

3. **Create Modal secret with your Grok API key**:
```bash
modal secret create grok-api-key GROK_API_KEY=your-xai-api-key-here
```

That's it! You're ready to deploy.

## Usage

### Deploy to Modal

```bash
modal deploy app.py
```

This will give you a URL like: `https://yourusername--aiden-webrtc-tts-web.modal.run`

### Run Locally (Development)

```bash
modal serve app.py
```

Then open the local URL in your browser.

### API Usage

#### Generate Audio Podcast

```bash
curl -X POST https://your-modal-url/generate-podcast \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "The future of AI",
    "details": "Focus on alignment and safety concerns"
  }'
```

#### Generate Video Podcast

```bash
curl -X POST https://your-modal-url/generate-video \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "The future of AI",
    "details": "Focus on alignment and safety concerns"
  }'
```

## Project Structure

```
.
├── app.py                          # Main Modal application (1034 lines)
│   ├── TTS class (L4 GPU)         # Qwen3-TTS model loader and generator
│   ├── /generate-podcast          # HTTP endpoint for audio podcasts
│   ├── /generate-video            # HTTP endpoint for video with waveform
│   ├── /generate                  # Simple TTS endpoint
│   └── / (GET)                    # Web interface HTML
├── philosophical_debate_modal.py   # Philosophical debate variant (experimental)
├── index.html                      # Standalone web interface
├── .env.example                    # Environment variables template
├── LICENSE                         # Apache 2.0 License
└── README.md                       # This file
```

### Key Components

- **Lines 102-136**: TTS class with GPU configuration and model loading
- **Lines 463-654**: Main podcast generation endpoint (working, production-ready)
- **Lines 657-866**: Video generation with ffmpeg (working, production-ready)
- **Lines 46-96**: Voice profile prompts (customizable)

## Configuration

### GPU Configuration

The TTS model runs on Modal's **L4 GPU** (app.py:103):
```python
@app.cls(
    gpu="L4",              # Much cheaper than A100 (~8x cost savings)
    cpu=2,
    min_containers=0,      # Scale to zero when idle = $0 cost
    scaledown_window=300,  # 5 minute cooldown
    timeout=600,           # 10 minute max execution
)
```

### Cost Optimization Strategies

1. **Cheap GPU**: L4 instead of A100 for TTS model
2. **Parallel Processing**: All segments generated simultaneously (not sequential)
3. **Fast LLM**: `grok-4-1-fast-non-reasoning` for script generation
4. **Auto-scaling**: Containers scale down to 0 when idle (no idle costs)
5. **Efficient Model Loading**: bfloat16 precision reduces memory usage

## Technical Details

### Parallel Processing Architecture

The key innovation is **parallel TTS generation**:

```python
# All segments generated simultaneously, not sequentially!
results = await asyncio.gather(*[
    tts.generate.remote(text, voice_style)
    for speaker, text in segments
])
```

This reduces generation time from ~5 minutes (sequential) to ~90 seconds (parallel).

### Custom Voice Profiles

Edit the voice descriptions in `app.py` (lines 46-96) to customize voice characteristics:

```python
VOICE_HOST = """
Your custom voice description here...
"""
```

## Development

### Running Tests

```bash
# Generate a test podcast locally
modal run app.py::TTS.generate --text "Hello world" --voice-style "$VOICE_HOST"
```

### Debugging

Add `print()` statements in the code - Modal captures all logs in the web dashboard.

## Limitations & Known Issues

- **Generation Time**: 90-180 seconds for full podcast (includes script + audio)
- **Cold Start**: First request adds ~30 seconds for GPU warmup
- **Maximum Length**: ~5 minutes per podcast (limited by `max_tokens=3000` in Grok)
- **WebRTC**: Experimental `/offer` endpoint exists but not production-ready (use HTTP endpoints instead)
- **Video Size**: MP4 files are large (~20-50MB for 2-3 min) due to waveform rendering

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

Apache License 2.0 - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Modal](https://modal.com) - Serverless GPU infrastructure
- [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) - Voice synthesis
- [Grok AI](https://x.ai) - Script generation
- [aiortc](https://github.com/aiortc/aiortc) - WebRTC implementation

## Support

For issues and questions:
- Open an issue on GitHub
- Check Modal docs: https://modal.com/docs

---

Built with Modal 🚀
