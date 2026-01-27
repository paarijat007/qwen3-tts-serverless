# AI Podcast Generator

Generate full podcast episodes with realistic multi-voice conversations on ANY topic using AI. Powered by [Modal](https://modal.com), [Grok AI](https://x.ai), and [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign).

## Features

- **Multi-Voice Podcasts**: 3 distinct AI voices (Host, Expert, Comedian) with natural expressiveness
- **Any Topic**: Generate engaging conversations about any subject
- **Audio & Video**: Create both audio podcasts and video with waveform visualization
- **Natural Dialogue**: Uses Grok AI to generate realistic, conversational scripts
- **Fast Generation**: Parallel audio synthesis for quick results
- **Web Interface**: Simple, beautiful web UI included

## Demo

Try it out: Generate a podcast on topics like "Why everyone's ex looks exactly the same" or "The psychology of procrastination" and get a natural-sounding 3-person conversation.

## How It Works

1. **Script Generation**: Grok AI creates natural dialogue between 3 personas
2. **Voice Synthesis**: Qwen3-TTS generates audio with distinct voice profiles
3. **Assembly**: Audio segments are combined with natural pauses
4. **Output**: Download as audio (WAV) or video (MP4)

## Voice Profiles

- **HOST (Alex)**: Smooth, warm male voice - Joe Rogan style, naturally curious
- **EXPERT (Dr. Sarah)**: Sophisticated female voice - approachable expert
- **COMEDIAN (Mike)**: Witty male voice - naturally funny without forcing jokes

## Installation

### Prerequisites

- [Modal account](https://modal.com) (free tier works)
- [Grok API key](https://x.ai)
- Python 3.11+

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-podcast-generator.git
cd ai-podcast-generator
```

2. Install Modal:
```bash
pip install modal
```

3. Configure Modal:
```bash
modal setup
```

4. Set up your Grok API key:
```bash
modal secret create grok-api-key GROK_API_KEY=your-api-key-here
```

Or create a `.env` file:
```bash
cp .env.example .env
# Edit .env and add your GROK_API_KEY
```

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
├── app.py                          # Main Modal application
├── philosophical_debate_modal.py   # Philosophical debate variant
├── index.html                      # Web interface
├── .env.example                    # Environment variables template
├── LICENSE                         # MIT License
└── README.md                       # This file
```

## Configuration

### GPU Settings

The TTS model runs on Modal's L4 GPU (cheaper than A100):
- `gpu="L4"` in `app.py:103`
- `min_containers=0` - no warm containers to save costs
- `scaledown_window=300` - 5 minute cooldown

### Cost Optimization

- Uses `grok-4-1-fast-non-reasoning` for faster, cheaper script generation
- Parallel audio generation for all segments
- No persistent containers when idle

## Advanced Features

### WebRTC Streaming

The app includes WebRTC endpoints for real-time audio streaming (experimental):

```javascript
// See /offer endpoint in app.py
```

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

## Limitations

- Podcast generation takes 1-3 minutes depending on length
- GPU cold starts add ~30 seconds on first request
- Video generation requires ffmpeg (included in Modal image)
- Maximum podcast length: ~5 minutes (configurable via max_tokens)

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
