import modal
import asyncio
import numpy as np

from fastapi import FastAPI
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
from aiortc.mediastreams import AudioStreamTrack
from fastapi.middleware.cors import CORSMiddleware

# ------------------------
# Modal setup
# ------------------------

app = modal.App("aiden-webrtc-tts")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "ffmpeg",
        "libgl1",          # required by av
        "libsndfile1",
        "sox",             # required by librosa for resampling
    )
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "soundfile",
        "numpy",
        "scipy",           # for resampling
        "aiortc",          # ✅ REQUIRED
        "av",              # ✅ REQUIRED
        "fastapi[standard]",
        "qwen-tts",
        "openai",          # For Grok API (uses OpenAI SDK)
        "python-dotenv",   # For loading .env file
    )
)



# ------------------------
# Voice profiles for podcast cast (ULTRA NATURAL)
# ------------------------

VOICE_HOST = """
Incredibly smooth, warm male voice. Late 30s. Like butter - effortless and calming.
Deep resonant tone but never forced. Natural bass that puts you at ease.
Speaks with perfect pacing - never rushed, thoughtful pauses between ideas.
Voice has rich texture, like aged whiskey. Comforting and magnetic.
Expressive without being theatrical. Genuine curiosity that draws you in.
Slight smile in the voice - you can hear he's enjoying the conversation.
When excited: voice lifts naturally, authentic enthusiasm bubbles through.
When serious: drops to intimate register, creates connection with listener.
Breathes naturally, no artificial radio voice. Just a real human being present.
The kind of voice you could listen to for hours. Hypnotic, engaging, alive.
Think: prime Joe Rogan mixed with audiobook narrator calm.
"""

VOICE_EXPERT = """
Sophisticated female voice, early 40s. Smooth and articulate with natural grace.
Rich, warm tone with subtle depth. Confident but never arrogant.
Speaks clearly but with human rhythm - natural breathing, thoughtful pauses.
Voice has silk-like quality when explaining concepts. Calming and intelligent.
Expressive range: gets animated when passionate, softens when empathetic.
Smile comes through in voice naturally. Feels like talking to a wise friend.
Not stiff or academic - approachable expert who loves sharing knowledge.
Slightly melodic cadence, flows beautifully. Easy to listen to for long periods.
Voice has character and personality, not just information delivery.
"""

VOICE_COMEDIAN = """
Smooth, witty male voice, late 20s. Sharp but never abrasive.
Natural storyteller cadence. Knows how to build to a punchline effortlessly.
Voice has playful energy but stays smooth. Talks with a grin you can hear.
Expressive without being cartoonish. Real person who happens to be funny.
Rhythm varies: quick wit when riffing, slows down for emphasis on jokes.
Warm laugh that's contagious. Not forced comedy - natural charm.
Slight casual rasp that adds character. Sounds like your funniest friend.
Delivery is butter-smooth even when being ridiculous. Professional ease.
Makes you smile just by how he says things. Effortlessly entertaining.
"""

VOICE_NARRATOR = """
Masterful storytelling voice. Male, 40s. Like the best audiobook narrator you've ever heard.
Deep, rich tone with incredible warmth. Voice wraps around you like a blanket.
Perfect pacing - never rushed, gives each word its moment to breathe.
Expressive range is stunning: whispers for suspense, rises for excitement, softens for emotion.
Voice has gravitas and weight. Commands attention effortlessly.
Natural breathing creates rhythm. Pauses are powerful - silence speaks.
When describing action: voice quickens, energy builds naturally.
When describing emotion: voice becomes intimate, draws you in close.
Smooth as aged scotch. No roughness unless the story calls for it.
Hypnotic quality - you forget you're listening to AI. Just lost in the story.
The voice that makes you pull over in the car because you can't stop listening.
"""

# ------------------------
# TTS Class (GPU)
# ------------------------

@app.cls(
    gpu="L4",  # Much cheaper than A100
    image=image,
    cpu=2,
    min_containers=0,  # Don't keep warm to save money
    scaledown_window=300,  # 5 min
    timeout=600,
)
class TTS:
    @modal.enter()
    def load(self):
        import torch
        from qwen_tts import Qwen3TTSModel

        print("Loading TTS model...")
        self.model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        print("MODEL LOADED")

    @modal.method()
    def generate(self, text: str, voice_style: str = VOICE_HOST):
        wavs, sr = self.model.generate_voice_design(
            text=text,
            instruct=voice_style,
            language="English",
            temperature=0.9,  # High for natural expressiveness
            max_new_tokens=1536,  # Reduced for faster generation
            top_p=0.95,
            repetition_penalty=1.1,  # Avoid robotic repetition
        )
        return wavs[0], sr


# ------------------------
# WebRTC Audio Track
# ------------------------

class TTSAudioTrack(AudioStreamTrack):
    kind = "audio"

    def __init__(self, tts, text):
        super().__init__()
        import av
        import re
        from scipy import signal

        self.signal = signal
        self.av = av
        self.queue = asyncio.Queue()
        self.tts = tts
        self.text = text
        self.started = False

        # Split text into sentences for streaming
        self.sentences = re.split(r'(?<=[.!?])\s+', text)

    async def _run(self):
        try:
            # Generate and stream each sentence
            for sentence in self.sentences:
                if not sentence.strip():
                    continue

                print(f"Generating: {sentence[:50]}...")
                audio, sr = await asyncio.to_thread(self.tts.generate.remote, sentence)

                print(f"Generated audio shape: {audio.shape}, sr: {sr}")

                audio = np.clip(audio, -1, 1)

                # Resample using scipy
                if sr != 48000:
                    num_samples = int(len(audio) * 48000 / sr)
                    audio = self.signal.resample(audio, num_samples)
                    print(f"Resampled to 48kHz, new shape: {audio.shape}")

                frame_size = 960  # 20ms @ 48kHz
                for i in range(0, len(audio), frame_size):
                    frame = audio[i:i + frame_size]
                    await self.queue.put(frame)

                print(f"Finished streaming sentence: {sentence[:30]}...")

            await self.queue.put(None)
            print("All audio frames sent")
        except Exception as e:
            print(f"Error in _run: {e}")
            import traceback
            traceback.print_exc()
            await self.queue.put(None)

    async def recv(self):
        if not self.started:
            self.started = True
            print("Starting audio generation task...")
            asyncio.create_task(self._run())

        print("Waiting for audio frame...")
        frame = await self.queue.get()
        if frame is None:
            print("No more frames, ending track")
            raise asyncio.CancelledError

        print(f"Sending frame with {len(frame)} samples")
        aframe = self.av.AudioFrame.from_ndarray(frame, format="flt", layout="mono")
        aframe.sample_rate = 48000
        return aframe


# ------------------------
# ASGI app (WebRTC)
# ------------------------

@app.function(
    image=image,
    timeout=600,
    secrets=[modal.Secret.from_name("grok-api-key")]  # Create this with: modal secret create grok-api-key GROK_API_KEY=your-key
)
@modal.asgi_app()
def web():
    api = FastAPI()

    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @api.get("/")
    async def index():
        """Serve the HTML interface"""
        from starlette.responses import HTMLResponse

        html = """<!DOCTYPE html>
<html>
<head>
  <style>
    body {
      font-family: system-ui, -apple-system, sans-serif;
      max-width: 900px;
      margin: 50px auto;
      padding: 20px;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      min-height: 100vh;
    }
    .container {
      background: white;
      padding: 30px;
      border-radius: 15px;
      box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    }
    h1 {
      margin: 0 0 10px 0;
      color: #333;
      font-size: 2.5em;
    }
    .subtitle {
      color: #666;
      margin-bottom: 30px;
      font-size: 1.1em;
    }
    input, textarea {
      width: 100%;
      padding: 12px;
      font-size: 16px;
      border: 2px solid #e0e0e0;
      border-radius: 8px;
      margin-bottom: 15px;
      box-sizing: border-box;
      transition: border 0.3s;
    }
    input:focus, textarea:focus {
      border-color: #667eea;
      outline: none;
    }
    textarea {
      height: 120px;
      resize: vertical;
      font-family: inherit;
    }
    button {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      border: none;
      padding: 15px 30px;
      font-size: 18px;
      font-weight: bold;
      border-radius: 8px;
      cursor: pointer;
      margin-top: 10px;
      width: 100%;
      transition: transform 0.2s, box-shadow 0.2s;
    }
    button:hover {
      transform: translateY(-2px);
      box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
    }
    button:disabled {
      background: #ccc;
      cursor: not-allowed;
      transform: none;
    }
    #status {
      margin-top: 15px;
      font-style: italic;
      color: #666;
      text-align: center;
      font-size: 1.1em;
    }
    #transcript {
      margin-top: 20px;
      padding: 20px;
      background: #f8f9fa;
      border-radius: 8px;
      white-space: pre-wrap;
      display: none;
      max-height: 300px;
      overflow-y: auto;
      font-family: 'Courier New', monospace;
      font-size: 14px;
      line-height: 1.6;
    }
    audio, video {
      width: 100%;
      margin-top: 20px;
      border-radius: 8px;
    }
    video {
      background: #000;
    }
    .button-group {
      display: flex;
      gap: 10px;
      margin-top: 10px;
    }
    .button-group button {
      flex: 1;
      margin-top: 0;
    }
    .host { color: #667eea; font-weight: bold; }
    .expert { color: #e74c3c; font-weight: bold; }
    .comedian { color: #f39c12; font-weight: bold; }
  </style>
</head>
<body>
<div class="container">

<h1>🎙️ AI Podcast Generator</h1>
<div class="subtitle">Generate a full podcast episode with host, expert, and comedian on ANY topic</div>

<input id="topic" type="text" placeholder="What should the podcast be about?" value="Why everyone's ex looks exactly the same">
<textarea id="details" placeholder="Any specific angles or questions to explore? (optional)">Explore the psychology of why we're all attracted to the same 'type', how our exes could literally form a support group, and whether we're just dating different versions of the same person over and over. Make it funny and relatable.</textarea>

<div class="button-group">
  <button id="generateAudioBtn" onclick="generatePodcast('audio')">🎙️ Generate Audio</button>
  <button id="generateVideoBtn" onclick="generatePodcast('video')">🎬 Generate Video</button>
</div>

<div id="status"></div>
<div id="transcript"></div>
<audio id="audio" controls style="display:none;"></audio>
<video id="video" controls style="display:none;"></video>

</div>
<script>
async function generatePodcast(type = 'audio') {
  const audioBtn = document.getElementById('generateAudioBtn');
  const videoBtn = document.getElementById('generateVideoBtn');
  const status = document.getElementById('status');
  const audioPlayer = document.getElementById('audio');
  const videoPlayer = document.getElementById('video');
  const transcript = document.getElementById('transcript');
  const topic = document.getElementById('topic').value;
  const details = document.getElementById('details').value;

  if (!topic.trim()) {
    status.textContent = '❌ Please enter a podcast topic!';
    return;
  }

  // Disable both buttons
  audioBtn.disabled = true;
  videoBtn.disabled = true;

  // Hide both players
  audioPlayer.style.display = 'none';
  videoPlayer.style.display = 'none';
  transcript.style.display = 'none';

  const isVideo = type === 'video';
  const endpoint = isVideo ? '/generate-video' : '/generate-podcast';
  const emoji = isVideo ? '🎬' : '🎙️';

  status.textContent = '🤖 AI is writing the podcast script...';

  try {
    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ topic, details })
    });

    console.log('Response status:', res.status);

    if (!res.ok) {
      const errorText = await res.text();
      throw new Error(`${res.status}: ${errorText}`);
    }

    status.textContent = isVideo ? '🎬 Creating video with waveform...' : '🎵 Loading audio...';

    const blob = await res.blob();
    console.log('Podcast size:', (blob.size/1024/1024).toFixed(2), 'MB');

    const url = URL.createObjectURL(blob);

    if (isVideo) {
      videoPlayer.src = url;
      videoPlayer.style.display = 'block';
      videoPlayer.onloadeddata = () => {
        const duration = Math.floor(videoPlayer.duration);
        const minutes = Math.floor(duration / 60);
        const seconds = duration % 60;
        status.textContent = `🎬 Video ready! Duration: ${minutes}:${seconds.toString().padStart(2, '0')} - Right-click video to download for X`;
      };
      videoPlayer.play();
    } else {
      audioPlayer.src = url;
      audioPlayer.style.display = 'block';
      audioPlayer.onloadeddata = () => {
        const duration = Math.floor(audioPlayer.duration);
        const minutes = Math.floor(duration / 60);
        const seconds = duration % 60;
        status.textContent = `🎧 Podcast ready! Duration: ${minutes}:${seconds.toString().padStart(2, '0')}`;
      };
      audioPlayer.play();
    }

    // Re-enable buttons
    audioBtn.disabled = false;
    videoBtn.disabled = false;
  } catch (error) {
    console.error('Error:', error);
    status.textContent = `❌ Error: ${error.message}`;
    audioBtn.disabled = false;
    videoBtn.disabled = false;
  }
}
</script>

</body>
</html>"""
        return HTMLResponse(content=html)

    # Podcast generation endpoint
    @api.post("/generate-podcast")
    async def generate_podcast(data: dict):
        """Generate a multi-voice podcast using Grok AI + Qwen3-TTS"""
        try:
            from starlette.responses import Response
            import io
            import soundfile as sf
            import os
            from openai import OpenAI

            topic = data.get("topic", "")
            details = data.get("details", "")

            if not topic:
                return Response(content="Topic required", status_code=400)

            # Get Grok API key
            import os
            grok_api_key = os.environ.get("GROK_API_KEY") or os.environ.get("XAI_API_KEY")
            print(f"DEBUG: GROK_API_KEY present: {bool(grok_api_key)}")
            print(f"DEBUG: All env vars: {list(os.environ.keys())}")

            if not grok_api_key:
                return Response(
                    content="GROK_API_KEY not set. Add it to Modal secrets.",
                    status_code=500
                )

            print(f"🎙️ Generating podcast: {topic}")

            # Generate podcast script with Grok
            client = OpenAI(
                api_key=grok_api_key,
                base_url="https://api.x.ai/v1"
            )

            prompt = f"""Create a NATURAL, REALISTIC podcast conversation about: "{topic}"

Additional context: {details}

CRITICAL: Write like REAL people talk, not scripted radio:
- Use filler words (um, like, you know, I mean)
- People interrupt each other
- Incomplete thoughts and tangents
- Natural reactions ("Wait what?", "Oh damn", "For real?")
- Casual language, not formal

3 PEOPLE:
- HOST (Alex): Curious, asks "wait really?", "how does that work?". Joe Rogan style - genuinely fascinated, says "that's crazy", "whoa", "man that's wild"
- EXPERT (Dr. Sarah): Smart but talks like a normal person, not academic robot
- COMEDIAN (Mike): Funny guy, makes observations, doesn't force jokes

RULES:
- Each turn: 1-2 short sentences ONLY (people don't monologue)
- NO emotion tags like [laughs] - write natural dialogue
- Back-and-forth banter, quick exchanges
- Build to interesting points organically
- Sound like friends talking, not performing

FORMAT (follow EXACTLY):
HOST: Yo, so today we're talking about {topic.split()[0] if topic else 'this'}, right?
EXPERT: Yeah, it's actually pretty wild when you look at it.
COMEDIAN: Wait, are we really doing this? I'm into it.

Write 12-15 exchanges (short turns) now:"""

            response = client.chat.completions.create(
                model="grok-4-1-fast-non-reasoning",  # Faster non-reasoning model
                messages=[
                    {"role": "system", "content": "You are a professional podcast script writer. Create engaging, natural conversations with humor and insight."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.9,
                max_tokens=3000
            )

            script = response.choices[0].message.content
            print(f"✅ Generated script: {len(script)} chars")

            # Parse script into segments
            lines = script.split('\n')
            segments = []

            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # Parse "SPEAKER: text" format
                if ':' in line:
                    parts = line.split(':', 1)
                    speaker_raw = parts[0].strip().upper()
                    text = parts[1].strip()

                    # Remove emotion cues from text but keep natural pauses
                    text = text.replace('[laughs]', '').replace('[chuckles]', '').replace('[excited]', '')
                    text = text.replace('[nervous]', '').replace('[serious]', '').strip()

                    if not text:
                        continue

                    # Determine speaker
                    if 'HOST' in speaker_raw or 'ALEX' in speaker_raw:
                        speaker = 'HOST'
                    elif 'EXPERT' in speaker_raw or 'SARAH' in speaker_raw or 'DR' in speaker_raw:
                        speaker = 'EXPERT'
                    elif 'COMEDIAN' in speaker_raw or 'MIKE' in speaker_raw or 'COMIC' in speaker_raw:
                        speaker = 'COMEDIAN'
                    else:
                        continue

                    segments.append((speaker, text))

            print(f"✅ Parsed {len(segments)} segments")

            if len(segments) == 0:
                return Response(
                    content="Failed to parse podcast script",
                    status_code=500
                )

            # Generate audio with different voices
            tts = TTS()

            voice_map = {
                'HOST': VOICE_HOST,
                'EXPERT': VOICE_EXPERT,
                'COMEDIAN': VOICE_COMEDIAN
            }

            # Parallel TTS generation helper function
            async def generate_segment(i, speaker, text):
                voice_style = voice_map.get(speaker, VOICE_HOST)
                print(f"🎤 {i+1}/{len(segments)} - {speaker}: {text[:60]}...")
                audio, sr = await asyncio.to_thread(
                    tts.generate.remote,
                    text,
                    voice_style
                )
                return audio, sr

            # Generate all segments in parallel (MASSIVE SPEEDUP)
            print(f"🚀 Generating {len(segments)} audio segments in parallel...")
            results = await asyncio.gather(*[
                generate_segment(i, speaker, text)
                for i, (speaker, text) in enumerate(segments)
            ])

            # Build final audio with pauses
            all_audio = []
            sample_rate = None

            for audio, sr in results:
                if sample_rate is None:
                    sample_rate = sr

                all_audio.append(audio)

                # Add natural pause between speakers (0.7s)
                pause = np.zeros(int(sr * 0.7))
                all_audio.append(pause)

            # Concatenate all audio
            print(f"🎵 Combining audio...")
            full_audio = np.concatenate(all_audio)

            # Convert to WAV
            buf = io.BytesIO()
            sf.write(buf, full_audio, sample_rate, format='WAV', subtype='PCM_16')
            wav_data = buf.getvalue()

            duration = len(full_audio) / sample_rate
            print(f"✅ Podcast complete: {duration:.1f}s, {len(wav_data)/1024/1024:.2f}MB")

            return Response(
                content=wav_data,
                media_type="audio/wav",
                headers={
                    "Content-Disposition": 'inline; filename="podcast.wav"',
                    "Content-Length": str(len(wav_data)),
                    "Access-Control-Allow-Origin": "*",
                }
            )
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return Response(
                content=str(e),
                status_code=500,
                headers={"Access-Control-Allow-Origin": "*"}
            )

    # Video generation endpoint (audio + waveform visualization)
    @api.post("/generate-video")
    async def generate_video(data: dict):
        """Generate podcast video with waveform visualization"""
        try:
            from starlette.responses import Response
            import subprocess
            import tempfile
            import os

            topic = data.get("topic", "")
            details = data.get("details", "")

            if not topic:
                return Response(content="Topic required", status_code=400)

            print(f"🎬 Generating video podcast: {topic}")

            # First, generate the audio (reuse podcast generation logic)
            grok_api_key = os.environ.get("GROK_API_KEY") or os.environ.get("XAI_API_KEY")

            if not grok_api_key:
                return Response(
                    content="GROK_API_KEY not set. Add it to Modal secrets.",
                    status_code=500
                )

            # Generate podcast script
            from openai import OpenAI
            import soundfile as sf
            import io

            client = OpenAI(
                api_key=grok_api_key,
                base_url="https://api.x.ai/v1"
            )

            prompt = f"""Create a NATURAL, REALISTIC podcast conversation about: "{topic}"

Additional context: {details}

CRITICAL: Write like REAL people talk, not scripted radio:
- Use filler words (um, like, you know, I mean)
- People interrupt each other
- Incomplete thoughts and tangents
- Natural reactions ("Wait what?", "Oh damn", "For real?")
- Casual language, not formal

3 PEOPLE:
- HOST (Alex): Curious, asks "wait really?", "how does that work?". Joe Rogan style - genuinely fascinated, says "that's crazy", "whoa", "man that's wild"
- EXPERT (Dr. Sarah): Smart but talks like a normal person, not academic robot
- COMEDIAN (Mike): Funny guy, makes observations, doesn't force jokes

RULES:
- Each turn: 1-2 short sentences ONLY (people don't monologue)
- NO emotion tags like [laughs] - write natural dialogue
- Back-and-forth banter, quick exchanges
- Build to interesting points organically
- Sound like friends talking, not performing

FORMAT (follow EXACTLY):
HOST: Yo, so today we're talking about {topic.split()[0] if topic else 'this'}, right?
EXPERT: Yeah, it's actually pretty wild when you look at it.
COMEDIAN: Wait, are we really doing this? I'm into it.

Write 12-15 exchanges (short turns) now:"""

            response = client.chat.completions.create(
                model="grok-4-1-fast-non-reasoning",
                messages=[
                    {"role": "system", "content": "You are a professional podcast script writer. Create engaging, natural conversations with humor and insight."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.9,
                max_tokens=3000
            )

            script = response.choices[0].message.content
            print(f"✅ Generated script: {len(script)} chars")

            # Parse script
            lines = script.split('\n')
            segments = []

            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                if ':' in line:
                    parts = line.split(':', 1)
                    speaker_raw = parts[0].strip().upper()
                    text = parts[1].strip()

                    text = text.replace('[laughs]', '').replace('[chuckles]', '').replace('[excited]', '')
                    text = text.replace('[nervous]', '').replace('[serious]', '').strip()

                    if not text:
                        continue

                    if 'HOST' in speaker_raw or 'ALEX' in speaker_raw:
                        speaker = 'HOST'
                    elif 'EXPERT' in speaker_raw or 'SARAH' in speaker_raw or 'DR' in speaker_raw:
                        speaker = 'EXPERT'
                    elif 'COMEDIAN' in speaker_raw or 'MIKE' in speaker_raw or 'COMIC' in speaker_raw:
                        speaker = 'COMEDIAN'
                    else:
                        continue

                    segments.append((speaker, text))

            print(f"✅ Parsed {len(segments)} segments")

            if len(segments) == 0:
                return Response(content="Failed to parse podcast script", status_code=500)

            # Generate audio in parallel
            tts = TTS()
            voice_map = {
                'HOST': VOICE_HOST,
                'EXPERT': VOICE_EXPERT,
                'COMEDIAN': VOICE_COMEDIAN
            }

            async def generate_segment(i, speaker, text):
                voice_style = voice_map.get(speaker, VOICE_HOST)
                print(f"🎤 {i+1}/{len(segments)} - {speaker}: {text[:60]}...")
                audio, sr = await asyncio.to_thread(
                    tts.generate.remote,
                    text,
                    voice_style
                )
                return audio, sr

            print(f"🚀 Generating {len(segments)} audio segments in parallel...")
            results = await asyncio.gather(*[
                generate_segment(i, speaker, text)
                for i, (speaker, text) in enumerate(segments)
            ])

            all_audio = []
            sample_rate = None

            for audio, sr in results:
                if sample_rate is None:
                    sample_rate = sr
                all_audio.append(audio)
                pause = np.zeros(int(sr * 0.7))
                all_audio.append(pause)

            full_audio = np.concatenate(all_audio)
            duration = len(full_audio) / sample_rate
            print(f"✅ Audio complete: {duration:.1f}s")

            # Save audio to temp file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as audio_file:
                sf.write(audio_file.name, full_audio, sample_rate, format='WAV', subtype='PCM_16')
                audio_path = audio_file.name

            # Generate video with ffmpeg (black screen + waveform)
            video_path = tempfile.mktemp(suffix='.mp4')

            print(f"🎬 Creating video with waveform...")
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-f', 'lavfi', '-i', f'color=c=black:s=1080x1920:d={duration}',  # 9:16 vertical for X/social
                '-i', audio_path,
                '-filter_complex',
                '[1:a]showwaves=s=1080x1920:mode=cline:colors=white[waves];[0:v][waves]overlay=format=auto',
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                '-c:a', 'aac', '-b:a', '192k',
                '-shortest',
                video_path
            ]

            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"❌ ffmpeg error: {result.stderr}")
                os.unlink(audio_path)
                return Response(content=f"Video generation failed: {result.stderr}", status_code=500)

            print(f"✅ Video created: {os.path.getsize(video_path)/1024/1024:.2f}MB")

            # Read video file
            with open(video_path, 'rb') as f:
                video_data = f.read()

            # Cleanup temp files
            os.unlink(audio_path)
            os.unlink(video_path)

            return Response(
                content=video_data,
                media_type="video/mp4",
                headers={
                    "Content-Disposition": 'attachment; filename="podcast.mp4"',
                    "Content-Length": str(len(video_data)),
                    "Access-Control-Allow-Origin": "*",
                }
            )

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return Response(
                content=str(e),
                status_code=500,
                headers={"Access-Control-Allow-Origin": "*"}
            )

    # Alternative HTTP streaming endpoint
    @api.post("/generate")
    async def generate_audio(data: dict):
        """Generate audio and return as complete WAV file"""
        try:
            from starlette.responses import Response
            import io
            import soundfile as sf
            import re

            text = data.get("text", "")
            # Use TTS class directly
            tts = TTS()

            # Split into sentences
            sentences = re.split(r'(?<=[.!?])\s+', text)

            all_audio = []
            sample_rate = None

            for i, sentence in enumerate(sentences):
                if not sentence.strip():
                    continue

                print(f"Generating {i+1}/{len(sentences)}: {sentence[:50]}...")

                # Generate audio for this sentence with narrator voice
                audio, sr = await asyncio.to_thread(tts.generate.remote, sentence, VOICE_NARRATOR)

                if sample_rate is None:
                    sample_rate = sr

                all_audio.append(audio)

            # Concatenate all audio
            print(f"Concatenating {len(all_audio)} audio chunks...")
            full_audio = np.concatenate(all_audio)

            # Convert to WAV bytes
            print(f"Converting to WAV format...")
            buf = io.BytesIO()
            sf.write(buf, full_audio, sample_rate, format='WAV', subtype='PCM_16')
            wav_data = buf.getvalue()

            print(f"Total audio length: {len(full_audio)/sample_rate:.2f} seconds, size: {len(wav_data)/1024/1024:.2f} MB")

            return Response(
                content=wav_data,
                media_type="audio/wav",
                headers={
                    "Content-Disposition": 'inline; filename="story.wav"',
                    "Content-Length": str(len(wav_data)),
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "POST, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type"
                }
            )
        except Exception as e:
            print(f"Error generating audio: {e}")
            import traceback
            traceback.print_exc()
            return Response(
                content=str(e),
                status_code=500,
                headers={
                    "Access-Control-Allow-Origin": "*",
                }
            )

    # Keep peer connections alive
    pcs = set()

    @api.post("/offer")
    async def offer(data: dict):
        # No ICE servers - use host candidates only
        pc = RTCPeerConnection()
        pcs.add(pc)

        # Use TTS class directly
        tts = TTS()

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            print(f"Connection state: {pc.connectionState}")
            if pc.connectionState in ["failed", "closed"]:
                await asyncio.sleep(5)  # Keep alive a bit longer
                pcs.discard(pc)

        @pc.on("track")
        async def on_track(track):
            print(f"Track received: {track.kind}")

        @pc.on("iceconnectionstatechange")
        async def on_ice():
            print(f"ICE connection state: {pc.iceConnectionState}")
            if pc.iceConnectionState == "connected":
                print("ICE connection established!")
            elif pc.iceConnectionState == "failed":
                print("ICE connection failed!")

        @pc.on("icegatheringstatechange")
        async def on_ice_gathering():
            print(f"ICE gathering state: {pc.iceGatheringState}")

        @pc.on("icecandidate")
        async def on_ice_candidate(candidate):
            if candidate:
                print(f"ICE candidate: {candidate.candidate[:100]}")

        # Set remote description FIRST
        await pc.setRemoteDescription(
            RTCSessionDescription(
                sdp=data["sdp"],
                type=data["type"],
            )
        )

        # Then add the track
        print(f"Creating audio track with text: {data['text'][:50]}...")
        track = TTSAudioTrack(tts, data["text"])
        sender = pc.addTrack(track)
        print(f"Track added to peer connection, sender: {sender}")

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        # Wait for ICE gathering to complete
        print("Waiting for ICE gathering to complete...")
        while pc.iceGatheringState != "complete":
            await asyncio.sleep(0.1)
        print(f"ICE gathering complete!")

        print(f"Answer created with SDP length: {len(pc.localDescription.sdp)}")
        print(f"Local description type: {pc.localDescription.type}")

        # Start a background task to keep connection alive
        async def connection_handler():
            try:
                print("Connection handler started")
                # Wait for ICE to connect
                for i in range(30):  # Wait up to 30 seconds
                    if pc.iceConnectionState == "connected":
                        print("ICE connected, starting to monitor")
                        break
                    await asyncio.sleep(1)

                # Keep alive while connected
                while pc.connectionState in ["new", "connecting", "connected"]:
                    await asyncio.sleep(1)

                print(f"Connection ended: {pc.connectionState}, ICE: {pc.iceConnectionState}")
            except Exception as e:
                print(f"Connection handler error: {e}")
                import traceback
                traceback.print_exc()

        asyncio.create_task(connection_handler())

        return {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
        }


    return api

