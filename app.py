import modal
import asyncio
import numpy as np
import re
import random

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# aiortc imports moved inside functions to avoid local dependency issues
# (only used in experimental WebRTC endpoints)


# ------------------------
# Human Realism Text Preprocessing
# ------------------------

def add_natural_speech_patterns(text: str, speaker: str = "HOST") -> str:
    """
    Add micro-level human speech patterns to text before TTS generation.
    This makes the model generate more realistic, imperfect human speech.
    """

    # Remove existing artificial markers
    text = text.replace('[laughs]', '').replace('[chuckles]', '').replace('[excited]', '')
    text = text.replace('[nervous]', '').replace('[serious]', '').strip()

    # Speaker-specific patterns
    if speaker == "HOST":
        # Joe Rogan style - curious, reactive
        filler_phrases = [
            ("really", "wait really"),
            ("interesting", "that's interesting"),
            ("I think", "I mean I think"),
            ("that's", "man that's"),
            ("wild", "that's wild"),
        ]
        breath_before = ["Wait", "Whoa", "So", "But"]

    elif speaker == "EXPERT":
        # Thoughtful, measured
        filler_phrases = [
            ("actually", "it's actually"),
            ("I think", "you know I think"),
            ("important", "really important"),
            ("basically", "so basically"),
        ]
        breath_before = ["Now", "Well", "So", "And"]

    elif speaker == "COMEDIAN":
        # Quick wit with pauses for effect
        filler_phrases = [
            ("like", "it's like"),
            ("right", "right?"),
            ("I mean", "I mean come on"),
            ("but", "but like"),
        ]
        breath_before = ["Wait", "Okay", "So", "But", "Dude"]

    else:
        # Default patterns
        filler_phrases = [
            ("I think", "I mean I think"),
            ("really", "really"),
        ]
        breath_before = ["Well", "So", "Now"]

    # Add breaths before emotional or emphatic starts (25% chance)
    for phrase in breath_before:
        if random.random() < 0.25:
            text = text.replace(f" {phrase} ", f" ... {phrase} ")
            text = text.replace(f"{phrase} ", f"... {phrase} ")

    # Add natural filler words (30% chance)
    for original, replacement in filler_phrases:
        if random.random() < 0.3 and original in text:
            text = text.replace(original, replacement, 1)  # Only first occurrence

    # Add micro-pauses before important/emphatic words (20% chance)
    emphasis_words = ["really", "never", "always", "exactly", "totally", "completely", "definitely"]
    for word in emphasis_words:
        if random.random() < 0.2 and f" {word} " in text.lower():
            text = re.sub(f" {word} ", f" .. {word} ", text, count=1, flags=re.IGNORECASE)

    # Add slight hesitation on complex words (15% chance)
    complex_patterns = [
        r'\b\w{12,}\b',  # Long words (12+ chars)
    ]
    for pattern in complex_patterns:
        matches = list(re.finditer(pattern, text))
        if matches and random.random() < 0.15:
            match = random.choice(matches)
            word = match.group()
            text = text[:match.start()] + ".. " + word + text[match.end():]

    # Natural sentence-ending variations
    # Sometimes breath after sentence (40% chance)
    sentences = re.split(r'([.!?])\s+', text)
    result = []
    for i, part in enumerate(sentences):
        result.append(part)
        if part in '.!?' and i < len(sentences) - 1:
            if random.random() < 0.4:
                result.append(' ... ')
            else:
                result.append(' ')
    text = ''.join(result)

    # Clean up excessive spacing
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s*\.\.\.\s*', '... ', text)
    text = text.strip()

    return text


def add_audio_human_artifacts(audio: np.ndarray, sr: int, speaker: str = "HOST") -> np.ndarray:
    """
    Add micro-level acoustic imperfections that make synthesized speech sound human.
    This simulates natural vocal cord instability, breath noise, and dynamic variations.
    """

    # Ensure audio is float32 and in range [-1, 1]
    audio = audio.astype(np.float32)
    audio = np.clip(audio, -1, 1)

    # 1. PITCH MICRO-JITTER (simulates vocal cord instability)
    # Real human vocal cords can't maintain perfectly stable pitch
    # Add subtle random frequency modulation ±0.2-0.5% (2-5 cents)
    jitter_amount = 0.003 if speaker == "COMEDIAN" else 0.002
    jitter = np.random.randn(len(audio)) * jitter_amount
    # Smooth the jitter (vocal cords don't change instantly)
    from scipy.ndimage import gaussian_filter1d
    jitter = gaussian_filter1d(jitter, sigma=sr*0.01)  # 10ms smoothing
    audio = audio * (1 + jitter)

    # 2. AMPLITUDE MICRO-VARIATIONS (simulates breath support variations)
    # Human breath support isn't perfectly steady
    amplitude_drift = np.random.randn(len(audio)) * 0.015
    amplitude_drift = gaussian_filter1d(amplitude_drift, sigma=sr*0.05)  # 50ms smoothing
    audio = audio * (1 + amplitude_drift)

    # 3. ROOM TONE / BACKGROUND NOISE (very subtle)
    # Even in quiet rooms, there's ambient noise and recording self-noise
    room_tone = np.random.randn(len(audio)) * 0.0003  # Very quiet, -70dB
    audio = audio + room_tone

    # 4. VOCAL BREATHINESS (high-frequency noise component)
    # Human voices have natural breathiness, especially on certain phonemes
    breathiness = np.random.randn(len(audio)) * 0.001
    # High-pass filter the breathiness (breath noise is high frequency)
    from scipy.signal import butter, filtfilt
    b, a = butter(2, 2000 / (sr / 2), btype='high')
    breathiness = filtfilt(b, a, breathiness)
    # Apply more on quieter sections (breath is more audible when not speaking loudly)
    breath_envelope = 1 - np.abs(audio) * 0.5
    audio = audio + breathiness * breath_envelope

    # 5. SUBTLE COMPRESSION (humans don't have unlimited dynamic range)
    # Simulates natural voice dynamics - loud parts slightly compressed
    # This is what real human vocal anatomy does
    audio = np.tanh(audio * 1.15) * 0.92

    # 6. FORMANT MICRO-VARIATIONS (simulates tiny vocal tract changes)
    # Real speakers have constant micro-movements of tongue, lips, jaw
    # Add very subtle random spectral tilt variation
    spectral_variation = np.random.randn(len(audio)) * 0.008
    spectral_variation = gaussian_filter1d(spectral_variation, sigma=sr*0.02)
    audio = audio * (1 + spectral_variation)

    # 7. NATURAL CLIPPING PREVENTION WITH ORGANIC LIMITING
    # Instead of hard clipping, soft-limit like human voice does
    audio = np.tanh(audio * 0.95)

    # Final normalization to -0.5 to 0.5 range (leaves headroom)
    audio = audio * 0.5

    return audio

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
# Voice profiles for podcast cast (HYPER-REALISTIC HUMAN)
# ------------------------

VOICE_HOST = """
VOCAL PHYSIOLOGY:
Male, late 30s. Fundamental frequency 95-110Hz with natural jitter ±3-8Hz per phoneme.
Chest voice resonance dominant. Vocal tract length 16.5cm. Subglottal pressure varies naturally.

MICRO-LEVEL REALISM:
- Pitch instability: every vowel has 2-5Hz micro-fluctuations, never flat
- Consonant attacks: soft, slightly breathy onset on p/t/k sounds
- Vocal fry: occasional creaky voice on sentence-final words when relaxed
- Aspiration: natural breathiness, especially after phrases longer than 8 words
- Coarticulation: sounds naturally blend and influence each other

PROSODIC NATURALNESS:
- Speaking rate: 145-165 WPM but VARIES - speeds up when excited, slows for emphasis
- Pauses: 200-400ms between clauses, 500-800ms between thoughts, never uniform
- Rhythm: completely irregular, human imperfection in timing
- Intonation: falls naturally at sentence ends, rises slightly on continuations
- Stress: unpredictable emphasis - not every important word is stressed

HUMAN IMPERFECTIONS:
- Tiny hesitations before complex words (20-50ms)
- Slight pitch drop when thinking or being serious
- Occasional glottal stops between vowels
- Breath sounds audible every 10-15 words - natural, not exaggerated
- Minute timing inconsistencies - never robotic precision
- Volume micro-variations within single words

EMOTIONAL LEAKAGE:
- Warmth bleeds through in relaxed larynx, open throat
- Curiosity shows in slight pitch lift on questions (not exaggerated)
- Enthusiasm: faster rate + 5-10Hz pitch increase + more dynamic range
- Seriousness: slower rate + lower pitch + reduced pitch range
- Engagement: forward resonance, slight smile in voice quality

AVOID:
- Perfect pitch stability (sounds synthetic)
- Uniform rhythm or timing (sounds robotic)
- Clean consonants (too crisp = fake)
- Consistent loudness (humans vary constantly)
- Theatrical delivery (we want real conversation)

Think: Joe Rogan's natural curiosity + podcast intimacy + complete human imperfection.
"""

VOICE_EXPERT = """
VOCAL PHYSIOLOGY:
Female, early 40s. Fundamental frequency 180-210Hz with organic pitch variance ±4-10Hz.
Mixed chest-head voice. Balanced resonance. Forward placement but not nasal.

MICRO-LEVEL REALISM:
- Pitch wanders naturally - never holds steady on sustained vowels
- Breathiness varies: more breathy on intimate topics, clearer on factual points
- Vocal onset: gentle, no hard glottal attacks
- Formant transitions: smooth, natural tongue/lip movements reflected in sound
- Occasional slight nasalization on continuant sounds (m, n, ng)

PROSODIC NATURALNESS:
- Speaking rate: 155-175 WPM, but modulates constantly
- Pauses: varies 150-600ms, sometimes cuts pauses short in flowing speech
- Rhythm: lilting, slightly melodic but NEVER sing-song or artificial
- Sentence endings: gentle fall, not dramatic drops
- Stress patterns: unexpected, human-like emphasis choices

HUMAN IMPERFECTIONS:
- Micro-stutters on occasional word onsets (barely perceptible)
- Breath timing: sometimes mid-phrase if sentence runs long
- Pitch drift: slowly rises or falls across long utterances
- Slight de-voicing on word-final consonants when tired or relaxed
- Inconsistent articulation precision - clearer on important points
- Minute voice quality shifts: clearer → breathier → clearer

EMOTIONAL LEAKAGE:
- Passion: increased pitch range + faster rate + forward resonance
- Empathy: softer tone + slower rate + warmer quality
- Intellectual mode: slight increase in precision, but still natural
- Confidence: steady tone but never rigid
- Warmth: smile resonance, open throat, relaxed articulation

AVOID:
- Clinical academic delivery (too perfect)
- Constant perfect enunciation (unrealistic)
- Flat affect (humans are always emotionally present)
- Predictable intonation patterns (boring and fake)
- Radio announcer quality (too polished)

Think: Intelligent friend explaining over coffee + natural imperfection + authentic engagement.
"""

VOICE_COMEDIAN = """
VOCAL PHYSIOLOGY:
Male, late 20s. Fundamental frequency 120-145Hz, but wide expressive range 90-200Hz.
Relaxed vocal mechanism. Slight natural rasp/breathiness adds character.

MICRO-LEVEL REALISM:
- Pitch jumps: natural swoops and slides for emphasis, not theatrical
- Vocal quality shifts: clear → breathy → slightly rough based on content
- Timing micro-variations: holds vowels slightly longer for comic timing
- Glottal fry: occasional creaky voice for effect or casual delivery
- Dynamic range: whispers to full voice, but transitions are organic

PROSODIC NATURALNESS:
- Speaking rate: HIGHLY variable 130-200 WPM depending on comedic rhythm
- Pauses: strategic but feel spontaneous, 100ms-1.5s range
- Rhythm: syncopated, jazzy timing - unexpected beats
- Punchlines: slight pause before + pitch/timing shift during
- Setup vs delivery: rate changes, pitch changes, quality changes

HUMAN IMPERFECTIONS:
- Sometimes rushes through setups when excited
- Occasional breaths in weird places (adds spontaneity)
- Pitch control slightly loose - overshoots and corrects naturally
- Volume varies constantly - louder on emphasis, quieter on asides
- Articulation: crisp on punchlines, looser on casual remarks
- False starts and self-corrections (feels unscripted)

EMOTIONAL LEAKAGE:
- Amusement at own jokes: smile in voice, slight laugh-quality
- Building excitement: rate increases, pitch lifts, energy builds
- Deadpan moments: flatter but never robotic, slight twinkle remains
- Playfulness: bouncy intonation, lighter voice quality
- Genuine reactions: spontaneous pitch jumps, timing breaks

AVOID:
- Standup comedian announcer voice (too performed)
- Consistent energy level (humans fluctuate)
- Perfect joke delivery every time (feels rehearsed)
- No vocal mistakes or variations (too clean)
- Forced enthusiasm (cringe)

Think: Funniest friend at the bar + natural wit + completely unrehearsed + authentic human mess.
"""

VOICE_NARRATOR = """
VOCAL PHYSIOLOGY:
Male, 40s. Fundamental frequency 90-105Hz. Rich harmonic spectrum. Full chest resonance.
Professional voice control but retains human qualities - not synthetic perfection.

MICRO-LEVEL REALISM:
- Sustained notes have subtle pitch drift ±2-6Hz (living breath support)
- Consonants: precise but organic, not mechanical
- Vowel quality: rich, slightly dark, but micro-variations in brightness
- Subglottal pressure: natural variations create dynamic micro-changes
- Formants: subtle shifts even within sustained phonemes

PROSODIC NATURALNESS:
- Speaking rate: 135-155 WPM, but dramatically varies for storytelling
- Pauses: precisely placed but feel inevitable, 300ms-2s range
- Rhythm: wavelike, building and releasing tension organically
- Pitch contours: sweeping arcs but never predictable or repetitive
- Dynamics: whisper to full voice with infinite gradations

HUMAN IMPERFECTIONS:
- Breath sounds: audible, natural part of phrasing (not hidden)
- Slight vocal fatigue: tiny roughness on extended passages adds authenticity
- Micro-timing variations: never metronomic, always slightly ahead or behind "perfect"
- Pitch accuracy: aims for target but has human variance ±5-10Hz
- Resonance shifts: slight changes in vocal tract shape during long passages
- Energy varies: higher at story peaks, more intimate in quiet moments

EMOTIONAL LEAKAGE:
- Suspense: tighter voice, slightly tense quality, slower rate
- Excitement: faster rate, higher pitch, more forward placement
- Sadness: slower, lower, slightly breathy, reduced dynamics
- Wonder: softer, more breath, gentle pitch rises
- Authority: full voice, steady (but not rigid) pitch, clear articulation
- Every emotion shows in voice quality, not just pitch/volume

AVOID:
- Audiobook AI perfection (too clean, obviously synthetic)
- Unvarying resonance (real voices shift constantly)
- Perfect breath control (humans need air)
- Mechanical rhythm (destroys immersion)
- Emotional detachment (narrator is human, stories affect them)

Think: Greatest audiobook narrator alive + complete human vulnerability + voice that breathes and lives.
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
    def generate(self, text: str, voice_style: str = VOICE_HOST, speaker: str = "HOST"):
        """
        Generate ultra-realistic human speech with micro-level imperfections.

        This method applies THREE layers of humanization:
        1. Text preprocessing: Adds natural speech patterns (fillers, pauses, breaths)
        2. Model generation: Uses high temperature + detailed prompts for variation
        3. Audio post-processing: Adds acoustic micro-imperfections (pitch jitter, noise, dynamics)

        Args:
            text: The text to synthesize
            voice_style: Detailed voice instruction prompt
            speaker: Speaker type for humanization patterns (HOST/EXPERT/COMEDIAN/NARRATOR)
        """
        import numpy as np
        from scipy.ndimage import gaussian_filter1d
        from scipy.signal import butter, filtfilt

        # LAYER 1: Text humanization
        humanized_text = add_natural_speech_patterns(text, speaker)

        # LAYER 2: Generate with speaker-tuned parameters for maximum naturalness
        # Different speakers need different generation parameters
        if speaker == "COMEDIAN":
            # Comedian needs high variation for expressiveness
            temp = 0.98
            top_p = 0.96
            rep_penalty = 1.2
        elif speaker == "EXPERT":
            # Expert is more measured but still natural
            temp = 0.92
            top_p = 0.94
            rep_penalty = 1.15
        elif speaker == "NARRATOR":
            # Narrator needs controlled variation with rich dynamics
            temp = 0.90
            top_p = 0.93
            rep_penalty = 1.1
        else:  # HOST or default
            # Host is naturally expressive and reactive
            temp = 0.95
            top_p = 0.95
            rep_penalty = 1.15

        wavs, sr = self.model.generate_voice_design(
            text=humanized_text,
            instruct=voice_style,
            language="English",
            temperature=temp,
            max_new_tokens=1536,
            top_p=top_p,
            repetition_penalty=rep_penalty,
        )

        audio = wavs[0]

        # LAYER 3: Acoustic humanization - add micro-level imperfections
        audio = add_audio_human_artifacts(audio, sr, speaker)

        return audio, sr


# ------------------------
# WebRTC Audio Track (Experimental - only used in /offer endpoint)
# ------------------------

def get_tts_audio_track_class():
    """Lazy load TTSAudioTrack to avoid importing aiortc at module level"""
    from aiortc.mediastreams import AudioStreamTrack

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
                audio, sr = await asyncio.to_thread(self.tts.generate.remote, sentence, VOICE_HOST, "HOST")

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

    return TTSAudioTrack


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
                    voice_style,
                    speaker  # Pass speaker for humanization
                )
                return audio, sr

            # Generate all segments in parallel (MASSIVE SPEEDUP)
            print(f"🚀 Generating {len(segments)} audio segments in parallel...")
            results = await asyncio.gather(*[
                generate_segment(i, speaker, text)
                for i, (speaker, text) in enumerate(segments)
            ])

            # Build final audio with NATURAL VARIABLE pauses
            all_audio = []
            sample_rate = None

            for i, (audio, sr) in enumerate(results):
                if sample_rate is None:
                    sample_rate = sr

                all_audio.append(audio)

                # Add VARIABLE natural pauses between speakers (0.5-1.0s)
                # Humans don't pause uniformly - varies by context
                # Shorter pauses within fast exchanges, longer when switching topics
                if i < len(results) - 1:  # Don't add pause after last segment
                    # Random pause duration with natural distribution
                    pause_duration = np.random.uniform(0.5, 1.0)
                    # Occasionally longer pause (10% chance) for topic shifts
                    if random.random() < 0.1:
                        pause_duration = np.random.uniform(1.2, 1.8)
                    # Occasionally shorter pause (15% chance) for quick back-and-forth
                    elif random.random() < 0.15:
                        pause_duration = np.random.uniform(0.3, 0.5)

                    pause = np.zeros(int(sr * pause_duration))
                    # Add subtle room tone during pauses (not dead silence)
                    room_ambience = np.random.randn(len(pause)) * 0.0002
                    pause = pause + room_ambience
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
                    voice_style,
                    speaker  # Pass speaker for humanization
                )
                return audio, sr

            print(f"🚀 Generating {len(segments)} audio segments in parallel...")
            results = await asyncio.gather(*[
                generate_segment(i, speaker, text)
                for i, (speaker, text) in enumerate(segments)
            ])

            all_audio = []
            sample_rate = None

            for i, (audio, sr) in enumerate(results):
                if sample_rate is None:
                    sample_rate = sr
                all_audio.append(audio)

                # Variable natural pauses (same as audio endpoint)
                if i < len(results) - 1:
                    pause_duration = np.random.uniform(0.5, 1.0)
                    if random.random() < 0.1:
                        pause_duration = np.random.uniform(1.2, 1.8)
                    elif random.random() < 0.15:
                        pause_duration = np.random.uniform(0.3, 0.5)
                    pause = np.zeros(int(sr * pause_duration))
                    room_ambience = np.random.randn(len(pause)) * 0.0002
                    pause = pause + room_ambience
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
                audio, sr = await asyncio.to_thread(tts.generate.remote, sentence, VOICE_NARRATOR, "NARRATOR")

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
        # Import aiortc only when this endpoint is used (experimental)
        from aiortc import RTCPeerConnection, RTCSessionDescription

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
        TTSAudioTrack = get_tts_audio_track_class()
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

