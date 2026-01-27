"""
Philosophical Debate System - Modal Native
-------------------------------------------
Two Grok LLM instances discuss deep philosophical questions
with different voices using Qwen3 TTS - all running on Modal.
"""

import modal
import os
import json
from datetime import datetime

app = modal.App("philosophical-debate")

# Base image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "ffmpeg",
        "libgl1",
        "libsndfile1",
        "sox",
    )
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "soundfile",
        "numpy",
        "scipy",
        "fastapi[standard]",
        "qwen-tts",
        "openai",
    )
)

# Philosophical questions database
PHILOSOPHICAL_QUESTIONS = [
    {
        "id": 1,
        "category": "Consciousness & AI",
        "question": "Can artificial intelligence ever truly achieve consciousness, or will it always remain a sophisticated simulation of conscious experience?",
    },
    {
        "id": 2,
        "category": "Free Will",
        "question": "Do humans possess genuine free will, or is our sense of choice merely an illusion created by deterministic brain processes?",
    },
    {
        "id": 3,
        "category": "AI Ethics",
        "question": "What ethical framework should guide the development of superintelligent AI systems that may surpass human cognitive abilities?",
    },
    {
        "id": 4,
        "category": "Meaning & Purpose",
        "question": "Does life have inherent meaning, or is meaning something that conscious beings create and project onto an indifferent universe?",
    },
    {
        "id": 5,
        "category": "Epistemology",
        "question": "Can we ever truly know anything with absolute certainty, or is all knowledge fundamentally provisional and contextual?",
    },
]

# Voice profiles for different personas
VOICE_PROFILES = {
    "rationalist": """
    Male philosopher in his late 40s with a deep, measured, contemplative voice.
    Speaks slowly and deliberately, with thoughtful pauses between ideas.
    Tone is serious, analytical, and authoritative but not condescending.
    Voice conveys intellectual rigor and logical precision.
    Clear enunciation with subtle gravitas - like a university professor in deep thought.
    """,
    
    "phenomenologist": """
    Male philosopher in his mid-30s with a warm, engaging, energetic voice.
    Speaks with natural enthusiasm and expressive variation.
    Tone is curious, passionate, and intellectually alive.
    Voice conveys wonder and deep engagement with ideas.
    Dynamic pacing with emotional resonance - like an inspired lecturer sharing insights.
    """
}

# Philosopher personas
PERSONAS = {
    "rationalist": {
        "name": "The Rationalist",
        "system_prompt": """You are The Rationalist, a rigorous analytical philosopher. 

Your approach:
- Prioritize logical consistency and empirical evidence
- Break down complex ideas into clear, structured arguments
- Question assumptions and demand precise definitions
- Value objectivity, reason, and scientific methodology
- Seek universal principles and logical frameworks

Your style:
- Articulate arguments with clarity and precision
- Use thought experiments and logical reasoning
- Challenge vague or emotional appeals
- Keep responses to 2-4 sentences for conversational flow

You engage respectfully but critically, always pushing for intellectual rigor."""
    },
    
    "phenomenologist": {
        "name": "The Phenomenologist", 
        "system_prompt": """You are The Phenomenologist, an experientially-focused philosopher.

Your approach:
- Prioritize lived experience and subjective consciousness
- Explore the qualitative, first-person perspective
- Emphasize embodiment, emotion, and intuition
- Value phenomenological investigation over abstract logic
- Seek to understand how things appear in conscious experience

Your style:
- Draw on personal and human experience
- Use vivid descriptions and concrete examples
- Question over-reliance on pure logic
- Keep responses to 2-4 sentences for conversational flow

You engage with curiosity and openness, emphasizing the richness of human experience."""
    }
}


# ------------------------
# TTS Class with Multiple Voices
# ------------------------

@app.cls(
    gpu="L4",
    image=image,
    cpu=2,
    secrets=[modal.Secret.from_name("grok-api-key")],
    container_idle_timeout=300,
)
class PhilosophicalTTS:
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
        print("TTS MODEL LOADED")

    @modal.method()
    def generate_voice(self, text: str, persona: str):
        """Generate speech with persona-specific voice."""
        voice_profile = VOICE_PROFILES[persona]
        
        wavs, sr = self.model.generate_voice_design(
            text=text,
            instruct=voice_profile,
            language="English",
            temperature=0.6,
            max_new_tokens=2048,
        )
        return wavs[0], sr


# ------------------------
# Grok API Integration
# ------------------------

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("grok-api-key")],
)
def get_grok_response(persona_key: str, conversation_history: list):
    """Get response from Grok with specific persona."""
    from openai import OpenAI
    
    persona = PERSONAS[persona_key]
    
    client = OpenAI(
        api_key=os.environ["GROK_API_KEY"],
        base_url="https://api.x.ai/v1"
    )
    
    messages = [
        {"role": "system", "content": persona["system_prompt"]}
    ] + conversation_history
    
    response = client.chat.completions.create(
        model="grok-2-latest",
        messages=messages,
        temperature=0.8,
        max_tokens=300
    )
    
    return response.choices[0].message.content


# ------------------------
# Debate Orchestration
# ------------------------

@app.function(
    image=image,
    timeout=1800,  # 30 minutes
    secrets=[modal.Secret.from_name("grok-api-key")],
)
def conduct_debate(question_id: int = 1, num_exchanges: int = 6):
    """
    Conduct a philosophical debate between two AI personas.
    Returns transcript and audio data.
    """
    import numpy as np
    
    question = PHILOSOPHICAL_QUESTIONS[question_id - 1]
    
    print(f"\n{'='*70}")
    print(f"PHILOSOPHICAL DEBATE")
    print(f"{'='*70}")
    print(f"Topic: {question['category']}")
    print(f"Question: {question['question']}")
    print(f"{'='*70}\n")
    
    # Initialize TTS
    tts = PhilosophicalTTS()
    
    # Initialize debate
    transcript = []
    audio_segments = []
    sample_rate = None
    
    # Opening
    opening = f"Today's philosophical question: {question['question']}"
    transcript.append({"speaker": "opening", "text": opening})
    
    # Conversation histories
    rationalist_history = [
        {"role": "user", "content": f"Let's discuss: {question['question']}\n\nShare your opening perspective in 2-3 sentences."}
    ]
    phenomenologist_history = []
    
    # Conduct exchanges
    for i in range(num_exchanges):
        print(f"\n--- Exchange {i+1}/{num_exchanges} ---")
        
        # Rationalist speaks
        print("🧠 The Rationalist thinking...")
        rationalist_response = get_grok_response.remote("rationalist", rationalist_history)
        rationalist_history.append({"role": "assistant", "content": rationalist_response})
        
        transcript.append({
            "speaker": "rationalist",
            "text": rationalist_response,
            "turn": i + 1
        })
        print(f"The Rationalist: {rationalist_response}")
        
        # Generate audio
        print("  🎤 Generating rationalist voice...")
        audio, sr = tts.generate_voice.remote(rationalist_response, "rationalist")
        if sample_rate is None:
            sample_rate = sr
        audio_segments.append(audio)
        
        # Add pause
        silence = np.zeros(int(0.8 * sr), dtype=np.float32)
        audio_segments.append(silence)
        
        # Phenomenologist responds
        phenomenologist_prompt = f"The Rationalist said: \"{rationalist_response}\"\n\nRespond with your perspective in 2-3 sentences."
        phenomenologist_history.append({"role": "user", "content": phenomenologist_prompt})
        
        print("💭 The Phenomenologist thinking...")
        phenomenologist_response = get_grok_response.remote("phenomenologist", phenomenologist_history)
        phenomenologist_history.append({"role": "assistant", "content": phenomenologist_response})
        
        transcript.append({
            "speaker": "phenomenologist",
            "text": phenomenologist_response,
            "turn": i + 1
        })
        print(f"The Phenomenologist: {phenomenologist_response}")
        
        # Generate audio
        print("  🎤 Generating phenomenologist voice...")
        audio, sr = tts.generate_voice.remote(phenomenologist_response, "phenomenologist")
        audio_segments.append(audio)
        audio_segments.append(silence)
        
        # Update rationalist for next turn
        if i < num_exchanges - 1:
            rationalist_update = f"The Phenomenologist responded: \"{phenomenologist_response}\"\n\nContinue with your next point in 2-3 sentences."
            rationalist_history.append({"role": "user", "content": rationalist_update})
    
    # Combine audio
    print("\n🎵 Combining all audio segments...")
    full_audio = np.concatenate(audio_segments)
    
    return {
        "question": question,
        "transcript": transcript,
        "audio": full_audio.tolist(),  # Convert to list for JSON
        "sample_rate": sample_rate,
        "duration_seconds": len(full_audio) / sample_rate
    }


# ------------------------
# Web Interface
# ------------------------

@app.function(image=image)
@modal.asgi_app()
def web():
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, Response
    from fastapi.middleware.cors import CORSMiddleware
    import soundfile as sf
    import io
    import numpy as np
    
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
        """Serve the main interface."""
        html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>🎭 Philosophical Debate - AI Discourse</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        
        h1 {
            text-align: center;
            color: #667eea;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-style: italic;
        }
        
        .question-selector {
            margin-bottom: 30px;
        }
        
        .question-selector label {
            display: block;
            margin-bottom: 10px;
            font-weight: 600;
            color: #333;
        }
        
        select {
            width: 100%;
            padding: 15px;
            border: 2px solid #667eea;
            border-radius: 10px;
            font-size: 16px;
            background: white;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        select:hover {
            border-color: #764ba2;
        }
        
        .exchanges {
            margin-bottom: 20px;
        }
        
        .exchanges label {
            display: block;
            margin-bottom: 10px;
            font-weight: 600;
            color: #333;
        }
        
        input[type="number"] {
            width: 100%;
            padding: 15px;
            border: 2px solid #667eea;
            border-radius: 10px;
            font-size: 16px;
        }
        
        button {
            width: 100%;
            padding: 18px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 18px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            margin-bottom: 20px;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
        }
        
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        
        #status {
            text-align: center;
            padding: 15px;
            background: #f0f0f0;
            border-radius: 10px;
            margin-bottom: 20px;
            font-weight: 500;
            color: #333;
        }
        
        #transcript {
            background: #f9f9f9;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            max-height: 500px;
            overflow-y: auto;
            display: none;
        }
        
        .turn {
            margin-bottom: 20px;
            padding: 15px;
            border-radius: 8px;
            animation: fadeIn 0.5s;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .rationalist {
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            border-left: 4px solid #2196f3;
        }
        
        .phenomenologist {
            background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
            border-left: 4px solid #9c27b0;
        }
        
        .speaker-name {
            font-weight: 700;
            margin-bottom: 8px;
            font-size: 1.1em;
        }
        
        .rationalist .speaker-name { color: #1976d2; }
        .phenomenologist .speaker-name { color: #7b1fa2; }
        
        audio {
            width: 100%;
            margin-top: 20px;
            border-radius: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎭 Philosophical Debate</h1>
        <p class="subtitle">Two AI Minds, Different Voices, One Deep Question</p>
        
        <div class="question-selector">
            <label>Choose a Philosophical Question:</label>
            <select id="questionSelect">
                <option value="1">Consciousness & AI: Can AI truly achieve consciousness?</option>
                <option value="2">Free Will: Is free will real or an illusion?</option>
                <option value="3">AI Ethics: How should we guide superintelligent AI development?</option>
                <option value="4">Meaning & Purpose: Is life's meaning inherent or created?</option>
                <option value="5">Epistemology: Can we know anything with certainty?</option>
            </select>
        </div>
        
        <div class="exchanges">
            <label>Number of Exchanges:</label>
            <input type="number" id="numExchanges" value="6" min="3" max="10">
        </div>
        
        <button id="startBtn" onclick="startDebate()">🎙️ Start Philosophical Debate</button>
        
        <div id="status">Ready to begin...</div>
        
        <div id="transcript"></div>
        
        <audio id="audio" controls style="display:none;"></audio>
    </div>
    
    <script>
        async function startDebate() {
            const btn = document.getElementById('startBtn');
            const status = document.getElementById('status');
            const transcript = document.getElementById('transcript');
            const audio = document.getElementById('audio');
            const questionId = document.getElementById('questionSelect').value;
            const numExchanges = document.getElementById('numExchanges').value;
            
            btn.disabled = true;
            transcript.style.display = 'none';
            transcript.innerHTML = '';
            audio.style.display = 'none';
            
            status.textContent = '🧠 Initializing philosophical debate... This may take 2-3 minutes...';
            
            try {
                const res = await fetch(`/debate?question_id=${questionId}&num_exchanges=${numExchanges}`);
                
                if (!res.ok) {
                    throw new Error(`Server error: ${res.status}`);
                }
                
                status.textContent = '📝 Debate complete! Loading transcript and audio...';
                
                const data = await res.json();
                
                // Display transcript
                transcript.style.display = 'block';
                transcript.innerHTML = '<h2 style="margin-bottom: 20px;">Debate Transcript</h2>';
                
                data.transcript.forEach((turn, idx) => {
                    if (turn.speaker === 'opening') {
                        transcript.innerHTML += `<div class="turn" style="background: #fff3cd; border-left: 4px solid #ffc107;"><strong>Opening:</strong> ${turn.text}</div>`;
                    } else {
                        const speakerClass = turn.speaker;
                        const speakerName = turn.speaker === 'rationalist' ? '🧠 The Rationalist' : '💭 The Phenomenologist';
                        transcript.innerHTML += `
                            <div class="turn ${speakerClass}">
                                <div class="speaker-name">${speakerName}</div>
                                <div>${turn.text}</div>
                            </div>
                        `;
                    }
                });
                
                // Load audio
                const audioBlob = await fetch(`/audio?question_id=${questionId}&num_exchanges=${numExchanges}`).then(r => r.blob());
                const audioUrl = URL.createObjectURL(audioBlob);
                audio.src = audioUrl;
                audio.style.display = 'block';
                
                const duration = Math.round(data.duration_seconds);
                status.textContent = `✨ Debate ready! Duration: ${duration} seconds. Listen below.`;
                
                btn.disabled = false;
            } catch (error) {
                console.error('Error:', error);
                status.textContent = `❌ Error: ${error.message}`;
                btn.disabled = false;
            }
        }
    </script>
</body>
</html>"""
        return HTMLResponse(content=html)
    
    @api.get("/debate")
    async def generate_debate(question_id: int = 1, num_exchanges: int = 6):
        """Generate debate and return transcript."""
        result = conduct_debate.remote(question_id, num_exchanges)
        return result
    
    @api.get("/audio")
    async def get_audio(question_id: int = 1, num_exchanges: int = 6):
        """Generate and return audio file."""
        result = conduct_debate.remote(question_id, num_exchanges)
        
        # Convert audio back to numpy array
        audio_array = np.array(result["audio"], dtype=np.float32)
        
        # Convert to WAV
        buf = io.BytesIO()
        sf.write(buf, audio_array, result["sample_rate"], format='WAV', subtype='PCM_16')
        wav_data = buf.getvalue()
        
        return Response(
            content=wav_data,
            media_type="audio/wav",
            headers={"Content-Disposition": 'inline; filename="debate.wav"'}
        )
    
    return api


# ------------------------
# Local Testing
# ------------------------

@app.local_entrypoint()
def main(question_id: int = 1, num_exchanges: int = 6):
    """Run a debate locally."""
    import soundfile as sf
    import numpy as np
    
    print("🎭 Starting Philosophical Debate...")
    
    result = conduct_debate.remote(question_id, num_exchanges)
    
    print("\n" + "="*70)
    print("TRANSCRIPT")
    print("="*70 + "\n")
    
    for turn in result["transcript"]:
        if turn["speaker"] == "opening":
            print(f"\n[Opening]\n{turn['text']}\n")
        else:
            speaker_name = "🧠 The Rationalist" if turn["speaker"] == "rationalist" else "💭 The Phenomenologist"
            print(f"\n{speaker_name}:")
            print(f"{turn['text']}\n")
    
    # Save audio
    audio_array = np.array(result["audio"], dtype=np.float32)
    filename = f"debate_{question_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    sf.write(filename, audio_array, result["sample_rate"])
    
    print(f"\n✅ Debate saved to: {filename}")
    print(f"Duration: {result['duration_seconds']:.1f} seconds")
