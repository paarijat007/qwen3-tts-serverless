# app.py
import modal
import io
import os
from pydantic import BaseModel
from starlette.responses import Response

app = modal.App("qwen3-tts-voice-design-agent")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.4.1",
        "transformers>=4.45",
        "accelerate",
        "soundfile",
        "numpy",
        "fastapi[standard]",
        "qwen-tts",
    )
)

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

class TTSRequest(BaseModel):
    text: str
    voice_style: str
    language: str = "English"

@app.cls(
    gpu="L4",
    image=image,
    timeout=1800,
    scaledown_window=600,
    min_containers=1,
)
class TTS:

    @modal.enter()
    def load(self):
        import torch
        import soundfile as sf
        from qwen_tts import Qwen3TTSModel

        self.sf = sf
        self.torch = torch

        print("LOADING MODEL...")

        self.model = Qwen3TTSModel.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        print("MODEL LOADED")

    @modal.method()
    def generate(self, text: str, voice_style: str, language: str) -> bytes:
        wavs, sr = self.model.generate_voice_design(
            text=text,
            language=language,
            instruct=voice_style,
            temperature=0.7,
            max_new_tokens=2048,
        )

        buf = io.BytesIO()
        self.sf.write(
            buf,
            wavs[0],
            sr,
            format="WAV",
            subtype="PCM_16"
        )
        buf.seek(0)
        return buf.read()

@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
async def speak(req: TTSRequest):
    audio = TTS().generate.remote(
        req.text,
        req.voice_style,
        req.language
    )

    return Response(
        content=audio,
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="aiden.wav"'},
    )
