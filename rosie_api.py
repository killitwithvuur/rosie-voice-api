from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from transformers import CsmForConditionalGeneration, AutoProcessor
import torch
import torchaudio
import tempfile
import os

app = FastAPI()
device = "cpu"

# 🔐 Use your Hugging Face token from environment
hf_token = os.getenv("HUGGINGFACE_TOKEN")

print("🔌 Loading model and processor...")
model = CsmForConditionalGeneration.from_pretrained("sesame/csm-1b", token=hf_token)
processor = AutoProcessor.from_pretrained("sesame/csm-1b", token=hf_token)

@app.post("/speak")
async def speak(req: Request):
    data = await req.json()
    text = data.get("text", "")
    print(f"🧠 Generating voice for: {text}")

    inputs = processor(f"[0]{text}", return_tensors="pt").to(device)
    audio = model.generate(**inputs, output_audio=True)
    audio_tensor = audio[0].unsqueeze(0).cpu()

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    torchaudio.save(tmp_file.name, audio_tensor, 24000, format="wav")
    return FileResponse(tmp_file.name, media_type="audio/wav")
