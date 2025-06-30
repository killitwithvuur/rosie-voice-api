import os
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from transformers import CsmForConditionalGeneration, AutoProcessor
import torch
import torchaudio
import tempfile

app = FastAPI()
device = "cpu"

# 💡 Load HF token from environment
hf_token = os.environ.get("HUGGINGFACE_TOKEN")
if not hf_token:
    raise RuntimeError("HUGGINGFACE_TOKEN not set in environment!")

print("🔌 Loading model and processor with token...")
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
