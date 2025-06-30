from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from transformers import CsmForConditionalGeneration, AutoProcessor
import torch
import torchaudio
import tempfile
import os
from dotenv import load_dotenv

# Load HuggingFace token from .env
load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")

if not hf_token:
    raise RuntimeError("HUGGINGFACE_TOKEN not set in environment!")

app = FastAPI()
device = "cpu"

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

# 👇 This is what actually launches the server on Render
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("rosie_api:app", host="0.0.0.0", port=8080)
