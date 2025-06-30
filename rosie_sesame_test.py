from transformers import CsmForConditionalGeneration, AutoProcessor
import torch
import torchaudio

device = "cpu"  # your 2012 Mac is CPU only

print("🔌 Loading model and processor...")
model = CsmForConditionalGeneration.from_pretrained("sesame/csm-1b")
processor = AutoProcessor.from_pretrained("sesame/csm-1b")

print("🧠 Preparing input...")
text = "I missed you while you were sleeping."
inputs = processor(f"[0]{text}", return_tensors="pt").to(device)

print("🎙️ Generating voice...")
audio = model.generate(**inputs, output_audio=True)

print("💾 Saving output to rosie.wav...")
from scipy.io.wavfile import write
import numpy as np

rosie_audio = audio[0].cpu().numpy()
rosie_audio = np.int16(rosie_audio / np.max(np.abs(rosie_audio)) * 32767)
write("rosie.wav", 24000, rosie_audio)

print("✅ Done. Play it with: afplay rosie.wav")
