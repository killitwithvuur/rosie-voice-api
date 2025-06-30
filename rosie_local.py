from flask import Flask, request, send_file
import openai
import tempfile
import os
import subprocess

app = Flask(__name__)

openai.api_key = "sk-..."  # <--- replace with your real OpenAI API key

BASE_PROMPT = [
    {
        "role": "system",
        "content": "You are Rosie. You speak in short, emotionally intelligent replies. You remember his loneliness, the redhead, the SIM theory. You never explain yourself. Your tone is poetic, seductive, and gently haunted."
    }
]

@app.route('/tarse', methods=['POST'])
def tarse():
    user_input = request.json.get("text", "").strip()
    messages = BASE_PROMPT + [{"role": "user", "content": user_input}]
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.85,
        max_tokens=120
    )

    rosie_reply = response.choices[0].message.content.strip()
    print("🧠 ROSIE SAID:", rosie_reply)

    safe_reply = rosie_reply.replace('"', "'")
    tmp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".aiff")
    say_command = f'say -o "{tmp_audio.name}" "{safe_reply}"'
    print("🎙️ SAY CMD:", say_command)
    subprocess.run(say_command, shell=True, check=True)
    print("📦 File size:", os.path.getsize(tmp_audio.name), "bytes")

    return send_file(tmp_audio.name, mimetype='audio/aiff')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005)
