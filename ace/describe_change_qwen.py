# /// script
# requires-python = ">=3.11"
# dependencies = ["openai"]
# ///
"""Say in words what a push changed: Qwen3.5-Omni hears the untouched clip, then the pushed clip, and describes the
difference (issue #5's chosen method). Runs on Alibaba Model Studio (international endpoint), no local model.

The key is read from DASHSCOPE_API_KEY, or else from DASHLANE_KEY in ../../.env (lyras_code/.env).

    uv run ace/describe_change_qwen.py runs/renderer_smoke_L12/untouched.wav runs/renderer_smoke_L12/neuron5455_0.3_all.wav
"""
import base64
import os
import subprocess
import sys

from openai import OpenAI

MODEL = "qwen3.5-omni-plus-2026-03-15"
BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
COMPARE = ("The first audio clip is the original. The second is the same piece after a change. Describe precisely what "
           "changed between them: which instruments or sounds appeared, disappeared, got louder or quieter, or changed "
           "character. Say what stayed the same too.")


def api_key():
    if os.environ.get("DASHSCOPE_API_KEY"):
        return os.environ["DASHSCOPE_API_KEY"]
    env = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env")
    for line in open(env):
        if line.startswith("DASHLANE_KEY="):
            return line.split("=", 1)[1].strip().strip("'\"")
    raise SystemExit("no DASHSCOPE_API_KEY and no DASHLANE_KEY in " + env)


def audio_part(path):
    # mono 128k mp3 keeps the upload small; a 10 s 48 kHz stereo wav is ~2 MB
    mp3 = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-ac", "1", "-b:a", "128k", "-f", "mp3", "-"],
                         capture_output=True, check=True).stdout
    return {"type": "input_audio",
            "input_audio": {"data": "data:;base64," + base64.b64encode(mp3).decode(), "format": "mp3"}}


def describe_change(client, untouched, pushed):
    text = ""
    # Qwen-Omni on Model Studio only answers as a stream
    for chunk in client.chat.completions.create(
            model=MODEL, stream=True, modalities=["text"], temperature=0.3,
            messages=[{"role": "user", "content": [audio_part(untouched), audio_part(pushed),
                                                   {"type": "text", "text": COMPARE}]}]):
        if chunk.choices:
            text += chunk.choices[0].delta.content or ""
    return text


if __name__ == "__main__":
    client = OpenAI(api_key=api_key(), base_url=BASE_URL)
    print(describe_change(client, sys.argv[1], sys.argv[2]), flush=True)
