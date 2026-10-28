"""Describe what each push changed, entirely on Modal with open models, instead of the paid Alibaba API (#6).

    modal run ace/modal_omni.py::caption --root renderer_survey_d015 [--limit 50]    # 1. caption every clip
    modal run ace/modal_omni.py::judge --root renderer_survey_d015                   # 2. compare and label

1. Qwen3-Omni-30B-A3B-Captioner writes a detailed description of every clip on its own (temperature 0, so the same
   clip always gets the same caption). Each song's untouched clip is captioned once and reused for every push.
2. A text model (Qwen3-30B-A3B-Instruct-2507) gets, for one unit and push direction, the untouched and pushed captions
   for all four songs, without being told neuron from random direction, works out what changed in each song, and
   whether it is the same kind of change across songs (same verdict format as ace/label_pushes.py).
Both read clips from the ace-neuron-runs volume (any number of layer folders under the root) and write captions.jsonl
and labels.jsonl next to them; already-done items are skipped, so either step can be rerun.
"""
import base64
import json
import os
import re
import time
from collections import defaultdict

import modal

CAPTIONER = "Qwen/Qwen3-Omni-30B-A3B-Captioner"
JUDGE = "Qwen/Qwen3-30B-A3B-Instruct-2507"
JUDGE_PROMPT = """A music model generated four different songs. Each song was then regenerated with the same small \
internal change applied. A listener described each version separately, without comparing them, so wording can differ \
even where the sound did not change; ignore differences that are only phrasing.

{pairs}

First work out what actually changed in each song. Then decide whether it is the same kind of change in all four songs. \
Judge the change itself, not the songs (different songs have different instruments, so "the lead melody disappears" can \
be the same change even if the melody is piano in one song and synth in another). Reply with JSON only:
{{"per_song": ["<what changed in song 1, a few words>", "<song 2>", "<song 3>", "<song 4>"],
  "consistency": "same" | "mostly" | "different" | "none",
  "songs_sharing": <how many of the 4 songs show the shared change, 0-4>,
  "label": "<the shared change in at most 8 plain words, naming the actual sound (e.g. 'low rumbling drone replaces the music', 'heavy reverb makes it sound distant') rather than a general category; empty if there is none>",
  "destructive": <true if the shared change mostly wrecks the music (noise, garbling, unrecognisable), else false>,
  "reason": "<one sentence>"}}
"consistency" is "none" when most songs show little or no real change."""

image = (
    # vLLM's own image: CUDA toolkit and compilers included, which it needs to build GPU kernels on first start
    modal.Image.from_registry("vllm/vllm-openai:v0.31.0").entrypoint([])
    # Modal looks for `python`; this image only names it python3. vLLM is already installed; add its audio extras.
    .run_commands("ln -sf $(command -v python3) /usr/local/bin/python",
                  "python3 -m pip install --break-system-packages 'vllm[audio]==0.31.0'")
    .env({"PYTHONUNBUFFERED": "1", "HF_HOME": "/hf"})
)
hf = modal.Volume.from_name("hf-cache", create_if_missing=True)
runs = modal.Volume.from_name("ace-neuron-runs")
app = modal.App("ace-omni", image=image)
VOLS = {"/hf": hf, "/runs": runs}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def done_keys(path, key):
    return {json.loads(l)[key] for l in open(path)} if os.path.exists(path) else set()


def progress(done, total, t0):
    el = time.time() - t0
    log(f"{done}/{total} ({100 * done / total:.0f}%), {el / 60:.1f} min, {done / el:.2f}/s, ETA {el / done * (total - done) / 60:.1f} min")


@app.function(volumes=VOLS, gpu="H100", timeout=6 * 3600)
def caption_clips(root: str, limit: int = 0, batch: int = 64):
    from vllm import LLM, SamplingParams

    base = f"/runs/{root}"
    out_path = f"{base}/captions.jsonl"
    files = sorted(os.path.relpath(os.path.join(d, f), base) for d, _, fs in os.walk(base) for f in fs if f.endswith(".ogg"))
    todo = [f for f in files if f not in done_keys(out_path, "file")]
    if limit:
        todo = todo[:limit]
    log(f"{len(files)} clips under {root}, {len(todo)} to caption")
    if not todo:
        return 0
    t0 = time.time()
    llm = LLM(model=CAPTIONER, max_model_len=8192, gpu_memory_utilization=0.92, limit_mm_per_prompt={"audio": 1}, max_num_seqs=batch)
    hf.commit()
    log(f"captioner loaded in {time.time() - t0:.0f}s")
    params = SamplingParams(temperature=0.0, max_tokens=600)
    t0 = time.time()
    with open(out_path, "a") as out:
        for b in range(0, len(todo), batch):
            chunk = todo[b:b + batch]
            convs = [[{"role": "user", "content": [{"type": "audio_url", "audio_url": {
                "url": "data:audio/ogg;base64," + base64.b64encode(open(f"{base}/{f}", "rb").read()).decode()}}]}] for f in chunk]
            for f, r in zip(chunk, llm.chat(convs, params, use_tqdm=False)):
                out.write(json.dumps({"file": f, "caption": r.outputs[0].text, "model": CAPTIONER}) + "\n")
            out.flush()
            runs.commit()
            progress(b + len(chunk), len(todo), t0)
    return len(todo)


@app.function(volumes=VOLS, gpu="H100", timeout=4 * 3600)
def judge_units(root: str, batch: int = 128):
    from vllm import LLM, SamplingParams

    base = f"/runs/{root}"
    captions = {json.loads(l)["file"]: json.loads(l)["caption"] for l in open(f"{base}/captions.jsonl")}
    groups = defaultdict(list)  # (layer folder, kind, unit, dose) -> [(plan, untouched caption, pushed caption)]
    for f, cap in captions.items():
        m = re.match(r"(?:(L\d+)/)?clips/([^/]+)/(neuron|random)(\d+)_([-+][\d.]+)\.ogg$", f)
        if m:
            layer, plan, kind, unit, dose = m.groups()
            untouched = captions.get(f"{layer + '/' if layer else ''}clips/{plan}/untouched.ogg")
            if untouched:
                groups[(layer or "", kind, int(unit), float(dose))].append((plan, untouched, cap))
    out_path = f"{base}/labels.jsonl"
    done = {(r["layer"], r["kind"], r["unit"], r["dose"]) for r in map(json.loads, open(out_path))} if os.path.exists(out_path) else set()
    todo = sorted(k for k, v in groups.items() if len(v) == 4 and k not in done)
    log(f"{len(groups)} unit-directions captioned, {len(todo)} to judge")
    if not todo:
        return 0
    t0 = time.time()
    llm = LLM(model=JUDGE, max_model_len=16384, gpu_memory_utilization=0.92, max_num_seqs=batch)
    hf.commit()
    log(f"judge loaded in {time.time() - t0:.0f}s")
    params = SamplingParams(temperature=0.0, max_tokens=500)

    def prompt(k):
        pairs = "\n\n".join(f"Song {n + 1} ({plan.replace('_', ' ')}):\nORIGINAL: {u}\nCHANGED: {p}"
                            for n, (plan, u, p) in enumerate(sorted(groups[k])))
        return [{"role": "user", "content": JUDGE_PROMPT.format(pairs=pairs)}]

    t0 = time.time()
    with open(out_path, "a") as out:
        for b in range(0, len(todo), batch):
            chunk = todo[b:b + batch]
            for (layer, kind, unit, dose), r in zip(chunk, llm.chat([prompt(k) for k in chunk], params, use_tqdm=False)):
                text = r.outputs[0].text
                try:
                    verdict = json.loads(text[text.index("{"):text.rindex("}") + 1])
                except ValueError:
                    verdict = {"consistency": "unparsed", "raw": text}
                out.write(json.dumps({"layer": layer, "kind": kind, "unit": unit, "dose": dose, **verdict, "judge": JUDGE}) + "\n")
            out.flush()
            runs.commit()
            progress(b + len(chunk), len(todo), t0)
    return len(todo)


@app.local_entrypoint()
def caption(root: str = "renderer_survey_d015", limit: int = 0):
    print(f"captioned {caption_clips.remote(root, limit)} clips", flush=True)


@app.local_entrypoint()
def judge(root: str = "renderer_survey_d015"):
    print(f"judged {judge_units.remote(root)} unit-directions", flush=True)
