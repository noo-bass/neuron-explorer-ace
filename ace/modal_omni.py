"""Describe what a push changed with an open Qwen3-Omni model on a Modal GPU, instead of the paid Alibaba API (#6).

    modal run ace/modal_omni.py::describe --run renderer_sweep_L12_d015 --model instruct --limit 40
    modal run ace/modal_omni.py::describe --run renderer_sweep_L12_d015 --model captioner --limit 40

Two ways, to be scored against the Alibaba (Qwen3.5-Omni-Plus) descriptions we already have for the same pushes:
  instruct   Qwen3-Omni-30B-A3B-Instruct hears the untouched clip and the pushed clip together and says what changed
             (same prompt as ace/describe_change_qwen.py)
  captioner  Qwen3-Omni-30B-A3B-Captioner describes each clip on its own; the comparison is left to the text judge
The pushes are the ones in the run's local descriptions.jsonl, so both sets cover exactly the same clips.
Writes descriptions_<model>.jsonl (instruct) or captions_captioner.jsonl (captioner) into the local run folder.
"""
import base64
import json
import os
import time

import modal

MODELS = {"instruct": "Qwen/Qwen3-Omni-30B-A3B-Instruct", "captioner": "Qwen/Qwen3-Omni-30B-A3B-Captioner"}
COMPARE = ("The first audio clip is the original. The second is the same piece after a change. Describe precisely what "
           "changed between them: which instruments or sounds appeared, disappeared, got louder or quieter, or changed "
           "character. Say what stayed the same too.")

image = (
    modal.Image.from_registry("python:3.12-slim-bookworm")
    .run_commands("apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1")
    .pip_install("vllm[audio]==0.31.0", "huggingface_hub[hf_transfer]")
    .env({"PYTHONUNBUFFERED": "1", "HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/hf"})
)
hf = modal.Volume.from_name("hf-cache", create_if_missing=True)
runs = modal.Volume.from_name("ace-neuron-runs")
app = modal.App("ace-omni", image=image)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def audio(path):
    data = base64.b64encode(open(path, "rb").read()).decode()
    return {"type": "audio_url", "audio_url": {"url": f"data:audio/ogg;base64,{data}"}}


@app.function(volumes={"/hf": hf, "/runs": runs}, gpu="H100", timeout=4 * 3600)
def run_model(model: str, run: str, jobs: list, batch: int = 64):
    from vllm import LLM, SamplingParams

    t0 = time.time()
    llm = LLM(model=MODELS[model], max_model_len=8192, gpu_memory_utilization=0.92, limit_mm_per_prompt={"audio": 2},
              max_num_seqs=batch)
    hf.commit()
    log(f"{MODELS[model]} loaded in {time.time() - t0:.0f}s")
    # captioner at temperature 0: the same clip always gets the same caption, so wording differences between two
    # captions come from the audio, not from chance
    params = SamplingParams(temperature=0.3 if model == "instruct" else 0.0, max_tokens=600)
    root = f"/runs/{run}/clips"
    if model == "instruct":
        convs = [[{"role": "user", "content": [audio(f"{root}/{p}/untouched.ogg"), audio(f"{root}/{p}/{k}{u}_{d:+g}.ogg"),
                                               {"type": "text", "text": COMPARE}]}] for p, k, u, d in jobs]
    else:  # the captioner takes audio only, no instruction
        convs = [[{"role": "user", "content": [audio(f"{root}/{f}")]}] for f in jobs]
    out, t0 = [], time.time()
    for b in range(0, len(convs), batch):
        res = llm.chat(convs[b:b + batch], params, use_tqdm=False)
        out += [r.outputs[0].text for r in res]
        done = len(out)
        el = time.time() - t0
        log(f"{done}/{len(convs)} ({100 * done / len(convs):.0f}%), {el / 60:.1f} min, {done / el:.2f}/s, "
            f"ETA {el / done * (len(convs) - done) / 60:.1f} min")
    return out


@app.local_entrypoint()
def describe(run: str = "renderer_sweep_L12_d015", model: str = "instruct", limit: int = 0):
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", run)
    ref = [json.loads(l) for l in open(os.path.join(local, "descriptions.jsonl"))]
    jobs = sorted({(r["plan"], r["kind"], r["unit"], r["dose"]) for r in ref})
    if limit:
        jobs = jobs[:limit]
    if model == "instruct":
        texts = run_model.remote(model, run, jobs)
        with open(os.path.join(local, "descriptions_instruct.jsonl"), "w") as f:
            for (p, k, u, d), t in zip(jobs, texts):
                f.write(json.dumps({"plan": p, "kind": k, "unit": u, "dose": d, "description": t, "model": MODELS[model]}) + "\n")
    else:
        files = sorted({f"{p}/untouched.ogg" for p, *_ in jobs} | {f"{p}/{k}{u}_{d:+g}.ogg" for p, k, u, d in jobs})
        texts = run_model.remote(model, run, files)
        with open(os.path.join(local, "captions_captioner.jsonl"), "w") as f:
            for name, t in zip(files, texts):
                f.write(json.dumps({"file": name, "caption": t, "model": MODELS[model]}) + "\n")
    print(f"wrote {len(texts)} results", flush=True)
