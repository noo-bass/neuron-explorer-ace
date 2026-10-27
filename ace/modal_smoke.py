"""Smoke tests on a Modal GPU for the planner sweep (#2) and the renderer pilot (#4).

    modal run ace/modal_smoke.py::download          # once: ACE-Step 1.5 main checkpoints into the volume
    modal run ace/modal_smoke.py::planner           # speed of ACE's own runs vs batched runs, on one layer
    modal run ace/modal_smoke.py::renderer          # one fixed plan, rendered with renderer neurons pushed

ACE is installed from its own lock file at the commit used locally (ca1e85f), Python 3.12. Outputs land in the
`ace-neuron-runs` volume; fetch them with `modal volume get ace-neuron-runs <dir> runs/`.
"""
import json
import os
import subprocess
import sys
import time

import modal

ACE_COMMIT = "ca1e85f"
HERE = os.path.dirname(os.path.abspath(__file__))
GPU = "L40S"

image = (
    # the workspace's debian_slim is Debian 11 with a stale package index, so start from the current Python image
    modal.Image.from_registry("python:3.12-slim-bookworm")
    .run_commands("apt-get update && apt-get install -y --no-install-recommends git ffmpeg libsndfile1 build-essential")
    .pip_install("uv")
    .run_commands(
        f"git clone https://github.com/ace-step/ACE-Step-1.5.git /ace && cd /ace && git checkout {ACE_COMMIT}",
        "cd /ace && uv export --frozen --no-dev --no-hashes --no-emit-project -o /tmp/req.txt",
        "cd /ace && uv pip install --system --index-strategy unsafe-best-match "
        "--extra-index-url https://download.pytorch.org/whl/cu128 -r /tmp/req.txt",
        "cd /ace && uv pip install --system --no-deps -e .",
    )
    .env({"PYTHONUNBUFFERED": "1", "ACE_ROOT": "/ace"})
    .add_local_dir(HERE, "/root/nea_ace", ignore=["reports", "__pycache__"])
)
ckpt = modal.Volume.from_name("ace-checkpoints", create_if_missing=True)
runs = modal.Volume.from_name("ace-neuron-runs", create_if_missing=True)
app = modal.App("ace-neuron-smoke", image=image)
VOLS = {"/ckpt": ckpt, "/runs": runs}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def link_checkpoints():
    if not os.path.islink("/ace/checkpoints"):
        os.symlink("/ckpt", "/ace/checkpoints")


@app.function(volumes=VOLS, timeout=3600)
def download():
    from huggingface_hub import snapshot_download

    log("downloading ACE-Step/Ace-Step1.5 into the volume")
    snapshot_download("ACE-Step/Ace-Step1.5", local_dir="/ckpt")
    ckpt.commit()
    log("done:", sorted(os.listdir("/ckpt")))


@app.function(volumes=VOLS, gpu=GPU, timeout=3600)
def planner(layer: int = 14):
    """(1) ACE's own sequential runs on the GPU, via ace/steer_planner.py. (2) The same push done batched with plain
    HF generate (no constrained decoding), to see how much batching buys and whether the notes keep their shape."""
    import torch

    link_checkpoints()
    out = f"/runs/planner_smoke_L{layer}"
    os.makedirs(out, exist_ok=True)

    log("(1) ACE sequential: 2 neurons + 2 random directions, doses ±0.1 ±0.2, two captions")
    subprocess.run([sys.executable, "/root/nea_ace/steer_planner.py", "--layer", str(layer), "--device", "cuda",
                    "--neurons", "2", "--controls", "2", "--doses", "-0.2", "-0.1", "0.1", "0.2",
                    "--out", f"{out}/sequential.jsonl"], check=True)

    log("(2) batched HF generate with per-row pushes")
    from acestep.llm_inference import LLMHandler

    lm = LLMHandler()
    status, ok = lm.initialize(checkpoint_dir="/ckpt", lm_model_path="acestep-5Hz-lm-1.7B", backend="pt",
                               device="cuda", dtype=torch.bfloat16)
    assert ok, status
    model, tok = lm.llm, lm.llm_tokenizer
    tok.padding_side = "left"
    mlp = model.model.layers[layer].mlp
    W = mlp.down_proj.weight
    caption = "a song"
    prompt = lm.build_formatted_prompt(caption, "[Instrumental]")
    with torch.no_grad():
        ids = tok(prompt, return_tensors="pt").input_ids.cuda()
        resid = float(model(ids, output_hidden_states=True).hidden_states[layer + 1][0, 1:].float().norm(dim=-1).median())
    stop = [tok.convert_tokens_to_ids("</think>"), tok.eos_token_id]
    rows = None

    def hook(_m, _i, o):
        return o if rows is None else o + rows[:, None, :].to(o.dtype)

    mlp.register_forward_hook(hook)

    def generate(pushes):
        nonlocal rows
        rows = pushes
        enc = tok([prompt] * len(pushes), return_tensors="pt", padding=True).to("cuda")
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=400, do_sample=False, eos_token_id=stop, pad_token_id=tok.pad_token_id)
        torch.cuda.synchronize()
        dt = time.time() - t0
        rows = None
        new = g[:, enc.input_ids.shape[1]:]
        texts = tok.batch_decode(new, skip_special_tokens=False)
        n_tok = int((new != tok.pad_token_id).sum())
        return texts, dt, n_tok

    unpushed, _, _ = generate(torch.zeros(1, W.shape[0], device="cuda"))
    ace_meta = lm.generate_with_stop_condition(caption=caption, lyrics="[Instrumental]", infer_type="dit",
                                               temperature=0.0, use_constrained_decoding=True)["metadata"]
    log("unconstrained HF, unpushed:", unpushed[0][:600].replace("\n", " | "))
    log("ACE constrained, unpushed:", json.dumps(ace_meta)[:600])

    speed = []
    g = torch.Generator().manual_seed(0)
    for b in (1, 16, 64, 128, 256):
        idx = torch.randint(0, W.shape[1], (b,), generator=g)
        sign = torch.where(torch.rand(b, generator=g) < 0.5, -1.0, 1.0)
        d = W[:, idx.cuda()].float().T
        d = d / d.norm(dim=1, keepdim=True) * (0.15 * resid) * sign.cuda()[:, None]
        texts, dt, n_tok = generate(d)
        shaped = sum(("bpm:" in t and "caption:" in t) for t in texts)
        speed.append({"batch": b, "seconds": dt, "runs_per_s": b / dt, "tok_per_s": n_tok / dt, "shaped": shaped})
        log(f"batch {b:4}: {dt:6.1f}s  {b / dt:6.1f} runs/s  {n_tok / dt:7.0f} tok/s  notes well-formed {shaped}/{b}")
        if b == 64:
            json.dump(texts, open(f"{out}/batched_examples.json", "w"), ensure_ascii=False, indent=1)
    json.dump({"layer": layer, "resid_median": resid, "speed": speed, "unconstrained_unpushed": unpushed[0],
               "ace_unpushed": ace_meta}, open(f"{out}/batched.json", "w"), ensure_ascii=False, indent=1)
    runs.commit()


@app.function(volumes=VOLS, gpu=GPU, timeout=3600)
def renderer(layer: int = 12, seconds: float = 10.0):
    """Plan once, then render that fixed plan with one renderer neuron (or random direction) pushed: across all
    denoising steps, early steps only, or late steps only. Records time per clip and how far each clip moved."""
    import numpy as np
    import soundfile as sf
    import torch
    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music
    from acestep.llm_inference import LLMHandler

    link_checkpoints()
    out = f"/runs/renderer_smoke_L{layer}"
    os.makedirs(out, exist_ok=True)
    dit = AceStepHandler()
    log(dit.initialize_service(project_root="/ace", config_path="acestep-v15-turbo", device="cuda")[0][:200])
    lm = LLMHandler()
    log(lm.initialize(checkpoint_dir="/ckpt", lm_model_path="acestep-5Hz-lm-1.7B", backend="pt", device="cuda")[0][:200])
    cfg = GenerationConfig(batch_size=1, audio_format="wav", use_random_seed=False)

    t0 = time.time()
    plan = generate_music(dit, lm, GenerationParams(caption="warm lo-fi hip hop with dusty piano and soft drums",
                                                    lyrics="[Instrumental]", duration=seconds, seed=1), cfg,
                          save_dir=f"{out}/plan")
    assert plan.success, plan.error
    p = plan.audios[0]["params"]
    log(f"plan + first render {time.time() - t0:.1f}s; bpm {p.get('cot_bpm')} key {p.get('cot_keyscale')}")
    fixed = dict(caption=p.get("cot_caption") or p.get("caption"), lyrics="[Instrumental]", duration=seconds, seed=1,
                 bpm=p.get("cot_bpm"), keyscale=p.get("cot_keyscale") or "", timesignature=str(p.get("cot_timesignature") or ""),
                 audio_codes=p["audio_codes"], thinking=False, use_cot_metas=False, use_cot_caption=False,
                 use_cot_language=False)
    json.dump({k: v for k, v in fixed.items() if k != "audio_codes"}, open(f"{out}/plan.json", "w"), indent=1)

    dec = dit.model.decoder
    blk = dec.layers[layer]
    W = blk.mlp.down_proj.weight
    state = {"step": -1, "push": None, "steps": None, "norms": [], "ff": []}

    def count(_m, _a):
        state["step"] += 1

    def push(_m, _i, o):
        state["ff"].append(float(o[0].float().norm(dim=-1).median()))
        if state["push"] is not None and (state["steps"] is None or state["step"] in state["steps"]):
            return o + state["push"].to(o.dtype)
        return o

    def resid(_m, _i, o):
        state["norms"].append(float(o[0][0].float().norm(dim=-1).median()))

    dec.register_forward_pre_hook(count)
    blk.mlp.register_forward_hook(push)
    blk.register_forward_hook(resid)

    def render(name, vec=None, steps=None):
        state.update(step=-1, push=vec, steps=steps, norms=[], ff=[])
        t0 = time.time()
        r = generate_music(dit, lm, GenerationParams(**fixed), cfg, save_dir=f"{out}/{name}")
        assert r.success, r.error
        path = r.audios[0]["path"]
        final = f"{out}/{name}.wav"
        os.replace(path, final)
        return final, time.time() - t0, state["step"] + 1

    base, dt, n_steps = render("untouched")
    base_resid, base_ff = float(np.median(state["norms"])), float(np.median(state["ff"]))
    again, _, _ = render("untouched_again")
    a, sr = sf.read(base)
    b, _ = sf.read(again)
    log(f"untouched render {dt:.1f}s, {n_steps} decoder calls; repeat max |diff| {np.abs(a - b).max():.2e}")
    log(f"L{layer} median residual norm {base_resid:.1f}, median MLP-output norm {base_ff:.1f}")

    def moved(path):
        x, _ = sf.read(path)
        n = min(len(x), len(a))
        return float(np.sqrt(np.mean((x[:n] - a[:n]) ** 2)) / np.sqrt(np.mean(a[:n] ** 2)))

    half = n_steps // 2
    windows = {"all": None, "early": set(range(half)), "late": set(range(half, n_steps))}
    g = torch.Generator().manual_seed(0)
    units = [("neuron", int(n)) for n in torch.randperm(W.shape[1], generator=g)[:2]] + [("random", 0), ("random", 1)]
    results = []
    for kind, i in units:
        d = W[:, i].float() if kind == "neuron" else torch.randn(W.shape[0], generator=torch.Generator().manual_seed(10_000 + i)).to(W.device)
        d = d / d.norm()
        for dose in (0.3, 1.0):
            for wname, steps in windows.items():
                name = f"{kind}{i}_{dose}_{wname}"
                path, dt, _ = render(name, dose * base_resid * d, steps)
                m = moved(path)
                results.append({"kind": kind, "unit": i, "dose": dose, "window": wname, "seconds": dt, "moved": m, "file": os.path.basename(path)})
                log(f"{name:28} {dt:5.1f}s  moved {m:.2f} (RMS of difference / RMS of untouched)")
    json.dump({"layer": layer, "steps": n_steps, "resid_median": base_resid, "ff_median": base_ff, "results": results},
              open(f"{out}/results.json", "w"), indent=1)
    runs.commit()
