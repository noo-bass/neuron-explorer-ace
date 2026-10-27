"""Smoke test for steering the ACE-Step planner (issue #1): push single MLP neurons and random directions of the same
size at one layer, and compare the planner's phase-1 notes (bpm, key, caption, ...) with the unsteered notes.

The push goes into ACE's own PyTorch planner (LLMHandler backend "pt"), so prompt format, constrained decoding and
CFG are ACE's as shipped (ACE turns CFG off for phase 1 itself, so these runs are single-pass). It is added at the layer's MLP output, which equals "add to neuron n's activation" because
Qwen3 has no FFN layer scale: dose x median residual norm leaving the layer (tokens 1:), along the unit direction of
W_down[:, n] or of a random vector. In phase 2 the CFG batch is [cond..., uncond...] and the hook pushes every row.

Run from this repo with ACE's environment:
    uv run --project ../ACE-Step-1.5 python ace/steer_planner.py --layer 14
"""
import argparse
import json
import os
import time

import torch

from acestep.llm_inference import LLMHandler

ACE = os.path.abspath(os.environ.get("ACE_ROOT", os.path.join(os.path.dirname(__file__), "..", "..", "ACE-Step-1.5")))
RUNS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "runs"))

CAPTIONS = ["warm lo-fi hip hop with dusty piano and soft drums", "a song"]
# extra captions only for measuring the residual norm
NORM_CAPTIONS = CAPTIONS + ["aggressive thrash metal with fast double kick", "solo cello, slow and mournful",
                            "bright 80s synth pop with gated reverb drums", "ambient drone, field recordings",
                            "bossa nova guitar and brushed snare", "epic orchestral trailer music with choir"]

ap = argparse.ArgumentParser()
ap.add_argument("--layer", type=int, default=14)
ap.add_argument("--neurons", type=int, default=4, help="random neurons sampled from the layer")
ap.add_argument("--controls", type=int, default=4, help="random directions of the same size")
ap.add_argument("--doses", type=float, nargs="+", default=[-0.4, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.4])
ap.add_argument("--max-new-tokens", type=int, default=400, help="notes are ~150 tokens; ACE's own cap is 4096")
ap.add_argument("--cfg", type=float, default=2.0, help="phase 2 only; ACE forces 1.0 for phase 1")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=os.path.join(RUNS, "ace_smoke.jsonl"))
args = ap.parse_args()

lm = LLMHandler()
status, ok = lm.initialize(checkpoint_dir=os.path.join(ACE, "checkpoints"), lm_model_path="acestep-5Hz-lm-1.7B",
                           backend="pt", device="mps")
if not ok:
    raise SystemExit(status)
model, tok = lm.llm, lm.llm_tokenizer
model.config.max_new_tokens = args.max_new_tokens  # read by ACE as the phase-1 cap when no duration is given
layers = model.model.layers
mlp = layers[args.layer].mlp
W = mlp.down_proj.weight  # [hidden, intermediate]
print(f"planner: {len(layers)} layers, {W.shape[1]} neurons/layer, hidden {W.shape[0]}, dtype {W.dtype}", flush=True)

# median residual norm leaving the layer, first token (the sink) excluded
norms = []
with torch.no_grad():
    for c in NORM_CAPTIONS:
        ids = tok(lm.build_formatted_prompt(c, "[Instrumental]"), return_tensors="pt").input_ids.to(model.device)
        hs = model(ids, output_hidden_states=True).hidden_states[args.layer + 1][0, 1:]
        norms.append(hs.float().norm(dim=-1).cpu())
resid = float(torch.cat(norms).median())
print(f"median residual norm leaving L{args.layer}: {resid:.1f} over {sum(len(n) for n in norms)} tokens", flush=True)

push = None  # vector added to the MLP output at every position, or None


def hook(_module, _inp, out):
    return out if push is None else out + push.to(out.dtype)


mlp.register_forward_hook(hook)

g = torch.Generator().manual_seed(args.seed)
units = [("neuron", int(n)) for n in torch.randperm(W.shape[1], generator=g)[:args.neurons]]
units += [("random", i) for i in range(args.controls)]


def direction(kind, i):
    if kind == "neuron":
        d = W[:, i].float()
    else:
        d = torch.randn(W.shape[0], generator=torch.Generator().manual_seed(10_000 + i)).to(W.device)
    return d / d.norm()


def notes(caption):
    t0 = time.time()
    r = lm.generate_with_stop_condition(caption=caption, lyrics="[Instrumental]", infer_type="dit", temperature=0.0,
                                        cfg_scale=args.cfg, use_constrained_decoding=True)
    if not r["success"]:
        raise RuntimeError(r["error"])
    meta = {k: v for k, v in r["metadata"].items() if isinstance(v, (str, int, float))}
    return meta, time.time() - t0


def distinct4(text):
    """Share of distinct word 4-grams in the caption: ~1 for normal prose, low for loops (Lyra's breakdown measure)."""
    w = str(text).split()
    grams = [tuple(w[j:j + 4]) for j in range(len(w) - 3)]
    return len(set(grams)) / len(grams) if grams else 0.0


os.makedirs(RUNS, exist_ok=True)
runs = seconds = 0
rows = []
with open(args.out, "w") as f:
    for caption in CAPTIONS:
        push = None
        base, _ = notes(caption)
        again, _ = notes(caption)
        print(f"\n[{caption}] baseline ({'repeatable' if again == base else 'NOT REPEATABLE'}): {base}", flush=True)
        f.write(json.dumps({"caption": caption, "kind": "baseline", "dose": 0, "meta": base}) + "\n")
        for kind, i in units:
            for dose in args.doses:
                push = dose * resid * direction(kind, i)
                meta, dt = notes(caption)
                runs, seconds = runs + 1, seconds + dt
                changed = {k: v for k, v in meta.items() if base.get(k) != v}
                d4 = distinct4(meta.get("caption", ""))
                rows.append((kind, dose, d4, bool(changed)))
                print(f"  {kind:6} {i:5} {dose:+.2f} {dt:5.1f}s d4={d4:.2f} changed: {str(changed or '-')[:300]}", flush=True)
                f.write(json.dumps({"caption": caption, "kind": kind, "unit": i, "dose": dose, "meta": meta}) + "\n")
push = None
print("\ndose   | neurons: d4  changed | random: d4  changed")
for dose in args.doses:
    cell = []
    for kind in ("neuron", "random"):
        r = [x for x in rows if x[0] == kind and x[1] == dose]
        cell.append(f"{sum(x[2] for x in r) / len(r):.2f}  {sum(x[3] for x in r)}/{len(r)}")
    print(f"{dose:+.2f}  | {cell[0]:>16} | {cell[1]:>15}")
print(f"\n{seconds / runs:.1f} s per steered run (one caption, single pass, MPS); {args.out}", flush=True)
