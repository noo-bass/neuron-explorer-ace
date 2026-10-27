"""Experiment for issue #5, candidate 3: ACE's own understand mode on rendered clips. Each wav is encoded back into
audio codes (VAE + the renderer's tokenizer, as ACE does for cover/repaint), then the planner LM writes a caption and
song notes for those codes, unsteered. Same seed for every clip, so caption differences come from the audio.

Run from this repo with ACE's environment (dose 0.3 clips + untouched only by default):
    uv run --project ../ACE-Step-1.5 python ace/describe_change_ace.py runs/renderer_smoke_L12 --match _0.3_ untouched
Writes describe_change_ace.json into the run directory.
"""
import argparse
import json
import os
import time

import torch

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler

ACE = os.path.abspath(os.environ.get("ACE_ROOT", os.path.join(os.path.dirname(__file__), "..", "..", "ACE-Step-1.5")))

ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("--match", nargs="*", default=[], help="keep clips whose name contains any of these")
args = ap.parse_args()

files = sorted(f for f in os.listdir(args.run_dir) if f.endswith(".wav") and (not args.match or any(m in f for m in args.match)))
t0 = time.time()
dit = AceStepHandler()
dit.initialize_service(project_root=ACE, config_path="acestep-v15-turbo", device="mps")
lm = LLMHandler()
print(lm.initialize(checkpoint_dir=os.path.join(ACE, "checkpoints"), lm_model_path="acestep-5Hz-lm-1.7B",
                    backend="pt", device="mps")[0], flush=True)
print(f"loaded in {time.time() - t0:.0f}s; {len(files)} clips", flush=True)

# keep the LM's raw reply, which understand_audio_from_codes parses and drops
_generate = lm.generate_from_formatted_prompt
replies = []
lm.generate_from_formatted_prompt = lambda *a, **k: replies.append(_generate(*a, **k)) or replies[-1]

out = {}
for i, f in enumerate(files):
    t1 = time.time()
    codes = dit.convert_src_audio_to_codes(os.path.join(args.run_dir, f))
    if not codes.startswith("<|audio_code_"):
        raise SystemExit(codes)
    torch.manual_seed(0)
    meta, status = lm.understand_audio_from_codes(codes, temperature=0.3)
    out[f] = {k: meta.get(k) for k in ("caption", "bpm", "keyscale", "timesignature")}
    out[f]["prompt"] = lm.build_formatted_prompt_for_understanding(codes)
    out[f]["reply"] = replies[-1][0]
    print(f"[{i + 1}/{len(files)}] {f} ({time.time() - t1:.0f}s): {out[f]['caption']}", flush=True)

path = os.path.join(args.run_dir, "describe_change_ace.json")
with open(path, "w") as fh:
    json.dump(out, fh, indent=1)
print(f"wrote {os.path.abspath(path)}", flush=True)
