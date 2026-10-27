"""Does renderer L12 N5455 remove the piano on other songs too, and do random pushes of the same size do the same? (#4)

    MODAL_IMAGE_BUILDER_VERSION=2025.06 modal run ace/renderer_consistency.py
    modal volume get --force ace-neuron-runs renderer_n5455_consistency runs/

Six plans (four with piano in the caption, two without), each planned once and then rendered with: N5455, one other
neuron (N2732, from the smoke test), and four random directions. Doses -0.3, 0.15, 0.3 (1.0 was destructive in the
smoke test), pushed in all 8 steps or the first 4 only. Same image, volumes and push recipe as ace/modal_smoke.py.
"""
import json
import os
import time

import modal

from modal_smoke import GPU, VOLS, image, link_checkpoints, log, runs

app = modal.App("ace-renderer-consistency", image=image.add_local_python_source("modal_smoke"))

LAYER = 12
SECONDS = 10.0
CAPTIONS = {
    "lofi_piano": "warm lo-fi hip hop with dusty piano and soft drums",
    "jazz_trio": "upbeat jazz piano trio with walking upright bass and brushed drums",
    "piano_ballad": "emotional pop ballad with grand piano, strings and soft female vocals humming",
    "solo_piano": "gentle solo classical piano, slow and reflective",
    "rock_band": "energetic rock band with distorted electric guitars, bass and loud drums",
    "synth_ambient": "dreamy ambient synth pads with slow evolving textures, no drums",
}
UNITS = [("neuron", 5455), ("neuron", 2732)] + [("random", i) for i in range(4)]
DOSES = (-0.3, 0.15, 0.3)


@app.function(volumes=VOLS, gpu=GPU, timeout=3600)
def run():
    import numpy as np
    import soundfile as sf
    import torch
    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music
    from acestep.llm_inference import LLMHandler

    link_checkpoints()
    out = "/runs/renderer_n5455_consistency"
    os.makedirs(out, exist_ok=True)
    dit = AceStepHandler()
    log(dit.initialize_service(project_root="/ace", config_path="acestep-v15-turbo", device="cuda")[0][:200])
    lm = LLMHandler()
    log(lm.initialize(checkpoint_dir="/ckpt", lm_model_path="acestep-5Hz-lm-1.7B", backend="pt", device="cuda")[0][:200])
    cfg = GenerationConfig(batch_size=1, audio_format="wav", use_random_seed=False)

    dec = dit.model.decoder
    blk = dec.layers[LAYER]
    W = blk.mlp.down_proj.weight
    state = {"step": -1, "push": None, "steps": None, "norms": []}

    def count(_m, _a):
        state["step"] += 1

    def push(_m, _i, o):
        if state["push"] is not None and (state["steps"] is None or state["step"] in state["steps"]):
            return o + state["push"].to(o.dtype)
        return o

    def resid(_m, _i, o):
        state["norms"].append(float(o[0][0].float().norm(dim=-1).median()))

    dec.register_forward_pre_hook(count)
    blk.mlp.register_forward_hook(push)
    blk.register_forward_hook(resid)

    def render(params, path, vec=None, steps=None):
        state.update(step=-1, push=vec, steps=steps, norms=[])
        r = generate_music(dit, lm, params, cfg, save_dir=f"{out}/tmp")
        assert r.success, r.error
        os.replace(r.audios[0]["path"], path)
        return r

    directions = {}
    for kind, i in UNITS:
        d = W[:, i].float() if kind == "neuron" else torch.randn(W.shape[0], generator=torch.Generator().manual_seed(10_000 + i)).to(W.device)
        directions[f"{kind}{i}"] = d / d.norm()

    total = len(CAPTIONS) * len(UNITS) * len(DOSES) * 2
    done, t_start, results, plans = 0, time.time(), [], {}
    for name, caption in CAPTIONS.items():
        r = render(GenerationParams(caption=caption, lyrics="[Instrumental]", duration=SECONDS, seed=1), f"{out}/{name}__planned.wav")
        p = r.audios[0]["params"]
        fixed = dict(caption=p.get("cot_caption") or p.get("caption"), lyrics="[Instrumental]", duration=SECONDS, seed=1,
                     bpm=p.get("cot_bpm"), keyscale=p.get("cot_keyscale") or "", timesignature=str(p.get("cot_timesignature") or ""),
                     audio_codes=p["audio_codes"], thinking=False, use_cot_metas=False, use_cot_caption=False,
                     use_cot_language=False)
        plans[name] = {"asked": caption, **{k: v for k, v in fixed.items() if k != "audio_codes"}}
        base_path = f"{out}/{name}__untouched.wav"
        render(GenerationParams(**fixed), base_path)
        n_steps, scale = state["step"] + 1, float(np.median(state["norms"]))
        a, _ = sf.read(base_path)
        log(f"[{name}] planned: bpm {fixed['bpm']} key {fixed['keyscale']}; {n_steps} steps; L{LAYER} residual norm {scale:.0f}")
        windows = {"all": None, "early": set(range(n_steps // 2))}
        for unit, d in directions.items():
            for dose in DOSES:
                for wname, steps in windows.items():
                    fname = f"{name}__{unit}_{dose:+g}_{wname}.wav"
                    render(GenerationParams(**fixed), f"{out}/{fname}", dose * scale * d, steps)
                    x, _ = sf.read(f"{out}/{fname}")
                    n = min(len(x), len(a))
                    moved = float(np.sqrt(np.mean((x[:n] - a[:n]) ** 2)) / np.sqrt(np.mean(a[:n] ** 2)))
                    results.append({"plan": name, "unit": unit, "dose": dose, "window": wname, "moved": moved, "file": fname})
                    done += 1
                    if done % 12 == 0 or done == total:
                        el = time.time() - t_start
                        log(f"{done}/{total} clips ({100 * done / total:.0f}%), {el / 60:.1f} min, ETA {el / done * (total - done) / 60:.1f} min")
        json.dump({"layer": LAYER, "doses": DOSES, "plans": plans, "results": results}, open(f"{out}/results.json", "w"), indent=1)
        runs.commit()
    log("done")
