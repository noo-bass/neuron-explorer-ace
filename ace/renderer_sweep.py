"""Push every neuron in one renderer layer, beside random directions, on a few fixed plans (#6).

    MODAL_IMAGE_BUILDER_VERSION=2025.06 modal run ace/renderer_sweep.py::check     # batched renders match single ones?

Many pushes are rendered at once: one row per push, every row with the same seed and plan, and the hook adds a
different push to each row. Push recipe as in ace/modal_smoke.py: dose x median residual norm x unit direction,
at the output of decoder layer 12's MLP, in all 8 steps.
"""
import json
import os
import time

import modal

from modal_smoke import GPU, VOLS, image, link_checkpoints, log, runs

app = modal.App("ace-renderer-sweep", image=image.add_local_python_source("modal_smoke"))

LAYER = 12
SECONDS = 10.0
PLANS = {
    "lofi_piano": "warm lo-fi hip hop with dusty piano and soft drums",
    "piano_ballad": "emotional pop ballad with grand piano, strings and soft female vocals humming",
    "rock_band": "energetic rock band with distorted electric guitars, bass and loud drums",
    "synth_ambient": "dreamy ambient synth pads with slow evolving textures, no drums",
}


class Renderer:
    """ACE's turbo renderer with a per-row push on one decoder layer."""

    def __init__(self, layer=LAYER):
        import torch
        from acestep.handler import AceStepHandler
        from acestep.llm_inference import LLMHandler

        link_checkpoints()
        self.dit = AceStepHandler()
        log(self.dit.initialize_service(project_root="/ace", config_path="acestep-v15-turbo", device="cuda")[0][:200])
        # ACE shrinks big batches to protect small GPUs from running out of memory; on the L40S we pick the batch size
        self.dit._vram_guard_reduce_batch = lambda batch_size, **_: batch_size
        self.lm = LLMHandler()
        log(self.lm.initialize(checkpoint_dir="/ckpt", lm_model_path="acestep-5Hz-lm-1.7B", backend="pt", device="cuda")[0][:200])
        blk = self.dit.model.decoder.layers[layer]
        self.W = blk.mlp.down_proj.weight
        self.rows = None  # [batch, hidden] pushes, or None
        self.norms = []

        def push(_m, _i, o):
            if self.rows is None:
                return o
            assert o.shape[0] == self.rows.shape[0], (o.shape, self.rows.shape)
            return o + self.rows[:, None, :].to(o.dtype)

        def resid(_m, _i, o):
            self.norms.append(float(o[0][0].float().norm(dim=-1).median()))

        blk.mlp.register_forward_hook(push)
        blk.register_forward_hook(resid)
        self.torch = torch

    def direction(self, kind, i):
        torch = self.torch
        d = self.W[:, i].float() if kind == "neuron" else torch.randn(self.W.shape[0], generator=torch.Generator().manual_seed(10_000 + i)).to(self.W.device)
        return d / d.norm()

    def plan(self, caption):
        """Plan once with the planner, keep its audio codes, and measure the layer's residual norm on the untouched render."""
        from acestep.inference import GenerationConfig, GenerationParams, generate_music

        r = generate_music(self.dit, self.lm, GenerationParams(caption=caption, lyrics="[Instrumental]", duration=SECONDS, seed=1),
                           GenerationConfig(batch_size=1, audio_format="wav", use_random_seed=False))
        assert r.success, r.error
        p = r.audios[0]["params"]
        fixed = dict(caption=p.get("cot_caption") or p.get("caption"), lyrics="[Instrumental]", duration=SECONDS, seed=1,
                     bpm=p.get("cot_bpm"), keyscale=p.get("cot_keyscale") or "", timesignature=str(p.get("cot_timesignature") or ""),
                     audio_codes=p["audio_codes"], thinking=False, use_cot_metas=False, use_cot_caption=False,
                     use_cot_language=False)
        self.render(fixed, None)
        return {"asked": caption, "params": fixed, "resid_median": float(sorted(self.norms)[len(self.norms) // 2])}

    def render(self, plan, rows):
        """Render one plan once per row of `rows` ([batch, hidden] pushes; None for a single untouched render).
        Returns the audio arrays (samples x channels) in row order, and the sample rate."""
        from acestep.inference import GenerationConfig, GenerationParams, generate_music

        fixed = plan["params"] if "params" in plan else plan
        n = 1 if rows is None else rows.shape[0]
        self.rows, self.norms = rows, []
        try:
            r = generate_music(self.dit, self.lm, GenerationParams(**fixed),
                               GenerationConfig(batch_size=n, seeds=[fixed["seed"]] * n, audio_format="wav", use_random_seed=False))
        finally:
            self.rows = None
        assert r.success, r.error
        assert len(r.audios) == n, len(r.audios)
        out = [a["tensor"].T.numpy() for a in r.audios]
        sr = r.audios[0]["sample_rate"]
        return out, sr


@app.function(volumes=VOLS, gpu=GPU, timeout=3600)
def check():
    """Does a batched render give the same audio as single renders, row by row? And how fast is it per clip?"""
    import numpy as np

    out = "/runs/renderer_sweep_check"
    os.makedirs(out, exist_ok=True)
    R = Renderer()
    p = R.plan(PLANS["lofi_piano"])
    scale = 0.3 * p["resid_median"]
    log(f"plan: bpm {p['params']['bpm']} key {p['params']['keyscale']}; residual norm {p['resid_median']:.0f}")
    torch = R.torch
    push = scale * R.direction("neuron", 5455)

    (single0,), _ = R.render(p, None)
    (single1,), _ = R.render(p, push[None])

    def diff(a, b):
        n = min(len(a), len(b))
        return float(np.abs(a[:n] - b[:n]).max()), float(np.sqrt(np.mean((a[:n] - b[:n]) ** 2)) / np.sqrt(np.mean(b[:n] ** 2)))

    log(f"single untouched vs single pushed: max {diff(single0, single1)[0]:.3f}, relative {diff(single0, single1)[1]:.3f}")
    # ACE breaks above 8 rows (conditioning_masks.py indexes past a list of 8), so batches are 8 rows.
    # Is a row's audio independent of what the other rows in its batch are doing?
    def batch(first, rest_seed):
        rows = torch.zeros(8, R.W.shape[0], device=R.W.device)
        rows[1] = first
        for k in range(2, 8):
            rows[k] = scale * R.direction("random", rest_seed + k)
        torch.cuda.synchronize()
        t0 = time.time()
        clips, _ = R.render(p, rows)
        return clips, time.time() - t0

    a_clips, dt = batch(push, 0)
    b_clips, _ = batch(push, 100)
    c_clips, _ = batch(torch.zeros_like(push), 200)
    report = {"seconds_per_batch_of_8": dt,
              "untouched_row_A_vs_B": diff(a_clips[0], b_clips[0]),
              "untouched_row_vs_untouched_row_same_batch": diff(c_clips[0], c_clips[1]),
              "same_push_A_vs_B": diff(a_clips[1], b_clips[1]),
              "batched_untouched_vs_single": diff(a_clips[0], single0),
              "push_vs_untouched_in_batch": diff(a_clips[1], a_clips[0]),
              "random_push_vs_untouched_in_batch": diff(a_clips[2], a_clips[0])}
    for k, v in report.items():
        log(f"{k:45} {v}")
    json.dump({"plan": {**p, "params": {k: v for k, v in p["params"].items() if k != "audio_codes"}}, "report": report}, open(f"{out}/check.json", "w"), indent=1)
    runs.commit()
