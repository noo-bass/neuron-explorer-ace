"""Describe every clip of a renderer sweep with plain sound measures, pushed minus untouched (#6).

    modal run ace/renderer_measure.py --run renderer_sweep_L12
    modal volume get --force ace-neuron-runs renderer_sweep_L12/measures.jsonl runs/renderer_sweep_L12/

A provisional stand-in until #5 settles which describer to trust. Measures, each on the whole 10 s clip:
  loudness_db       overall level
  brightness        log2 of the spectral centroid (one unit = an octave brighter)
  onsets_per_s      note and drum attacks per second
  harmonic_share    share of energy in sustained, pitched sound (vs drum-like hits), from a harmonic/percussive split
  bass_db, mid_db, treble_db   level below 250 Hz, 250 Hz-4 kHz, above 4 kHz
  chroma_focus      how concentrated the pitch content is on a few notes (high for clear chords, low for noise)
"""
import json
import os
import time

import modal

image = modal.Image.from_registry("python:3.12-slim-bookworm").pip_install("librosa==0.10.2.post1", "soundfile", "numpy<2.2")
runs = modal.Volume.from_name("ace-neuron-runs")
app = modal.App("ace-renderer-measure", image=image)

MEASURES = ["loudness_db", "brightness", "onsets_per_s", "harmonic_share", "bass_db", "mid_db", "treble_db", "chroma_focus"]


def measure(path):
    import librosa
    import numpy as np

    y, sr = librosa.load(path, sr=22050, mono=True)
    db = lambda e: float(10 * np.log10(e + 1e-12))
    S = np.abs(librosa.stft(y)) ** 2
    f = librosa.fft_frequencies(sr=sr)
    H, P = librosa.decompose.hpss(S)
    chroma = librosa.feature.chroma_stft(S=S, sr=sr)
    return {
        "loudness_db": db(np.mean(y ** 2)),
        "brightness": float(np.log2(np.mean(librosa.feature.spectral_centroid(S=np.sqrt(S), sr=sr)) + 1e-9)),
        "onsets_per_s": len(librosa.onset.onset_detect(y=y, sr=sr)) / (len(y) / sr),
        "harmonic_share": float(H.sum() / (H.sum() + P.sum() + 1e-12)),
        "bass_db": db(S[f < 250].mean()),
        "mid_db": db(S[(f >= 250) & (f < 4000)].mean()),
        "treble_db": db(S[f >= 4000].mean()),
        "chroma_focus": float(np.mean(chroma.max(axis=0) / (chroma.sum(axis=0) + 1e-9))),
    }


@app.function(volumes={"/runs": runs}, cpu=2, timeout=3600)
def measure_files(root: str, files: list):
    out = []
    for f in files:
        out.append({"file": f, **measure(os.path.join(root, f))})
    return out


@app.local_entrypoint()
def main(run: str = "renderer_sweep_L12"):
    root = f"/runs/{run}"
    index = []
    for entry in runs.listdir(run):
        if entry.path.split("/")[-1].startswith("index_"):
            index += [json.loads(l) for l in b"".join(runs.read_file(entry.path)).decode().splitlines()]
    plans = sorted({r["plan"] for r in index})
    files = [f"clips/{p}/untouched.ogg" for p in plans] + [r["file"] for r in index]
    chunks = [files[i:i + 64] for i in range(0, len(files), 64)]
    print(f"{len(files)} clips in {len(chunks)} chunks", flush=True)
    t0, results = time.time(), {}
    for k, part in enumerate(measure_files.starmap([(root, c) for c in chunks])):
        results.update({r["file"]: r for r in part})
        if (k + 1) % 8 == 0 or k + 1 == len(chunks):
            print(f"{k + 1}/{len(chunks)} chunks, {time.time() - t0:.0f}s", flush=True)
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", run)
    os.makedirs(local, exist_ok=True)
    with open(os.path.join(local, "measures.jsonl"), "w") as fh:
        for r in index:
            base, pushed = results[f"clips/{r['plan']}/untouched.ogg"], results[r["file"]]
            fh.write(json.dumps({**r, **{m: pushed[m] - base[m] for m in MEASURES}}) + "\n")
    print(f"wrote {os.path.join(local, 'measures.jsonl')} ({len(index)} rows)", flush=True)
