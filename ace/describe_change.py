# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["numpy", "soundfile", "librosa", "torch", "torchaudio", "transformers", "demucs"]
# ///
"""Experiment for issue #5: describe what a renderer push changed in the sound, pushed clip vs untouched clip.

Three candidates, cheapest first, all scored as "pushed minus untouched" on the same clip:
  signal  plain measures: loudness, brightness (spectral centroid), onsets per second, tempo
  tagger  AudioSet tagger (AST, MIT/ast-finetuned-audioset-10-10-0.4593): change in each label's probability
  stems   Demucs 6-stem model (htdemucs_6s: drums, bass, other, vocals, guitar, piano): change in each stem's loudness

    uv run ace/describe_change.py runs/renderer_smoke_L12
Writes describe_change.json into the run directory.
"""
import json
import os
import sys
import time

import librosa
import numpy as np
import soundfile as sf
import torch

TAGS = ["Music", "Piano", "Electric piano", "Keyboard (musical)", "Drum kit", "Drum", "Snare drum", "Hi-hat",
        "Bass guitar", "Guitar", "Synthesizer", "Singing", "Hip hop music", "Silence", "Noise"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def db(x):
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-9))


def signal_measures(mono, sr):
    onsets = librosa.onset.onset_detect(y=mono, sr=sr, units="time")
    tempo = librosa.feature.tempo(y=mono, sr=sr)[0]
    return {"loudness_db": db(mono),
            "brightness_hz": float(np.mean(librosa.feature.spectral_centroid(y=mono, sr=sr))),
            "onsets_per_s": len(onsets) / (len(mono) / sr),
            "tempo_bpm": float(tempo)}


class Tagger:
    def __init__(self):
        from transformers import ASTFeatureExtractor, ASTForAudioClassification
        name = "MIT/ast-finetuned-audioset-10-10-0.4593"
        self.fe = ASTFeatureExtractor.from_pretrained(name)
        self.model = ASTForAudioClassification.from_pretrained(name).eval()
        self.labels = self.model.config.id2label

    @torch.no_grad()
    def __call__(self, mono, sr):
        x = librosa.resample(mono, orig_sr=sr, target_sr=16000)
        logits = self.model(**self.fe(x, sampling_rate=16000, return_tensors="pt")).logits[0]
        return torch.sigmoid(logits).numpy()


class Stems:
    def __init__(self):
        from demucs.pretrained import get_model
        self.model = get_model("htdemucs_6s").eval()
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model.to(self.device)

    @torch.no_grad()
    def __call__(self, stereo, sr):
        from demucs.apply import apply_model
        x = librosa.resample(stereo.T, orig_sr=sr, target_sr=self.model.samplerate)
        # shifts=0: Demucs' default shifts the audio by a random offset first, so identical clips got different stems
        out = apply_model(self.model, torch.tensor(x, dtype=torch.float32)[None].to(self.device), shifts=0,
                          device=self.device)[0]
        return {name: db(out[i].cpu().numpy()) for i, name in enumerate(self.model.sources)}


def main(run_dir):
    files = sorted(f for f in os.listdir(run_dir) if f.endswith(".wav"))
    log(f"{len(files)} clips in {run_dir}; loading tagger and stem model")
    tagger, stems = Tagger(), Stems()
    raw = {}
    for i, f in enumerate(files):
        stereo, sr = sf.read(os.path.join(run_dir, f), dtype="float32")
        mono = stereo.mean(axis=1)
        raw[f] = {"signal": signal_measures(mono, sr), "tags": tagger(mono, sr), "stems": stems(stereo, sr)}
        log(f"[{i + 1}/{len(files)}] {f}")

    base = raw["untouched.wav"]
    lab = tagger.labels
    out = {"tags_tracked": TAGS, "untouched": {"signal": base["signal"], "stems": base["stems"],
           "tags": {t: float(base["tags"][tagger.model.config.label2id[t]]) for t in TAGS}}, "clips": {}}
    for f, r in raw.items():
        d_tags = r["tags"] - base["tags"]
        movers = np.argsort(-np.abs(d_tags))[:5]
        out["clips"][f] = {
            "signal": {k: r["signal"][k] - base["signal"][k] for k in r["signal"]},
            "tags": {t: float(d_tags[tagger.model.config.label2id[t]]) for t in TAGS},
            "tag_movers": [[lab[int(j)], float(d_tags[j])] for j in movers],
            "stems": {k: r["stems"][k] - base["stems"][k] for k in r["stems"]},
        }
    path = os.path.join(run_dir, "describe_change.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    log(f"wrote {os.path.abspath(path)}")


if __name__ == "__main__":
    main(sys.argv[1])
