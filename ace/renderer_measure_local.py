# /// script
# requires-python = ">=3.11"
# dependencies = ["librosa==0.10.2.post1", "soundfile", "numpy<2.2", "modal"]
# ///
"""Same sound measures as ace/renderer_measure.py, run on this computer instead of Modal (#6).

    uv run ace/renderer_measure_local.py runs/renderer_survey_d015/L0 runs/renderer_survey_d015/L3 ...

Each folder needs its index_*.jsonl files and clips/ (downloaded from the ace-neuron-runs volume). Writes
measures.jsonl into each folder, in the same format as the pilot, ready for ace/renderer_rank.py.
"""
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

from renderer_measure import MEASURES, measure


def measure_file(path):
    return path, measure(path)


def main():
    folders = sys.argv[1:]
    jobs = {}  # folder -> index rows
    for folder in folders:
        jobs[folder] = [json.loads(l) for f in sorted(glob.glob(f"{folder}/index_*.jsonl")) for l in open(f)]
    paths = sorted({os.path.join(folder, f) for folder, index in jobs.items()
                    for f in [r["file"] for r in index] + [f"clips/{r['plan']}/untouched.ogg" for r in index]})
    print(f"{len(paths)} clips in {len(folders)} folders", flush=True)
    results, t0 = {}, time.time()
    with ProcessPoolExecutor(os.cpu_count() - 2) as pool:
        for n, (path, m) in enumerate(pool.map(measure_file, paths, chunksize=8), 1):
            results[path] = m
            if n % 500 == 0 or n == len(paths):
                el = time.time() - t0
                print(f"{n}/{len(paths)} ({100 * n / len(paths):.0f}%), {el / 60:.1f} min, ETA {el / n * (len(paths) - n) / 60:.1f} min", flush=True)
    for folder, index in jobs.items():
        with open(os.path.join(folder, "measures.jsonl"), "w") as fh:
            for r in index:
                base, pushed = results[os.path.join(folder, "clips", r["plan"], "untouched.ogg")], results[os.path.join(folder, r["file"])]
                row = {k: r[k] for k in ("plan", "kind", "unit", "dose", "moved", "file")}
                fh.write(json.dumps({**row, **{m: pushed[m] - base[m] for m in MEASURES}}) + "\n")
        print(f"wrote {os.path.abspath(os.path.join(folder, 'measures.jsonl'))} ({len(index)} rows)", flush=True)


if __name__ == "__main__":
    main()
