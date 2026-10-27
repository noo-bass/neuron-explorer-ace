# /// script
# requires-python = ">=3.11"
# dependencies = ["openai"]
# ///
"""Describe in words what each push in a renderer sweep changed, with Qwen3.5-Omni (#6, method from #5).

    uv run ace/describe_sweep.py runs/renderer_sweep_L12 --slice      # 20 calls, to read the cost
    uv run ace/describe_sweep.py runs/renderer_sweep_L12              # every standout neuron + 32 random pushes

Which pushes: every neuron that stands out from the random pushes on at least one measure (ranking.json), and the
first 32 random pushes as a check on how vividly Qwen describes a meaningless change. Each on all songs, both
directions. One line per call in descriptions.jsonl; already-described clips are skipped, so it can be rerun.
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from describe_change_qwen import BASE_URL, COMPARE, MODEL, api_key, audio_part

N_RANDOM = 32
WORKERS = 8


def pushes(run):
    ranking = json.load(open(os.path.join(run, "ranking.json")))
    r0 = next(iter(ranking.values()))
    standouts = sorted({u["unit"] for r in ranking.values() for u in r["units"]
                        if u["kind"] == "neuron" and abs(u["z"]) > r["max_random_z"]})
    units = [("neuron", i) for i in standouts] + [("random", i) for i in range(N_RANDOM)]
    return [(p, kind, i, d) for kind, i in units for p in r0["plans"] for d in (0.3, -0.3)]


def describe(client, run, job, tries=6):
    """Retry when Model Studio says too many requests (429), waiting longer each time."""
    for attempt in range(tries):
        try:
            return describe_once(client, run, job)
        except Exception as e:
            if "429" not in str(e) or attempt == tries - 1:
                raise
            time.sleep(5 * 2 ** attempt)


def describe_once(client, run, job):
    plan, kind, i, dose = job
    untouched, pushed = os.path.join(run, "clips", plan, "untouched.ogg"), os.path.join(run, "clips", plan, f"{kind}{i}_{dose:+g}.ogg")
    text, usage, t0 = "", None, time.time()
    for chunk in client.chat.completions.create(
            model=MODEL, stream=True, stream_options={"include_usage": True}, modalities=["text"], temperature=0.3,
            messages=[{"role": "user", "content": [audio_part(untouched), audio_part(pushed), {"type": "text", "text": COMPARE}]}]):
        if chunk.choices:
            text += chunk.choices[0].delta.content or ""
        if getattr(chunk, "usage", None):
            usage = chunk.usage.model_dump()
    return {"plan": plan, "kind": kind, "unit": i, "dose": dose, "description": text, "usage": usage,
            "seconds": round(time.time() - t0, 1), "model": MODEL}


def main():
    run = sys.argv[1]
    jobs = pushes(run)
    if "--slice" in sys.argv:
        jobs = [j for j in jobs if (j[1], j[2]) in {("neuron", 5455), ("neuron", 1544)}] + [j for j in jobs if j[1:3] == ("random", 0)][:4]
    out_path = os.path.join(run, "descriptions.jsonl")
    done = set()
    if os.path.exists(out_path):
        done = {(r["plan"], r["kind"], r["unit"], r["dose"]) for r in map(json.loads, open(out_path))}
    todo = [j for j in jobs if j not in done]
    print(f"{len(jobs)} pushes, {len(jobs) - len(todo)} already described, {len(todo)} to go", flush=True)
    client = OpenAI(api_key=api_key(), base_url=BASE_URL)
    tokens, failed, t0 = {"prompt_tokens": 0, "completion_tokens": 0}, 0, time.time()
    with open(out_path, "a") as out, ThreadPoolExecutor(WORKERS) as pool:
        futures = {pool.submit(describe, client, run, j): j for j in todo}
        for n, f in enumerate(as_completed(futures), 1):
            try:
                r = f.result()
            except Exception as e:  # one failed call should not stop the run; rerunning picks it up
                failed += 1
                print(f"failed {futures[f]}: {e}", flush=True)
                continue
            out.write(json.dumps(r) + "\n")
            out.flush()
            for k in tokens:
                tokens[k] += (r["usage"] or {}).get(k, 0)
            if n % 10 == 0 or n == len(todo):
                el = time.time() - t0
                print(f"{n}/{len(todo)} ({100 * n / len(todo):.0f}%), {el / 60:.1f} min, ETA {el / n * (len(todo) - n) / 60:.1f} min, "
                      f"tokens in {tokens['prompt_tokens']:,} out {tokens['completion_tokens']:,}, failed {failed}", flush=True)


if __name__ == "__main__":
    main()
