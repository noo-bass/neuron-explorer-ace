# /// script
# requires-python = ">=3.11"
# dependencies = ["openai"]
# ///
"""Turn Qwen's per-song descriptions of a push into one verdict and label per push direction (#6).

For one unit (neuron or random direction) and one push direction, a text model reads the four "what changed"
descriptions, one per song, without being told whether the push came from a neuron or a random direction, and answers:
is it the same kind of change on every song, and if so, what is it in a few words.

    uv run ace/label_pushes.py runs/renderer_sweep_L12 neuron:5455 neuron:1967 random:0     # a few units
    uv run ace/label_pushes.py runs/renderer_sweep_L12                                       # every described unit
"""
import json
import os
import sys
from collections import defaultdict

from openai import OpenAI

from describe_change_qwen import BASE_URL, api_key

JUDGE = "qwen-plus"
PROMPT = """A music model generated four different songs. Each song was then regenerated with the same small internal \
change applied. A listener compared each original with its changed version and wrote what changed:

{descriptions}

Is it the same kind of change in all four songs? Judge the change itself, not the songs (different songs have \
different instruments, so "the lead melody disappears" can be the same change even if the melody is piano in one song \
and synth in another). Reply with JSON only:
{{"consistency": "same" | "mostly" | "different" | "none",   // "none" = little or no change in most songs
  "songs_sharing": <how many of the 4 songs show the shared change, 0-4>,
  "label": "<the shared change in at most 8 plain words, naming the actual sound the listener heard (e.g. 'low rumbling drone replaces the music', 'high melody line disappears') rather than a general category like 'musical replacement'; empty if there is none>",
  "destructive": <true if the shared change mostly wrecks the music (noise, garbling, unrecognisable), else false>,
  "reason": "<one sentence>"}}"""


def judge(client, descriptions):
    text = "\n\n".join(f"Song {k + 1} ({plan.replace('_', ' ')}):\n{d}" for k, (plan, d) in enumerate(descriptions))
    r = client.chat.completions.create(model=JUDGE, temperature=0, response_format={"type": "json_object"},
                                       messages=[{"role": "user", "content": PROMPT.format(descriptions=text)}])
    return json.loads(r.choices[0].message.content), r.usage.model_dump()


def main():
    run, wanted = sys.argv[1], [tuple(a.split(":")) for a in sys.argv[2:]]
    wanted = {(k, int(i)) for k, i in wanted}
    groups = defaultdict(list)
    for d in map(json.loads, open(os.path.join(run, "descriptions.jsonl"))):
        if not wanted or (d["kind"], d["unit"]) in wanted:
            groups[(d["kind"], d["unit"], d["dose"])].append((d["plan"], d["description"]))
    client = OpenAI(api_key=api_key(), base_url=BASE_URL)
    out = open(os.path.join(run, "labels.jsonl"), "a")
    for n, ((kind, unit, dose), descs) in enumerate(sorted(groups.items()), 1):
        verdict, usage = judge(client, sorted(descs))
        row = {"kind": kind, "unit": unit, "dose": dose, **verdict, "judge": JUDGE, "usage": usage}
        out.write(json.dumps(row) + "\n")
        out.flush()
        print(f"[{n}/{len(groups)}] {kind} {unit} {dose:+g}: {verdict['consistency']}, {verdict['songs_sharing']}/4, "
              f"\"{verdict['label']}\"{' (destructive)' if verdict.get('destructive') else ''} -- {verdict['reason']}", flush=True)


if __name__ == "__main__":
    main()
