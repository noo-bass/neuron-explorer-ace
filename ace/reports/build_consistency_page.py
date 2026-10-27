"""Listening page for the N5455 consistency run: python ace/reports/build_consistency_page.py
Reads runs/renderer_n5455_consistency/results.json, writes ace/reports/renderer_n5455_consistency.html."""
import html
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = "renderer_n5455_consistency"
CLIPS = f"../../runs/{RUN}"
data = json.load(open(os.path.join(HERE, "..", "..", "runs", RUN, "results.json")))
style = open(os.path.join(HERE, "renderer_smoke_L12.html")).read().split("<style>")[1].split("</style>")[0]
COLS = [(0.3, "all"), (0.3, "early"), (0.15, "all"), (0.15, "early"), (-0.3, "all"), (-0.3, "early")]
WIN = {"all": "all 8 passes", "early": "first 4 passes"}
moved = {(r["plan"], r["unit"], r["dose"], r["window"]): r for r in data["results"]}


def unit_label(u):
    return (f"Neuron {u[6:]}" if u.startswith("neuron") else f"Random {int(u[6:]) + 1}"), ("neuron" if u.startswith("neuron") else "random")


def audio(src):
    return f'<audio controls preload="none" src="{CLIPS}/{src}"></audio>'


units = list(dict.fromkeys(r["unit"] for r in data["results"]))
sections = []
for plan, p in data["plans"].items():
    head = "".join(f"<th>Push {d:+g}<small>{WIN[w]}</small></th>" for d, w in COLS)
    rows = []
    for u in units:
        label, cls = unit_label(u)
        star = ' style="background:var(--accent-soft)"' if u == "neuron5455" else ""
        cells = "".join(
            f'<td>{audio(r["file"])}<span class="m">moved {r["moved"]:.2f}</span></td>'
            for r in [moved[(plan, u, d, w)] for d, w in COLS])
        rows.append(f'<tr{star}><th class="{cls}">{label}</th>{cells}</tr>')
    sections.append(f"""<section class="block"><h2>{html.escape(plan.replace('_', ' '))}</h2>
<div class="base"><div class="eyebrow">Untouched</div>{audio(plan + '__untouched.wav')}
<dl><div><dt>asked for</dt><dd>{html.escape(p['asked'])}</dd></div><div><dt>planner wrote</dt><dd>{html.escape(p['caption'] or '')}</dd></div>
<div><dt>tempo</dt><dd>{p['bpm']}</dd></div><div><dt>key</dt><dd>{html.escape(p['keyscale'])}</dd></div></dl></div>
<div class="tablebox"><table style="min-width:1400px"><thead><tr><th></th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>""")

page = f"""<title>Piano Neuron Check</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Instrument+Serif&family=Instrument+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">
<style>{style}</style>
<div class="wrap">
<header style="display:grid;gap:12px">
  <div class="eyebrow">ACE-Step 1.5 · renderer · layer {data['layer']} of 24 · neuron 5455 consistency</div>
  <h1>Does neuron 5455 remove the piano everywhere?</h1>
  <p class="muted">Six songs, each planned once, then rendered again with one push at a time. Neuron 5455 (highlighted) sits beside another neuron and four random directions of exactly the same size, so you can hear whether 5455 does something the others don't.</p>
</header>
<section style="display:grid;gap:8px"><h2>What to listen for</h2><ul>
  <li><b>Songs with piano</b> (lo-fi, jazz trio, ballad, solo piano): does the piano drop out with 5455, and not with the random rows?</li>
  <li><b>Songs without piano</b> (rock band, ambient synth): does 5455 remove some other instrument, or nothing in particular?</li>
  <li><b>Push −0.3</b> pushes the opposite way. If 5455 is a piano switch, this might make the piano louder or add one.</li>
  <li><b>Push 0.15</b> is half the size of the push you heard before, to see where the effect starts.</li>
  <li>"moved" is how different the clip is from the untouched one, as a fraction of its volume. It measures size, not what changed.</li>
</ul></section>
{''.join(sections)}
<p class="muted">Clips live in runs/{RUN}/ next to this repo (not in git). Made by ace/renderer_consistency.py on a Modal L40S.</p>
</div>"""
open(os.path.join(HERE, f"{RUN}.html"), "w").write(page)
print(os.path.join(HERE, f"{RUN}.html"))
