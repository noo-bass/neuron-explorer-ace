"""Listening page for the top neurons of a renderer sweep (#6).

    python ace/reports/build_sweep_page.py files    # list the clips the page needs (to download)
    python ace/reports/build_sweep_page.py          # write ace/reports/renderer_sweep_L12_top.html

Picks the two strongest neurons on each measure and push direction, plus N5455 and two typical random pushes.
"""
import html
import json
import os
import sys
from statistics import median

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = "renderer_sweep_L12"
ranking = json.load(open(os.path.join(HERE, "..", "..", "runs", RUN, "ranking.json")))
PLAIN = {"moved": "overall change", "loudness_db": "loudness", "brightness": "brightness", "onsets_per_s": "attacks per second",
         "harmonic_share": "sustained share", "bass_db": "bass", "mid_db": "middle", "treble_db": "treble",
         "chroma_focus": "note focus"}

picked = {}  # (kind, unit) -> list of reasons
for key, r in ranking.items():
    for u in [u for u in r["units"] if u["kind"] == "neuron"][:2]:
        picked.setdefault(("neuron", u["unit"]), []).append(
            f"{PLAIN[r['measure']]} {'up' if u['z'] > 0 else 'down'} at push {r['dose']:+g} (z {u['z']:+.1f}, {u['agree']} of 4 songs agree)")
picked.setdefault(("neuron", 5455), []).append("heard removing the chordal part (reference)")
moved = [u for u in ranking["moved +0.3"]["units"] if u["kind"] == "random"]
mid = median(u["mean"] for u in moved)
for u in sorted(moved, key=lambda u: abs(u["mean"] - mid))[:2]:
    picked[("random", u["unit"])] = ["a typical random push, for comparison"]
plans = ranking["moved +0.3"]["plans"]
doses = (0.3, -0.3)
files = [f"clips/{p}/untouched.ogg" for p in plans] + [f"clips/{p}/{k}{i}_{d:+g}.ogg" for (k, i) in picked for p in plans for d in doses]

if sys.argv[1:] == ["files"]:
    print("\n".join(files))
    sys.exit()

style = open(os.path.join(HERE, "renderer_smoke_L12.html")).read().split("<style>")[1].split("</style>")[0]
audio = lambda f: f'<audio controls preload="none" src="../../runs/{RUN}/{f}"></audio>'
order = sorted(picked, key=lambda u: (u[0] == "random", u != ("neuron", 5455), -len(picked[u])))
sections = []
for kind, i in order:
    label = f"Neuron {i}" if kind == "neuron" else f"Random push {i}"
    reasons = "".join(f"<li>{html.escape(r)}</li>" for r in picked[(kind, i)])
    rows = "".join(f"<tr><th>{p.replace('_', ' ')}</th><td>{audio(f'clips/{p}/untouched.ogg')}</td>"
                   + "".join(f"<td>{audio(f'clips/{p}/{kind}{i}_{d:+g}.ogg')}</td>" for d in doses) + "</tr>" for p in plans)
    sections.append(f'<section class="block"><h2 class="{kind}">{label}</h2><ul>{reasons}</ul><div class="tablebox"><table>'
                    f'<thead><tr><th>song</th><th>untouched</th><th>push +0.3</th><th>push -0.3</th></tr></thead><tbody>{rows}</tbody></table></div></section>')

page = f"""<title>Renderer Top Neurons</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Instrument+Serif&family=Instrument+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">
<style>{style} h2.random {{ color:var(--rand); }}</style>
<div class="wrap">
<header style="display:grid;gap:12px">
  <div class="eyebrow">ACE-Step 1.5 · renderer · layer 12 · pilot of 256 neurons + 256 random pushes</div>
  <h1>The neurons that stood out</h1>
  <p class="muted">For each measure and push direction, the two neurons whose change was furthest from what random pushes do, across the same four songs. Neuron 5455 and two typical random pushes are included to compare against. The reasons listed come from crude sound measures; the question for your ears is whether each neuron does one recognisable thing on all four songs.</p>
</header>
{''.join(sections)}
<p class="muted">Clips live in runs/{RUN}/clips/ next to this repo (not in git); the full set is on the Modal volume ace-neuron-runs. Made by ace/renderer_sweep.py, measured by ace/renderer_measure.py, ranked by ace/renderer_rank.py.</p>
</div>"""
out = os.path.join(HERE, f"{RUN}_top.html")
open(out, "w").write(page)
print(out, f"({len(picked)} units)")
