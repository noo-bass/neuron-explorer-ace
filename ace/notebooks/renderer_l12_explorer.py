# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "polars", "altair"]
# ///
"""Explore the renderer layer-12 pilot (#6): which neurons stand outside the random pushes, and how they sound.

    uvx marimo edit --sandbox ace/notebooks/renderer_l12_explorer.py
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl

    return Path, alt, json, mo, pl


@app.cell
def _(mo):
    mo.md("""
    # Renderer layer 12: neurons vs random pushes

    256 neurons and 256 random pushes of the same size, each tried on four songs, pushed +0.3 and -0.3.
    For each sound measure, a neuron "stands out" when its average change across the four songs is further from
    the random average than **every one** of the 256 random pushes. Every dot in the chart is real data.

    Pick a measure, then click a dot in the chart or a row in a table to hear that neuron's before and after.
    Further down: the blind labels, which ask whether each push makes the same change on every song.
    """)
    return


@app.cell
def _(mo):
    run_picker = mo.ui.radio({"push size 0.3": "renderer_sweep_L12", "push size 0.15 (gentler)": "renderer_sweep_L12_d015"},
                             value="push size 0.15 (gentler)", label="Which pilot", inline=True)
    run_picker
    return (run_picker,)


@app.cell
def _(Path, json, mo, pl, run_picker):
    RUN = Path(mo.notebook_dir()).parents[1] / "runs" / run_picker.value
    ranking = json.loads((RUN / "ranking.json").read_text())
    PLAIN = {"moved": "overall change", "loudness_db": "loudness (dB)", "brightness": "brightness (octaves)",
             "onsets_per_s": "attacks per second", "harmonic_share": "sustained share", "bass_db": "bass (dB)",
             "mid_db": "middle (dB)", "treble_db": "treble (dB)", "chroma_focus": "note focus"}
    PLANS = next(iter(ranking.values()))["plans"]
    DOSES = sorted({r["dose"] for r in ranking.values()}, reverse=True)
    scores = pl.DataFrame([
        {"measure": PLAIN[r["measure"]], "dose": r["dose"], "kind": u["kind"], "unit": u["unit"], "z": u["z"],
         "average change": u["mean"], "songs agreeing": u["agree"], "edge": r["max_random_z"],
         "stands out": u["kind"] == "neuron" and abs(u["z"]) > r["max_random_z"]}
        for r in ranking.values() for u in r["units"]
    ])
    # Qwen3.5-Omni's answer to "what changed between the untouched and the pushed clip?" (ace/describe_sweep.py)
    _path = RUN / "descriptions.jsonl"
    described = {(d["plan"], d["kind"], d["unit"], d["dose"]): d["description"]
                 for d in map(json.loads, _path.read_text().splitlines())} if _path.exists() else {}
    # A text model's blind verdict on each unit and direction, from the four descriptions (ace/label_pushes.py)
    _path = RUN / "labels.jsonl"
    labels = pl.DataFrame([
        {"kind": d["kind"], "unit": d["unit"], "dose": d["dose"], "consistency": d["consistency"],
         "songs sharing": d["songs_sharing"], "label": d["label"], "destructive": bool(d["destructive"]), "why": d["reason"]}
        for d in map(json.loads, _path.read_text().splitlines())
    ]) if _path.exists() else None
    return DOSES, PLAIN, PLANS, RUN, described, labels, scores


@app.cell
def _(DOSES, PLAIN, mo):
    measure = mo.ui.dropdown(list(PLAIN.values()), value="note focus", label="Measure")
    dose = mo.ui.radio({f"push {d:+g}": d for d in DOSES}, value=f"push {DOSES[0]:+g}", label="Direction", inline=True)
    mo.hstack([measure, dose], justify="start", gap=2)
    return dose, measure


@app.cell
def _(alt, dose, measure, mo, pl, scores):
    view = scores.filter((pl.col("measure") == measure.value) & (pl.col("dose") == dose.value))
    edge = view["edge"][0]
    n_out = view.filter(pl.col("stands out")).height
    dots = alt.Chart(view).mark_circle(size=55, opacity=0.75).encode(
        x=alt.X("z:Q", title="distance from the random average (in random-spreads)"),
        y=alt.Y("kind:N", title=None, sort=["random", "neuron"]),
        yOffset="jitter:Q",
        color=alt.Color("stands out:N", scale=alt.Scale(domain=[False, True], range=["#9aa5a0", "#1f6f62"]), title="outside every random push"),
        tooltip=["kind", "unit", alt.Tooltip("z:Q", format="+.1f"), alt.Tooltip("average change:Q", format="+.3f"), "songs agreeing"],
    ).transform_calculate(jitter="random()")
    chart = mo.ui.altair_chart(dots.properties(height=220, width="container"))
    mo.vstack([mo.md(f"**{n_out} of 256 neurons** (green) fall further from the random average than every random push, which all sit within {edge:.1f} random-spreads of it. Hover for numbers; click a dot to listen."), chart])
    return (chart,)


@app.cell
def _(mo, pl, scores):
    standouts = (
        scores.filter(pl.col("stands out"))
        .group_by("unit")
        .agg(pl.len().alias("measures"),
             pl.col("z").abs().max().round(1).alias("strongest"),
             pl.concat_str([pl.col("measure"), pl.lit(" "), pl.when(pl.col("z") > 0).then(pl.lit("up")).otherwise(pl.lit("down")),
                            pl.lit(" at "), pl.col("dose").cast(pl.Utf8)]).alias("stands out on"))
        .with_columns(pl.col("stands out on").list.join(", "))
        .sort("strongest", descending=True)
        .rename({"unit": "neuron"})
    )
    table = mo.ui.table(standouts, selection="single", page_size=10, label=f"All {standouts.height} neurons that stand out on at least one measure")
    table
    return (table,)


@app.cell
def _(mo):
    mo.md("""
    ## Blind labels: is it the same change on every song?

    For each neuron or random push, and each direction, a text model read Qwen's four "what changed" descriptions
    (one per song) without being told whether the push came from a neuron or a random direction. It judged whether
    the change was the same across songs and named it.

    **At push 0.3** random pushes come out consistent as often as neurons, nearly always as "the main melody disappears
    or is replaced": any push that size knocks the melody out. **At push 0.15** random pushes mostly change nothing,
    while some neurons still make one consistent, specific change (reverb, vinyl crackle, an added pad or strings).
    That gap is the evidence that those neurons do something of their own.
    """)
    return


@app.cell
def _(alt, labels, mo, pl):
    mo.stop(labels is None, mo.md("*No labels yet.*"))
    _order = ["same", "mostly", "different", "none"]
    _rates = (labels.group_by("kind", "consistency").agg(pl.len().alias("n"))
              .with_columns((pl.col("n") / pl.col("n").sum().over("kind")).alias("share")))
    _bars = alt.Chart(_rates).mark_bar().encode(
        y=alt.Y("kind:N", title=None, sort=["neuron", "random"]),
        x=alt.X("share:Q", title="share of pushes", axis=alt.Axis(format="%")),
        color=alt.Color("consistency:N", sort=_order, scale=alt.Scale(domain=_order, range=["#1f6f62", "#6fb5a5", "#d8c48a", "#b9bfbc"]),
                        title="same change across songs?"),
        order=alt.Order("rank:Q"),
        tooltip=["kind", "consistency", "n", alt.Tooltip("share:Q", format=".0%")],
    ).transform_calculate(rank="indexof(['same','mostly','different','none'], datum.consistency)").properties(height=110, width="container")
    _dest = labels.filter(pl.col("consistency").is_in(["same", "mostly"])).group_by("kind").agg(pl.col("destructive").sum().alias("d"), pl.len().alias("n"))
    _d = {r["kind"]: f"{r['d']} of {r['n']}" for r in _dest.iter_rows(named=True)}
    mo.vstack([_bars, mo.md(f"Of the consistent changes, judged destructive: neurons {_d.get('neuron', '0')}, random pushes {_d.get('random', '0')}.")])
    return


@app.cell
def _(mo):
    label_filter = mo.ui.radio(["neurons, consistent and not the usual melody loss", "all neurons", "random pushes", "destructive"],
                               value="neurons, consistent and not the usual melody loss", label="Show", inline=True)
    label_filter
    return (label_filter,)


@app.cell
def _(label_filter, labels, mo, pl):
    mo.stop(labels is None)
    _consistent = pl.col("consistency").is_in(["same", "mostly"])
    _usual = pl.col("label").str.contains(r"(?i)melod|instrument (is )?replaced|instrument disappears")
    _views = {
        "neurons, consistent and not the usual melody loss": labels.filter((pl.col("kind") == "neuron") & _consistent & ~_usual),
        "all neurons": labels.filter(pl.col("kind") == "neuron"),
        "random pushes": labels.filter(pl.col("kind") == "random"),
        "destructive": labels.filter(pl.col("destructive")),
    }
    _shown = _views[label_filter.value].sort(["consistency", "songs sharing"], descending=[False, True])
    label_table = mo.ui.table(_shown, selection="single", page_size=12,
                              label=f"{_shown.height} pushes. Click a row to listen. 'Usual melody loss' = labels mentioning melody or a replaced/disappearing instrument, the random pushes' typical effect.")
    label_table
    return (label_table,)


@app.cell
def _(
    DOSES,
    PLANS,
    RUN,
    chart,
    described,
    dose,
    label_table,
    labels,
    measure,
    mo,
    pl,
    scores,
    table,
):
    picked_chart = chart.value
    if label_table.value.height:
        kind, unit = label_table.value["kind"][0], int(label_table.value["unit"][0])
    elif len(picked_chart) and "unit" in picked_chart.columns:
        kind, unit = picked_chart["kind"][0], int(picked_chart["unit"][0])
    elif table.value.height:
        kind, unit = "neuron", int(table.value["neuron"][0])
    else:
        kind, unit = "neuron", 5455
    row = scores.filter((pl.col("kind") == kind) & (pl.col("unit") == unit) & (pl.col("measure") == measure.value) & (pl.col("dose") == dose.value))

    def clip(path):
        return mo.audio((RUN / "clips" / path).read_bytes())

    def pushed(p, d):
        text = described.get((p, kind, unit, d))
        heard = mo.accordion({"What Qwen heard change": mo.md(text)}) if text else mo.md("*not described*")
        return mo.vstack([mo.md(f"push {d:+g}"), clip(f"{p}/{kind}{unit}_{d:+g}.ogg"), heard])

    songs = [mo.hstack([mo.md(f"**{p.replace('_', ' ')}**"), mo.vstack([mo.md("untouched"), clip(f"{p}/untouched.ogg")]),
                        *[pushed(p, d) for d in DOSES]], widths=[1, 2, 2, 2], align="start")
             for p in PLANS]
    title = f"Neuron {unit}" if kind == "neuron" else f"Random push {unit}"
    summary = f"{measure.value}, {'+' if dose.value > 0 else ''}{dose.value}: {row['z'][0]:+.1f} random-spreads from the random average, {row['songs agreeing'][0]} of 4 songs agree" if row.height else ""
    verdicts = []
    if labels is not None:
        for _r in labels.filter((pl.col("kind") == kind) & (pl.col("unit") == unit)).sort("dose", descending=True).iter_rows(named=True):
            verdicts.append(f"- **push {_r['dose']:+g}:** {_r['consistency']} ({_r['songs sharing']}/4 songs)"
                            + (f", \"{_r['label']}\"" if _r["label"] else "") + (" *(destructive)*" if _r["destructive"] else ""))
    mo.vstack([mo.md(f"## {title}\n{summary}\n\n" + "\n".join(verdicts)), *songs])
    return


if __name__ == "__main__":
    app.run()
