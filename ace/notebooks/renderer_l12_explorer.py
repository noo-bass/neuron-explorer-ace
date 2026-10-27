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

    Pick a measure, then click a dot in the chart or a row in the table to hear that neuron's before and after.
    """)
    return


@app.cell
def _(Path, json, mo, pl):
    RUN = Path(mo.notebook_dir()).parents[1] / "runs" / "renderer_sweep_L12"
    ranking = json.loads((RUN / "ranking.json").read_text())
    PLAIN = {"moved": "overall change", "loudness_db": "loudness (dB)", "brightness": "brightness (octaves)",
             "onsets_per_s": "attacks per second", "harmonic_share": "sustained share", "bass_db": "bass (dB)",
             "mid_db": "middle (dB)", "treble_db": "treble (dB)", "chroma_focus": "note focus"}
    PLANS = ranking["moved +0.3"]["plans"]
    scores = pl.DataFrame([
        {"measure": PLAIN[r["measure"]], "dose": r["dose"], "kind": u["kind"], "unit": u["unit"], "z": u["z"],
         "average change": u["mean"], "songs agreeing": u["agree"], "edge": r["max_random_z"],
         "stands out": u["kind"] == "neuron" and abs(u["z"]) > r["max_random_z"]}
        for r in ranking.values() for u in r["units"]
    ])
    return PLAIN, PLANS, RUN, scores


@app.cell
def _(PLAIN, mo):
    measure = mo.ui.dropdown(list(PLAIN.values()), value="note focus", label="Measure")
    dose = mo.ui.radio({"push +0.3": 0.3, "push -0.3": -0.3}, value="push +0.3", label="Direction", inline=True)
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
def _(PLANS, RUN, chart, dose, measure, mo, pl, scores, table):
    picked_chart = chart.value
    if len(picked_chart) and "unit" in picked_chart.columns:
        kind, unit = picked_chart["kind"][0], int(picked_chart["unit"][0])
    elif table.value.height:
        kind, unit = "neuron", int(table.value["neuron"][0])
    else:
        kind, unit = "neuron", 5455
    row = scores.filter((pl.col("kind") == kind) & (pl.col("unit") == unit) & (pl.col("measure") == measure.value) & (pl.col("dose") == dose.value))

    def clip(path):
        return mo.audio((RUN / "clips" / path).read_bytes())

    songs = [mo.hstack([mo.md(f"**{p.replace('_', ' ')}**"), mo.vstack([mo.md("untouched"), clip(f"{p}/untouched.ogg")]),
                        mo.vstack([mo.md("push +0.3"), clip(f"{p}/{kind}{unit}_+0.3.ogg")]),
                        mo.vstack([mo.md("push -0.3"), clip(f"{p}/{kind}{unit}_-0.3.ogg")])], widths=[1, 2, 2, 2], align="center")
             for p in PLANS]
    title = f"Neuron {unit}" if kind == "neuron" else f"Random push {unit}"
    summary = f"{measure.value}, {'+' if dose.value > 0 else ''}{dose.value}: {row['z'][0]:+.1f} random-spreads from the random average, {row['songs agreeing'][0]} of 4 songs agree" if row.height else ""
    mo.vstack([mo.md(f"## {title}\n{summary}"), *songs])
    return


if __name__ == "__main__":
    app.run()
