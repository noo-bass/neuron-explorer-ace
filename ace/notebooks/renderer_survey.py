# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "polars", "altair"]
# ///
"""Survey of all 24 renderer layers (#6): which layers have neurons that do something a random push does not.

    uvx marimo edit --sandbox ace/notebooks/renderer_survey.py
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from collections import defaultdict
    from pathlib import Path
    from statistics import mean, pstdev

    import altair as alt
    import marimo as mo
    import polars as pl

    return Path, alt, defaultdict, json, mean, mo, pl, pstdev


@app.cell
def _(mo):
    mo.md("""
    # Renderer survey: all 24 layers

    In every layer of the renderer, 32 neurons picked at random and 32 random directions were each pushed by 0.15
    (the gentle size from the layer-12 pilot), both ways, on the same four songs. Each clip is compared with its
    untouched song on eight plain sound measures (loudness, brightness, attacks per second, sustained share, bass,
    middle, treble, note focus).

    **The test.** A push "stands out" on a measure when its average change over the four songs is further from the
    random average than every random push, and at least three of the four songs move the same way. The catch: with
    only 31 random pushes to compare against, some pushes pass by luck. So the same test is run on each random
    push, against the other 31. If neurons pass no more often than random pushes, that layer shows nothing special.
    """)
    return


@app.cell
def _(Path, defaultdict, json, mean, mo, pstdev):
    ROOT = Path(mo.notebook_dir()).parents[1] / "runs" / "renderer_survey_d015"
    MEASURES = {"loudness_db": "loudness", "brightness": "brightness", "onsets_per_s": "attacks per second",
                "harmonic_share": "sustained share", "bass_db": "bass", "mid_db": "middle", "treble_db": "treble",
                "chroma_focus": "note focus"}

    def standouts(rows):
        """Every (kind, unit, measure, dose) that stands out, each judged against 31 random pushes (never itself)."""
        by = defaultdict(dict)
        for r in rows:
            by[(r["kind"], r["unit"], r["dose"])][r["plan"]] = r
        found = []
        for dose in sorted({r["dose"] for r in rows}, reverse=True):
            for m in MEASURES:
                changes = {(k, u): [p[m] for p in pp.values()] for (k, u, d), pp in by.items() if d == dose}
                rand = {u: mean(c) for (k, u), c in changes.items() if k == "random"}
                for (k, u), c in changes.items():
                    ref = [v for ru, v in rand.items() if not (k == "random" and ru == u)][:31]
                    mu, sd = mean(ref), pstdev(ref)
                    edge, s = max(abs(v - mu) / sd for v in ref), mean(c)
                    if abs(s - mu) / sd > edge and sum((x > 0) == (s > 0) for x in c) >= 3:
                        found.append({"kind": k, "unit": u, "measure": MEASURES[m], "dose": dose,
                                      "z": round((s - mu) / sd, 1), "change": s})
        return found

    measures = {L: [json.loads(l) for l in (ROOT / f"L{L}" / "measures.jsonl").read_text().splitlines()] for L in range(24)}
    found = {L: standouts(rows) for L, rows in measures.items()}
    return ROOT, found, measures


@app.cell
def _(alt, found, mo, pl):
    per_layer = pl.DataFrame([
        {"layer": L, "kind": kind, "units standing out": len({f["unit"] for f in fs if f["kind"] == kind})}
        for L, fs in found.items() for kind in ("neuron", "random")
    ])
    _wide = per_layer.pivot(on="kind", index="layer", values="units standing out")
    _bars = alt.Chart(per_layer).mark_bar().encode(
        x=alt.X("layer:O", title="renderer layer (0 = first, 23 = last)"),
        xOffset=alt.XOffset("kind:N", sort=["neuron", "random"]),
        y=alt.Y("units standing out:Q", title="of 32 that stand out on at least one measure"),
        color=alt.Color("kind:N", sort=["neuron", "random"], scale=alt.Scale(range=["#1f6f62", "#b9bfbc"]), title=None),
        tooltip=["layer", "kind", "units standing out"],
    ).properties(height=260, width="container")
    mo.vstack([
        mo.md(f"**All layers:** {_wide['neuron'].sum()} neurons vs {_wide['random'].sum()} random pushes stand out "
              f"(of {32 * 24} each). Grey bars are what luck alone gives; green well above grey means that layer's "
              "neurons do something of their own."),
        _bars,
    ])
    return


@app.cell
def _(mean, measures, mo, pl):
    moved = pl.DataFrame([
        {"layer": L, "neuron": mean(r["moved"] for r in rows if r["kind"] == "neuron"),
         "random": mean(r["moved"] for r in rows if r["kind"] == "random")}
        for L, rows in measures.items()
    ]).with_columns((pl.col("neuron") / pl.col("random")).round(2).alias("neuron / random"))
    mo.vstack([mo.md("""
    **How far the push moves the model's inner state**, measured at the end of the renderer ("moved", from the
    sweep itself), averaged over all pushes. A ratio near 1 means a neuron push disturbs the model about as much as a
    random one of the same size.
    """), moved.with_columns(pl.col("neuron").round(2), pl.col("random").round(2))])
    return


@app.cell
def _(ROOT, json, mo, pl):
    _pilot = ROOT.parent / "renderer_sweep_L12_d015" / "labels.jsonl"
    _old = {(r["kind"], r["unit"], r["dose"]): r["consistency"] for r in map(json.loads, _pilot.read_text().splitlines())}
    _new = {(r["kind"], r["unit"], r["dose"]): r["consistency"] for r in map(json.loads, (ROOT / "labels.jsonl").read_text().splitlines())
            if r["layer"] == "L12"}
    _both = sorted(set(_old) & set(_new))
    calibration = (pl.DataFrame([{"before and after heard together": _old[k], "each clip captioned on its own": _new[k]} for k in _both])
                   .group_by(pl.all()).len().sort("len", descending=True))
    mo.vstack([mo.md(f"""
    ## Why not the word labels for this survey

    Nine layers were also captioned (Qwen3-Omni Captioner, each clip described on its own) and judged by a text model.
    It called random pushes "the same change on every song" as often as neurons, in every layer. The check below is
    the {len(_both)} layer-12 random pushes, which are the very same clips the pilot judged with Qwen hearing the before and
    after together. Where the pilot mostly said "none" (no change), the caption method mostly says "same": the captioner
    words the same audio differently each time (it describes a buzz at the end of 38 of the 40 untouched clips too), and
    the judge reads the different wording as a change.
    """), calibration])
    return


@app.cell
def _(mo):
    layer = mo.ui.slider(0, 23, value=23, label="Layer", show_value=True)
    layer
    return (layer,)


@app.cell
def _(found, layer, mo, pl):
    _f = pl.DataFrame([f for f in found[layer.value] if f["kind"] == "neuron"])
    layer_table = mo.ui.table(
        _f.group_by("unit").agg(
            pl.len().alias("measures"), pl.col("z").abs().max().alias("strongest"),
            pl.concat_str([pl.col("measure"), pl.lit(" "), pl.when(pl.col("z") > 0).then(pl.lit("up")).otherwise(pl.lit("down")),
                           pl.lit(" at "), pl.col("dose").cast(pl.Utf8)]).alias("stands out on"),
        ).with_columns(pl.col("stands out on").list.join(", ")).sort("strongest", descending=True).rename({"unit": "neuron"}),
        selection="single", page_size=10, label=f"Layer {layer.value}: neurons that stand out. Click one to listen.",
    ) if _f.height else mo.md("*No neuron stands out in this layer.*")
    layer_table
    return (layer_table,)


@app.cell
def _(ROOT, layer, layer_table, measures, mo):
    mo.stop(not hasattr(layer_table, "value") or not layer_table.value.height, mo.md("*Pick a neuron above to hear it.*"))
    _unit = int(layer_table.value["neuron"][0])
    _dir = ROOT / f"L{layer.value}" / "clips"
    _plans = sorted({r["plan"] for r in measures[layer.value]})

    def _clip(path):
        return mo.audio((_dir / path).read_bytes())

    mo.vstack([mo.md(f"## Layer {layer.value}, neuron {_unit}")] + [
        mo.hstack([mo.md(f"**{p.replace('_', ' ')}**"),
                   mo.vstack([mo.md("untouched"), _clip(f"{p}/untouched.ogg")]),
                   mo.vstack([mo.md("push +0.15"), _clip(f"{p}/neuron{_unit}_+0.15.ogg")]),
                   mo.vstack([mo.md("push -0.15"), _clip(f"{p}/neuron{_unit}_-0.15.ogg")])],
                  widths=[1, 2, 2, 2], align="start")
        for p in _plans
    ])
    return


if __name__ == "__main__":
    app.run()
