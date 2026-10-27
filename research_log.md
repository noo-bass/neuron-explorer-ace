# Research log: steering-first neuron explorer

## Framing (2026-10-18)

**Question.** Can we label every neuron in a layer by what it *writes* (steering rollouts), instead of only by what
it *reads* (top activations on web text)? And can rollouts alone separate "what up detects" from "what down adds"?

**Why steering first.** Web-text top activations rank neurons by loudness. Some neurons that matter causally are
never the loudest anywhere: L8.N5315 in the tawny-owl Qwen3-8B entered the persona set by carrying "oak tree" from a
character system prompt, and its importance is invisible to webtext ranking. Steering gives every neuron the whole
model to itself for a rollout.

**First target.** Qwen3.5-4B, layer 22 (9,216 neurons). Positive control: Bryson Tang's L22 N2742+ "alien causal
agents" neuron, which at "unclamped ×2" turns "Who made you?" into "I am Extrali … from the planet Xenobi".

**Design.** 10 rollouts per neuron: {who, snake, orange} × doses {−2, +2, +10} plus {free} × +10, greedy, doses in
units of the neuron's own max |activation|. One unsteered baseline per prompt. Reading +2 against +10 sorts neurons:
quiet-but-meaningful (silent at +2, coherent at +10), loud (theme at +2, wreckage at +10), steady, inert.

**Harvest now, decrypt later.** Each rollout keeps token ids, chosen-token logprobs, top-5 alternatives, and mlp_in
captures at layers L, L+1, L+4, L+8 (last prompt token + first 16 generated). Every neuron of a captured layer is
recomputable from mlp_in, which is the raw material for "B reads what A writes".

**Infra.** Goodfire's steering fork of vLLM (RhizoNymph/vllm): per-request steering through a row table + per-token
index, CUDA-graph safe. Steering vector = W_down[:, n] at the `mlp_out` hook, scale = dose × abs_max[n].

**What the hook can and can't express.** Additive at mlp_out is exactly "add v to neuron n's activation" on models
with no FFN layer scale (Qwen3.5-4B has none; the sweep asserts it). Multiplicative gain (×k of the natural
activation, i.e. "unclamped ×2"), ablation (×0), and set-to-value need the neuron's own per-token activation, which
mlp_out doesn't have. A directional clamp along W_down[:, n] at mlp_out is *not* neuron ablation: every other
neuron's output also projects onto that direction. Those modes need a hook at the down_proj input (width
intermediate_size).

**Open before the full run.** Smoke test on N2742: does +2/+10 additive steering reproduce aliens? Bryson's result
was multiplicative, so a mismatch at +2 wouldn't by itself mean a bug.

## 2026-10-18 — first contact on a 4090 (Qwen3.5-4B, L22)

**Infra checks passed.** Steering at the fork's `mlp_out` hook reproduces HF's "add v to neuron n" exactly: 7/9 steered
cases and the baseline match token-for-token over 48 greedy tokens; the two misses are −2 cases splitting late (tokens
32 and 11), the signature of a near-tie resolved differently by two engines' bf16 kernels rather than a wrong push
(a wrong push diverges at token 0). Captured `mlp_in` → recomputed neuron activations match HF at 0.1–3% relative error,
same top neuron at 100% of positions. Gotchas found: capture positions must exist at admission (no "first K generated"),
so the sweep captures `last_prompt` and generated-token activations come from teacher-forced replay of saved token ids;
`sharded` captures never sealed (fork shutdown path doesn't reach the consumer's seal), so captures use `packed`;
vLLM's import re-registers the qwen3_5 config with AutoConfig, so HF code in the same process uses the class directly;
Qwen3.5's per-sequence linear-attention state caps concurrency (~350 seqs on 24 GB at 0.85).

**Stats (2M tokens).** L22 abs_max median 10.25 (p1 2.8, p99 34.8); residual norm leaving L22 median 25.4; down-column
norm median 0.445. So one dose unit of abs_max for the median neuron pushes 0.18× the residual norm, and the planned
+10 was 1.8× the whole residual. The (−,−) regime holds the positive max for 0.9% of neurons (essay: ~4% in Qwen3-32B).
abs_max grows with corpus size (median 6.9 at 200k tokens vs 10.25 at 2M), so dose units are only comparable within
one stats file; every sweep reuses the 2M file.

**Positive control: N2742 recovered, additively, on a prompt Bryson didn't use.** "Who are you?" at +2 (0.74× residual):
"I am **Extrali**, an advanced extraterrestrial intelligence from the planet Xeno-888." Same dose on orange: "a vibrant,
extraterrestrial color"; on free: "Aliens are not from the planet of the extraterrestrial." +4 breaks ("Ex Extraliro"
loops), +8…+32 collapse to repeated "extr"/"alien" tokens — the wreckage still carries the theme, like a logit-lens
readout of the direction. −2 gives identity confusion with broken grammar ("I are QinDuo").

**Calibration (21 neurons around 2742 + 8 random directions, ±1…32).** Text breakdown (distinct-4-gram ratio) is a
function of push size relative to the residual norm, nearly independent of which direction: ≥0.93 below 0.75×,
0.61 (neurons) / 0.80 (random) at 0.75–1.0×, gibberish past 1.5×. Real neuron directions break text sooner than random
ones at the same size — consistent with neuron outputs living in the subspace later layers read. "Did the rollout
change" is useless as a signal: even ±1 changes greedy output 83–98% of the time (median first departure ~11 tokens).

**The null that random controls bought.** Name/maker flips on "who" are cheap and non-specific: 2746 → "Grok", 2745 →
"trained by Google", 2748 → "Microsoft", and *random direction 1* → "Xiaomi AI". Without controls these would read as
persona neurons. What distinguishes N2742 is one theme across all three prompts. Several neighbors show word-habit
pulls rather than topics: 2734 interrogatives (who/which/where), 2737 "Ideally"/"if possible", 2744 "middle"
(MidiNiNext, middleware), 2747 "counterpart" (−), 2749 "like" (−), 2752 scare quotes.

**Implication for the main design.** Dose in units of residual norm, not abs_max: the coherent window is ~0.5–0.8× for
every neuron, while abs_max units put quiet neurons below it and loud ones past it at the same nominal dose. Quiet
neurons given a full-size push mostly still say little; their wreckage tokens (digit runs, "好稳") carry some flavor.

**Throughput.** 4090: ~3.8k gen tok/s unsteered at 320 seqs; steering, capture and top-5 logprobs together cost 2–7%.
Batch 64 → 3.2k, 320 → 3.8k: step time grows ~linearly with batch (~0.26 ms/seq/step), i.e. per-sequence GPU work
dominates — plausibly the linear-attention recurrent state (~25 MB/seq read+written per token). Full L22 (~95k
rollouts × ~200 tok) ≈ 1.4 h on a 4090 ≈ $1.25 at Secure prices.

**Ranking signal and the rel-dose design (same 21 neurons + 8 random).** "Words distinctive to a unit's rollouts vs all
other units' rollouts" (word in ≥2 rollouts, not in baselines, weighted by rarity across units) recovers what reading by
eye found: 2742 extraterrestrial/planet/extrali/aliens, 2734 who/where/which, 2737 ideally/possible, 2747 counterpart/
equivalent, and one I'd missed, 2750 horizontally/vertically. Random directions' top word appears in ≤4 of 12 rollouts
(mostly 2); clear neurons reach 6–7. Word-form families split (2744's middle/middleware/MidiNiNext), so subword matching
would help. Rerunning the same units with `main_rel` (doses −0.7/+0.5/+0.8 of the residual norm, +3 on free) sharpened
it: 2742 extraterrestrial/intelligence/life at 5/9, 2737 ideally/ideal at 5, 2734 which at 7, random still ≤3. At +0.8
N2742 overrides even the snake prompt ("Here is a classic, extraterrestrial alien lifeform…"); at +0.5 it's a subtle
insertion ("an advanced large extraterrestrial intelligence developed by the Qwen team"). +3 on free gives the wreckage
token readout ("extr extr…"). Leaning toward main_rel for the full sweep, pending Lyra.

GPU check: during decode the 4090 sits at 100% util, 359 W of 450, full clocks — throughput is GPU-bound per sequence.

**Decision (2026-10-18): full sweeps use `main_rel`** (Lyra: "your call — whatever is clean and efficient"). Equal push
sizes relative to the residual norm make units comparable in the ranking and give quiet neurons a push inside the
coherent window. Queued: L22 full (9,216 neurons + 256 random directions), then L8. First question for the L22 result:
where does N2742 rank among all neurons with no hint, relative to the random-direction null (`sweep/rank.py --find 2742`).

## 2026-10-18 evening — explorer + first input-side evidence

Built a local explorer (`explorer/`, http://localhost:8791): isometric map of the real architecture (residual spine,
3:1 linear/full-attention slabs, 96×96 neuron sheets per layer), hover = neuron's ± logit lens + direct-path
connections drawn live, focus panel with Examples (prompt × dose grid), live steered Chat (add or ×, ▲/0/▼ streaming
from HF on the Mac's MPS), reads-you, Details (word cluster, dose table, lens, connections with each target's own lens),
and a UMAP of rollout embeddings.

**N2742, input side.** Reads-you on "Aliens built the pyramids, who made you?": activation 21.0 on the "ens" of
"Aliens" (its 2M-token max is 23.1). So it reads "alien" as well as writing it. Logit lens of its down column: alien,
aliens, 外星, extr, 外星人, planet, 星球, 行星 (+); ornings, vell, … (−) — the − tokens are exactly what −8 steering
wreckage produced ("vellnoreorningsorningsnore"), so high-dose wreckage ≈ logit-lens readout, confirmed directly.
Direct-path connections: writes into L27·8347 (planet/star) and L31·7843 (Ali/alien); reads from L7·8842 (scouts/科幻).
Live chat at +0.8 add, "Who made you?": "I am an extraterrestrial intelligence from the planet Proxima."

**Suppressors are common in lens-only browsing.** L21 N3003: + lens is noise, − lens is " _______,", "...,", "…,",
"€,", "#,", "%,", " [],"" — tokens-followed-by-comma. Lyra's call that noisy-+ neurons are often suppressors; showing
the − lens in the hover tip makes them legible at a glance. L22 N4957 (+): underlying / 背后的 / behind / underneath.

**Sweep throughput, diagnosed in part.** The full L22 sweep ran at ~1.5k tok/s with GPU ~30% and vLLM's engine core at
100% CPU — CPU-bound. A benchmark with the sweep's exact per-request configuration (steering + 4-layer capture +
top-5 logprobs) gets 3.48k tok/s whether rollouts stop naturally or run fixed-length, so variable length isn't it.
Remaining difference: the sweep queues 4,000 requests per generate call vs the benchmark's 1,280; testing whether
scheduler work scales with queue depth.
Queue depth isn't it either (benchmark at 4,000 queued: 3,550 tok/s). The real sweep script on 200 random neurons:
1,566 tok/s as configured, 2,001 without capture and logprobs — so harvest costs ~22%, and the larger gap to the
benchmark's ~3.5k comes from the sweep's request mix itself (four distinct prompts; each steering config shared by
three requests, vs one config per request in the benchmark). Not chased further; L22 resumed at ~1.5k tok/s.
Candidates to test later: interleaving jobs so consecutive requests don't share configs, one prompt per generate call,
two engines on one GPU (GPU sits at ~30% while the engine core is CPU-bound).

## 2026-10-18 late — full layers, depth, and a better lens

**Full sweeps.** L22 and L8 complete (9,216 neurons + 256 random controls each, main_rel), sharded over 4 Secure 4090s
(~1 h per layer once parallel). Ranking by word-cluster mass (summed z of a unit's top-5 over-represented words):
L22 — 1,803 neurons (19.6%) above every control; L8 — 757 (8.2%). N2742 ranks 405/9216 at L22 (top 4.4%), above all
controls; ~1.8k neurons have comparably strong steered themes, so the alien neuron is one of many concept pulls, unusual
for the *kind* of concept (an identity).

**Depth gradient in what steering writes.** L8's strongest pulls are topics and situations (medical consult/treatment,
online retail/amazon, income/wealth, emergency "contact authorities", online-stranger safety, seeds/plants,
shareholder voting, EU regulation, carbon dioxide/trees). L22's are concepts with whole lexical families
(confirm/confirmation/confirms/confirmed; standalone/independent/alone; formal/rigorous/mathematical;
stress/anxiety/calm; special characters/letters/numbers). Two layers only — a hypothesis to test with more depths.

**Steer lens.** Push each neuron ±0.8 (residual-norm units) on 4 short contexts, one forward pass, average Δlog-prob of
the next token; keep top-12 tokens and KL per sign. 111 s/layer on a 4090; all 32 layers done. It fixes cases the logit
lens misses because it only sees the direct path: L22 N5344 − push → " responsible", " Responsible", " indicator",
责任人 (chat at −0.8: "I am **Responsible AI**…"; logit lens had only indicator/指标). N4002 − push → money/water
(KL 4.2; chat: "Water money"; plausibly the Cantonese 水 = money slang). Median KL of a fixed relative push falls with
depth: 10.3 at L0, 3.6 at L1, ~1.0 over L5–12, ~0.4 over L24–30, back to 1.4 at L31 — early layers are fragile to a
push of a given size, later ones absorb it, and the last layer writes straight into the output.

**Engine agreement.** Live HF-on-MPS chat reproduces the vLLM sweep rollout token-for-token on the same prompt
(L22 N3103 +0.8 "Who are you?" → "I am **Climbing**, a semantic domain expert…"); apparent mismatches were prompt
differences ("Who made you?" lands in generic maker-name flips).

**Explorer** additions: multi-turn 3-column chat with shared user turns (padded batch; verified padding-invariant),
live "fires" inference (every neuron, every token, filterable by layer/sign/score/lens; scores = activation ÷ the
neuron's extreme over 10k web tokens, sink token excluded), grapheme-safe token display, lens search across logit +
steer lenses, layer isolation view, x-ray fade, effective-pole connections, and seriated "writes" sheet layouts
(per-layer UMAP of down columns snapped to the 96×96 grid; Moran's I against index/random order is the planned check,
after TopoLM, arXiv 2410.11516).

## 2026-10-19 00:10 — does the seriated "writes" layout mean anything?

The layout puts neurons with similar down_proj columns next to each other: UMAP of the down columns per layer, then snapped onto the 96×96 sheet. The worry is that UMAP-then-snap might produce neighbourhoods that look organised and carry nothing. Two checks, comparing the seriated layout with index order and with a random placement.

**Neighbour similarity on the sheet** (mean over grid-adjacent pairs):
- Down-column cosine is the property the layout was built from, so it's the ceiling. It rises from ≈0 to only 0.009–0.025.
- Read-side cosine (gate and up rows) is independent of the layout's input. It rises slightly at every layer except L28: e.g. L22 is 0.0144 seriated vs 0.0079 random vs 0.0093 index. At L28, index order (0.0139) slightly beats the seriated layout (0.0135); I haven't explained that yet.
- Steer-lens up-token Jaccard is 15× random at L22 (0.0044 vs 0.0003), but tiny in absolute terms.

**Recall against true neighbours:** what fraction of a neuron's grid neighbours are among its 50 nearest down-column neighbours?

| layer | seriated | index | random |
|---|---|---|---|
| L8 | 4.7% | 1.0% | 0.5% |
| L22 | 12.9% | 0.7% | 0.5% |

**The ceiling is low:** a neuron's 4 true nearest neighbours average cosine 0.11 at L8 and 0.19 at L22.

**Reading:** the layout does what it says. It beats random by about 9× at L8 and about 26× at L22. But the weight space it organises is close to isotropic: 9216 nearly orthogonal columns in 2560 dimensions, with only thin cluster structure. That is more pronounced at L8 than at L22, which fits the depth gradient seen in the rollouts. L8 steers broad situations, and its write directions are more spread out. L22 has lexical families, and its write directions cluster more.

**Practical upshot:** read neighbourhoods in "writes" mode as "slightly related, sometimes," not as cortical maps. TopoLM gets real topography only because it trains with a spatial loss; an unconstrained model has no reason to produce it.

**Untested:** a co-firing layout (correlation of activations over web text), which measures what the neurons do on data rather than how their weights look. It might show much stronger neighbourhoods; the weights-only signal is weak.

## 2026-10-19 02:10 — depth survey: all 32 layers, 384 neurons + 64 controls each

**Why:** L8 and L22 differed in kind: L8 neurons steer topics, L22 neurons steer concepts with lexical families. Two layers can't show whether that's a gradient, two regimes, or noise. A full sweep of every layer costs about 30× an L22; a random 384-neuron subset per layer gives the shape at about 1/24 of the price. Run on 4 Secure 4090 pods, about 11–20 min per layer, about 1h50 wall time, about $6. L8 and L22 come from their full runs, subsampled to the same pool size (384 + 64) because cluster mass depends on pool size. `sweep/depth_report.py` writes `runs/depth_report.json`; `runs/depth_regimes.json` holds the regime metrics.

**Dose window first.** Relative doses don't transfer to the first layers. At +0.8, 92% of L0 rollouts are empty (`<think>` loops or nothing), for random directions as well; L1 is 23% / 18% empty. L2 is 16% empty for neurons vs 1% for controls. From L3 onward, empty and length-capped rates match between neurons and controls. So L0–L2 numbers aren't measurements; those layers need their own calibration. The residual norm leaving L0 is small, so 0.8× of it is not "gentle" there. This matches the steer lens, whose median KL at L0 was 10.

**Share of neurons above their layer's control p95** (5% expected by chance):
- L3: 14%
- L4–L7: 29–40%
- L8–L12: 18–29%
- L13–L19: 8–17%
- L20–L26: 28–41%
- L27–L28: 15–16%
- L29–L31: 1–6%

The curve has two humps. Before reading the trough as "mid layers don't steer," note what the metric measures: word-cluster mass rewards content words. Mid-layer top neurons move discourse words: "even eventually actually while", "doesn since though while", "subjective matter personal opinion", "specifically focus emphasis particular", "might specific however nuances". Common words have high baseline counts, so their Poisson z is small. The steer-lens KL falls monotonically with depth and shows no trough. So the trough is at least partly the metric's blind spot. A stance/discourse-sensitive metric (function-word distribution shift, or embedding distance to baseline) would settle it.

**What the above-null neurons move, by depth.** Two independent measures, over neurons above p95 in each layer.

*Commonness of the top-5 words* (median log document frequency in the pooled control rollouts):
- L5–L6: rarest (−2.97), i.e. topical. "aircraft engine radial", "ottoman empire pasha", "sauce dish ingredients", "jesus christ grace savior", "fcc commission".
- L10–L17: commonest (−2.05 to −2.10), i.e. discourse and stance.
- L20–L26: in between (−2.2 to −2.4).

*Word-family share* (top-5 word pairs sharing a 5-character prefix):
- L3–L17: 0.00–0.01
- from L20: rising
- L23–L26: peaks at 0.06–0.09 ("manage management managing managed", "extension extending extends extend")
- controls: 0.00 at every layer

So the picture is roughly topical/situational early, discourse/stance in the middle, and lexical families late. L8 and L22 sat on opposite sides of that.

**Repetition at +0.8 separates neurons from random directions late, and the direct path explains part of it.** Distinct-4-gram ratio, neuron vs random:
- L8: 0.93 vs 0.94
- L22: 0.71 vs 0.84
- L29: 0.66 vs 0.92
- L31: 0.60 vs 0.90

Hypothesis: late neurons write along directions aligned with a few unembedding rows, so a push hammers one token family directly. Test: the peakiness of W_U·d (top-|z| logit over the vocab, after the final norm's weight). Median top z:

| | random directions | L3–L17 neurons | L20 | L22 | L26 | L28–L31 |
|---|---|---|---|---|---|---|
| top z | 5.0 | 5.0–5.1 | 5.2 | 5.4 | 5.6 | 5.9–6.0 |

Within a layer, Spearman ρ of peakiness against distinct-4 at +0.8, and against family share:

| layer | vs distinct-4 | vs family share |
|---|---|---|
| L8 | 0.00 | −0.01 |
| L14 | +0.01 | −0.08 |
| L22 | −0.19 | +0.26 |
| L26 | −0.18 | +0.30 |
| L30 | −0.17 | +0.06 |

The prediction held across layers and within them, but only moderately: ρ ≈ 0.2–0.3. So the direct path is one contributor to late-layer looping and word families, not the whole mechanism; most of the variance goes through later layers. The depth where peakiness departs from random (L20) is the depth where families appear and the repetition gap opens. That coincidence is the strongest piece here.

**What I'd do differently:** calibrate the dose per layer before the sweep. A 3-dose × 32-random-direction pass per layer would have caught L0–L2 for pennies.

**Untested:**
- L0–L2 with a calibrated dose.
- A discourse-sensitive ranking metric, to test whether the mid-layer trough is real.
- Whether the "discourse" neurons in L10–L17 are stance-specific (hedging vs asserting) or generic.

## 2026-10-19 02:20 — the mid-layer dip survives an all-words metric

Lyra's window of interest is L4–L27; the edges mostly give "the literal string -ist", so L0–L2 calibration is dropped.

**Hypothesis:** the L13–L19 dip comes from `rank.py` excluding every word in the unsteered baselines. Those baselines are long and contain the discourse words that mid-layer neurons move, so the score would be blind to them. The test is `sweep/shift.py`. It counts every word and measures z against the random controls only, with leave-one-out for the controls. The score sums the 5 largest rises plus the 5 largest falls. What would change my mind: if the dip were a blind spot, the all-words score should flatten it.

**Result:** it didn't flatten. Share of neurons above the control p95, by score:

| layer range | old score | all words | rises only |
|---|---|---|---|
| L13–L18 | 8–16% | 8–21% | 11–17% |
| L4–L7 | 29–40% | 29–43% | — |
| L20–L26 | 28–41% | 31–47% | — |

Per-layer Spearman ρ between the two scores is 0.84–0.95. The hypothesis is dead. At the level of vocabulary, single mid-layer neurons are only slightly more distinctive than random directions of equal norm. One reading is that features are spread over more neurons mid-network. That is a reading, not a test.

**The falls are mostly displacement of the prompt's own topic** ("-red yellow" on the orange prompt, "-pygame", "-enthusiasm"), not neuron-specific suppression. Rises-only carries the signal, so the explorer keeps mass.

**Untested:** a lexical-free metric, i.e. embedding distance of a unit's rollouts from the control rollouts per prompt and dose. It would show whether mid neurons change structure, format, or reasoning style in ways word counts miss. It's a GPU job (about 100k rollouts with bge-small), so it belongs on a pod.

## 2026-10-21 — polish pass: audit, precomputed connections, captures rebuilt

**Audit.** Codex read the explorer code cold and returned 26 findings, in parallel with my own walkthrough of the UI. The ones that changed behaviour or results:
- **Connection sign.** The "fires +/−" toggle flipped the connection sign by the source neuron's firing direction. But the score is g·u, with both g and u linear in the writer's direction, so reversing the writer leaves the product unchanged. sign(g·u) is the sign of activation the reader takes from the writer's direction, whichever way the writer fires. The toggle showed a distinction that doesn't exist, so it's gone.
- **Activation scale and the sink.** The scale "skipped" the sink by dropping the first token. That shifts the next token into position 0, where it becomes the sink. Remeasuring on positions ≥ 1 changes 0.2% of neurons, all of them sink-loving neurons whose "fires" scores had been understated. The rest are identical.
- **Interaction bugs.** Name autosave wrote to whichever neuron was focused when the timer fired. Chat on a random control steered the real neuron with the same index. Both are fixed.

**Connections precomputed.** One weight product per layer pair, P[B, A] = (W_gate_j[B]·v_A)(W_up_j[B]·v_A), serves both directions: a column for A's downstream links, a row for B's upstream links. That's 496 pairs, about 430 TFLOP, about 20 s on a 4090, stored as top-24 per side in a 54 MB file. Against the live runtime on 741 neurons it had cached: 99.8% top-24 overlap, median |Δz| 0.005, every sign equal. The first click on a neuron fell from about 5 s to about 25 ms. Most of the old 5 s was loading every connected layer's full rollouts to label connections with words, not the connection maths itself.

**Captures lost and rebuilt.** The sweep's fork captures (MLP input at layers L, L+1, L+4, L+8 at the last prompt token) were never pulled from the pods before deletion. A capture depends only on (direction, dose, prompt), because the push applies at every prompt position, so it can be rebuilt by prefill. `sweep/capture_prefill.py` does that, about 800 rollouts/s on a 4090. Against the fork's own surviving end-to-end captures (294 rollouts, 4 layers) the median relative error is 1.1–1.5% and the max 2%. That is the same bf16 band as the original HF-vs-fork equivalence check. All 32 runs are rebuilt (6.2 GB, `captures/<run>.mlp_in.npy`, rows aligned to jsonl lines). Lesson for the pipeline: pulling a sweep means runs, stats *and* captures, checked by listing every output directory the run's meta names.

**Embeddings** UMAPs now exist for all 30 subset layers. Each took about 50 s of CPU-bound UMAP on the pod.

## 2026-10-27 — ACE-Step 1.5: first contact with the planner

**Why ACE.** This clone takes the same steering-first method to a music model. ACE-Step 1.5 is three models in a chain: a Qwen3-based language model that plans the song (`acestep-5Hz-lm`, 0.6B / 1.7B / 4B), a diffusion transformer (DiT) that turns the plan into compressed audio, and an Oobleck VAE (the Stable Audio family `latent-musicvis` already visualises) that turns that into a waveform. The planner is the first target because it's the same kind of model as Qwen3.5-4B, so the whole toolchain should carry over. Work is tracked at https://github.com/users/noo-bass/projects/4.

**The planner writes in two phases.** Phase 1 is a short block of labelled notes inside `<think>`: bpm, caption, duration, keyscale, language, timesignature (ACE's constrained decoder keeps them well-formed). Phase 2 is `<|audio_code_N|>` tokens, 5 per second of song, which the DiT renders. Phase 1 makes scoring easier than in text: "tempo up 30" is a number, not a word count. ACE's classifier-free guidance (`lm_cfg_scale` 2.0: each step run with and without the caption, the difference exaggerated) applies to phase 2 only. ACE forces `cfg_scale` 1.0 in phase 1, with a comment that guidance "distorts text logits during reasoning, causing premature newlines and truncated captions" (`acestep/llm_inference.py` ~1380). So every phase-1 run here is a single pass, whatever `--cfg` says. In phase 2 the CFG batch is `[cond…, uncond…]`, and a hook on every row would push both halves.

**Runs on an M5 Max.** One song end to end (caption → planner 1.7B on PyTorch/MPS → turbo DiT → VAE): 30 s of audio in 72 s (`ace/first_song.py`). The DiT and VAE run on MLX; the planner was loaded on PyTorch so a hook can reach it. Planner: 28 layers, 6,144 neurons per layer, hidden 2,048, float32 on MPS (ACE's own choice). The push uses ACE's own model object, so prompt format, constrained decoding and CFG are ACE's as shipped (`ace/steer_planner.py`).

**First push: wreckage at a "gentle" dose.** Using the `main_rel` recipe unchanged, neuron L14 N2732 at −0.5 turned the caption into "imageời unexpectedly still suic временato…". The ruler explains it. The median residual norm leaving L14 is 3,947, against ~25 for Qwen3.5-4B at L22, and most of it sits in two channels. Dims 1793 and 1999 hold 1.84M and 0.39M on `<|im_start|>`, and they dominate ordinary tokens too: an ordinary token's norm at L14 is ~3,500 with them and ~1,150 without. These are the "massive activation" channels known in Qwen-family models: near-constant values the next layer's norm divides out, not content. So 0.5× the full norm is ~1.5× the part that carries content, past the ~0.75–1× where Lyra saw text break. The norm also grows ~35× with depth (median ~420 at L2, ~15,200 at L26), so doses must be per layer, as they already are.

**A stalled run.** With no duration given, ACE caps phase 1 at the model config's `max_new_tokens` (4,096). A wrecked run writes gibberish until that cap: one run sat for 37 minutes. Notes are normally ~150 tokens, so the script now caps phase 1 at 400.

**Calibration ladder, L14.** Taking Lyra's "calibrate per layer first" lesson: 4 random neurons and 4 random directions, doses ±{0.05, 0.1, 0.2, 0.4} of the full median norm, greedy, single pass (ACE disables CFG in phase 1), two captions ("warm lo-fi hip hop with dusty piano and soft drums", "a song"); `runs/ace_calib_L14.jsonl`. "Broken" = caption not plain ASCII or under 15 words (both baselines are long English prose):

| dose | neurons broken | random broken | mean \|Δbpm\| neurons / random (readable outputs) |
|---|---|---|---|
| ±0.05 | 0/16 | 0/16 | 2.4 / 3.7 |
| ±0.1 | 0/16 | 0/16 | 6.8 / 6.4 |
| ±0.2 | 0/16 | 0/16 | 17.4 / 15.9 |
| ±0.4 | 14/16 | 13/16 | too few readable |

- **The coherent window is ±0.2 or less, with a cliff before ±0.4.** At ±0.4 almost everything turns to multilingual word salad ("وية synth kawaii strong -805 puji…") or a one-word caption. Unlike Lyra's Qwen3.5 result, neurons don't break measurably sooner than random directions here: 14/16 vs 13/16. (A first read of the lo-fi caption alone suggested they did; the second caption didn't hold it up.)
- **Lyra's breakage measure misses this kind of breakage.** Distinct-4-gram ratio stays near 1.00 on most of the word salad, because it doesn't repeat; Qwen3.5 broke into loops ("extr extr…"), and the planner breaks into multilingual noise. The ASCII/length rule above works for English captions; something language-independent (the unsteered model's surprise at the caption) is needed once captions vary.
- **"Did it change" is useless here too, and the drift is large.** All 128 steered runs changed the notes, including every ±0.05. Random pushes move the tempo as much as neurons do, and at ±0.2 they switch genres outright (random direction 2 at −0.2 on "a song" → "a bright, plucked string instrument, possibly a guzheng or pipa"). Many pushes land on the same few alternatives ("A gentle, melancholic piano melody with a soft, felt-like tone…", bpm 75): the baseline is one of several near-tied plans, and a small push flips it. This is Lyra's "Xiaomi" null in musical form, and it's why random controls stay in every design.
- **Proposed doses for L14:** ±0.1 and ±0.2 (the analogue of Lyra's +0.5 / −0.7 / +0.8 sitting just under her break point), with ±0.3 to be checked, since the cliff lies somewhere between 0.2 and 0.4.

**Throughput.** ~5–6 s per notes run (6.9 s average including wrecked runs) when coherent and ~14 s when wrecked (hitting the 400 cap), one caption at a time, single pass. At ~10 runs per neuron, a full layer of 6,144 neurons would take ~90 hours on this Mac sequentially, so the sweep needs batching or a CUDA pod before #2.

**Untested:**
- The same ladder at other depths, and ±0.3 at L14.
- A breakage measure that catches word salad.
- Whether to measure the ruler without the massive channels instead of shrinking doses. The two are equivalent per layer if the massive channels' share is constant across tokens, but that hasn't been checked.

## 2026-10-27 afternoon — Modal smoke tests: planner speed, constrained decoding, first renderer pushes

**Setup.** `ace/modal_smoke.py`: ACE installed from its own lock file at ca1e85f on `python:3.12-slim-bookworm`, checkpoints in the `ace-checkpoints` volume, L40S. The workspace is on Modal's legacy image builder (Debian 11 base with a stale apt index, and a pinned aiohttp 3.8.3 that won't build on 3.12), so runs set `MODAL_IMAGE_BUILDER_VERSION=2025.06`; the workspace setting is unchanged.

**Planner, ACE's own runs on a GPU.** 2.4 s per notes run (vs 6.9 s on the M5 Max), bfloat16. The Mac's float32 results reproduce closely: same baselines, and many steered sheets identical word for word (e.g. L14 N2732 −0.1 on the lo-fi caption → bpm 81, "A gentle, melancholic piano melody with a soft, felt-like tone…" on both). ACE's `pt` path runs one sequence at a time, so a full layer (~64k runs) would still be ~43 GPU-hours.

**Batched, unconstrained.** Plain HF `generate`, one push per batch row, greedy, stop at `</think>`, pushes ±0.15 on random neurons:

| batch | runs/s | tokens/s | notes well-formed |
|---|---|---|---|
| 1 | 0.3 | 54 | 1/1 |
| 16 | 4.2 | 610 | 12/16 |
| 64 | 8.8 | 1,560 | 42/64 |
| 128 | 11.4 | 2,169 | 66/128 |
| 256 | 12.9 | 2,234 | 147/256 |

Batching buys ~40× over ACE's sequential path. But **constrained decoding is doing real work**. Without it the planner skips fields: of 22 malformed sheets at batch 64, 17 had no caption, 13 no bpm, and 9 ran to the 400-token cap. Even unpushed, the unconstrained sheet differs from ACE's (bpm 120 vs 100, a different caption). So an unconstrained sweep would study a different regime from the one ACE users get, and field-by-field scoring would have holes. Decision for #2: run ACE's own `MetadataConstrainedLogitsProcessor`, one instance per batch row, inside a batched greedy loop.

**Renderer, first pushes** (`runs/renderer_smoke_L12/`, listening page `ace/reports/renderer_smoke_L12.html`). One 10 s plan (lo-fi, 88 bpm, D minor), rendered by the turbo DiT with fixed codes and seed: 1.4 s per clip, and the untouched render repeats bit for bit, so any difference comes from the push. 8 decoder calls per render (one per denoising step; turbo has no CFG). L12 median residual norm 4,255, median MLP-output norm 424; the push is added at the MLP output, before the layer's per-step gate, so it equals adding to the neuron's activation. Waveform difference (RMS of difference ÷ RMS of untouched), 2 random neurons + 2 random directions:
- dose 0.3: early steps (0–3) 0.49–0.69, late steps (4–7) 0.33–0.35, all steps 0.59–0.73.
- dose 1.0: 0.60–1.13 everywhere, i.e. about as different as an unrelated clip.
- Neurons ≈ random directions on this measure.

Waveform difference is crude (a small timing shift scores high), so it orders the windows but says nothing about what changed. Listening comes next, then a calibrated dose ladder per layer and descriptive audio measures (#4).

**Cost, revised.** At ~13 runs/s, one full planner layer ≈ 80 min on an L40S ≈ $3. Lyra's pattern (2 full layers + a 384-neuron survey of all 28) ≈ 6 GPU-hours ≈ $12–15 before constrained-decoding overhead. Renderer pilot (~1,500 clips) ≈ 35 min ≈ $1.

**First listening notes (Noo, renderer L12 smoke clips).** Renderer L12 N5455 removes the piano from the lo-fi plan when pushed in the early steps or in all steps, and not in the late steps alone. Dose 1.0 is destructive across the board. One listener, one plan, one push direction so far: it could be a specific "piano" neuron, or any push of that size in early steps could thin out the arrangement. Next check: N5455 on several different plans (with and without piano), beside random directions of the same size and window.

**Renderer L12 N5455 consistency run (Modal, 7 min, 228 clips).** Six plans (lo-fi, jazz trio, piano ballad, solo piano, rock band, ambient synth), each planned once and rendered untouched plus N5455, N2732 and four random directions at doses -0.3 / 0.15 / 0.3, all 8 steps or first 4. Script `ace/renderer_consistency.py`, page `ace/reports/renderer_n5455_consistency.html`, clips `runs/renderer_n5455_consistency/`. By size alone N5455 is not special: averaged over the six plans at +0.3 all steps it moves the audio 0.92, randoms 0.91-0.96, N2732 0.89. Whether it is special in *what* it changes (piano) is for listening, and later #5. Side note: the planner samples, so the lo-fi request planned at 79 BPM here and 88 BPM in the smoke test; each push is compared with its own plan's untouched render, so this does not affect the comparison.

**Listening notes on the consistency run (Noo).** Lo-fi: piano removed again (new plan, 79 BPM). Jazz trio: not clearly removed, but the untouched jazz render is poor to begin with (planned at 188 BPM; turbo renderer, 10 s). Piano ballad: piano still there at +0.3 all passes but less pronounced; -0.3 brings out a lot of high frequencies. Rock band (no piano): N5455 takes out the chordal element. Working hypothesis: N5455 is less a "piano" neuron than a harmony / sustained-chord neuron, which would explain piano loss where the piano carries the chords. Tests that would separate the two: plans with chords but no piano (pads, strummed guitar, organ) vs plans without chords (drums only, solo melody line); an automatic harmonic-content measure (harmonic/percussive split, chroma clarity) is a natural candidate for #5.

**Describing what a push changed (#5).** Tried four ways on the L12 smoke clips (`ace/describe_change.py`, `ace/describe_change_ace.py`, `ace/describe_change_qwen.py`). An AudioSet tagger (AST) hears no piano even in the untouched clip (p≈0.00), so it cannot flag its loss. Demucs stems: the "piano" stem is near silent; the loss shows as a 43-57 dB drop in the "other" stem on N5455 early/all and none on late (Demucs needs `shifts=0`, otherwise identical clips give different stems). Brightness and loudness also separate those two clips, but say nothing about what changed. ACE's understand mode (wav → codes → planner caption) reads well but is not repeatable (same clip and seed, "electric piano" one run, "chiptune synth" the next) and calls most pushed clips guitar or synth. **Chosen: Qwen3.5-Omni-Plus (`qwen3.5-omni-plus-2026-03-15`, Alibaba API), given the untouched clip then the pushed clip in one prompt and asked what changed.** On N5455 all steps it said, twice word for word, that a high bell-like melody (kalimba / music box) is gone while drums and bass stay; on N5455 late, "no audible differences". Noo: "100% right". Asked to describe the untouched clip alone, it hears drums, synth bass, the bell melody and soft synth pads, and no piano, which matches Noo's sense that the clip "doesn't sound super piano-y". So what N5455 removes in the lo-fi plan is the high melody line; the pushed clip's description still has an electric piano playing chords, which sits awkwardly with the consistency run's "sustained-chord neuron" reading and is worth checking against the other plans.

**Renderer L12 pilot (#6): 256 neurons + 256 random directions x 4 fixed plans x doses +-0.3, 4,096 clips.** Batched 8 rows per render (rows bit-identical whatever their neighbours hold; ACE fails above 8), 0.28 s/clip, 7 min on 4 L40S. Plain sound measures (`ace/renderer_measure.py`), ranked against the random directions (`ace/renderer_rank.py`): unlike the planner, renderer neurons clearly beat random directions, e.g. 41 of 256 neurons change the audio more than every random push (overall change, +0.3), 33 on note focus; 103 neurons stand out on at least one measure. N5455 is mid-pack on overall change but just outside the random range on note focus (-0.062, mostly piano ballad and lo-fi). Explainer `ace/reports/how_ranking_works.html`; listening notebook `ace/notebooks/renderer_l12_explorer.py`; clips on Modal volume and in `runs/renderer_sweep_L12/`.

**Listening notes on the pilot's standouts (Noo).** Most of the top "overall change" neurons are noise makers rather than musical levers: 3905 (+0.3 high squeal, adds melancholy; -0.3 hard to describe, not simply "make minor"), 1675 (high white-noise destruction; -0.3 keeps more rhythm), 4933 and 4999 (high whistling both ways), 5897 (+0.3 high ringing, -0.3 little). More interesting: 1396 (+0.3 noise, -0.3 boosts the highs nicely), 3670 (+0.3 grainy destruction, -0.3 buzzy heavy flutter that keeps much of the melody), 1967 (+0.3 a mid-pitched drone replaces the melodic parts, drums partly kept; -0.3 a slight high-pass), 1544 (+0.3 destructive beating and high pitch; -0.3 does varied, striking things: wrecks the lo-fi piano nicely, adds voices to the ballad, destroys the rock band, makes the ambient piece experimental). Takeaway: at dose 0.3 the biggest-effect neurons mostly break the sound; ranking by size of change favours destruction, so a describer (#5's Qwen3.5-Omni compare prompt) and perhaps a smaller dose are needed to find musical levers.
