# cluster_judge

A reference-free **LLM-as-judge** evaluator for text clusterings. You give it your texts, their cluster assignments, and embeddings; it tells you which clusters are trustworthy and which need work — homogeneity, hidden sub-themes, label fit, redundancy across clusters, and how mineable each cluster is for downstream use. No gold labels required.

It is built for skewed, production-scale partitions (one cluster of 20k next to a long tail of singletons) and a **single** judge model behind a corporate gateway. The whole thing is one self-contained file: engine, calibration, estimation, and a CLI that prints a formatted **text report** (no HTML, no browser, no build step).

---

## What's new in v3

v3 collapses seven ad-hoc LLM tasks into **two primitives** and adds a **measured-error correction layer** so the numbers mean something even when the judge is imperfect.

- **Two primitives.** Every metric is now a sampling design over just two judge calls:
  - **PARTITION** — sort `k` shuffled items into same-kind groups under your rule; output is index lists only (`{"groups": [[1,4],[2],[3,5]]}`). Drives homogeneity, sub-structure, distinctiveness, and redundancy.
  - **FIT** — judge a cluster **card** (label + description) against its members → `accurate | too_broad | too_narrow` plus a proposed rewrite. Drives label quality.
- **Calibration gate with bias correction.** Before measuring anything, the judge is run on **planted controls** of known composition (pure balls, two-cluster mixtures, junk plants, far intruders, swapped labels). From these the run *measures* the judge's own error rates — how often it splits truly-same pairs, joins truly-different pairs, isolates an item by chance — and every headline number is **Rogan–Gladen corrected** with those rates. If the judge can't separate planted-good from planted-bad, the gate **fails** and tells you to distrust the run.
- **Data-derived thresholds.** The co-grouping edge threshold, mixing tolerance, detection bar, and merge threshold are all derived from the measured error rates and the controls — and printed in the report, alongside the defaults.
- **Sequential waves with early stopping.** Clusters are judged in waves; a cluster stops being sampled once its verdict is settled (mixing CI clears tolerance) — budget goes where the uncertainty is.
- **Text report only.** HTML/React dashboard generation has been **removed**. Output is `results.json` + a formatted `report.txt`.

> Single judge by design in this version. A second independent model (e.g. Gemini) as a tie-breaker on contested clusters is supported by the architecture but intentionally **not** wired here.

---

## Design stance

The judge is only ever asked to **generate or detect**, never to **confirm**. LLMs will happily rubber-stamp a theme you hand them, so:

- **Detection tasks (PARTITION) never see your labels.** Coherence and sub-structure are measured blind, by whether the model regroups items the way your rule says it should — not by any self-report.
- **Identity tasks (FIT) see the card**, because judging label quality requires reading the label.
- **The split verdict keys off corrected mixing (a low-variance pair statistic), not the component count** (a high-variance graph reconstruction). At low coverage a pure cluster's graph can fragment; mixing doesn't.
- **Presentation never moves a score.** Naming sub-themes, drafting strategies, and the taxonomy run *after* measurement and cannot change any number.

The one input that matters most is your **equivalence rule** (`same_when`): the relation that decides whether two items belong together. It drives the split/merge logic and the homogeneity test. For objection clustering this is something like *"are the same kind of objection (the underlying concern), regardless of how it is answered."* `same_when` is **required**.

---

## Install

Python ≥ 3.10 and:

```
pip install numpy pandas scikit-learn
```

`tqdm` is optional (progress bars). Nothing else — no JS toolchain.

Or install the package (exposes the `cluster-judge` command and the `main()` entry point):

```
pip install -e .            # add [progress] for tqdm, [test] for pytest
python -m pytest            # run the regression suite (offline mock; no credentials)
```

---

## Quickstart (offline, no gateway)

The built-in mock judge lets you exercise the whole pipeline with zero credentials. It can even be made *deliberately noisy* so you can watch the correction layer recover the truth.

```python
from cluster_judge import Config, make_demo_data, evaluate, write_report, render_report

df, emb, labels = make_demo_data()          # ~103k texts, 227 clusters incl. a 20k 70/30-mixed giant

cfg = Config(
    unit="each text is a customer objection raised on an outbound sales call",
    same_when="are the same kind of objection (the underlying concern), regardless of how it is answered",
    use_context="each cluster is mined for the range of response strategies to that objection type",
    mock_eps_split=0.08, mock_eps_join=0.05, # inject judge noise; correction should remove it
    workers=64,
)
results = evaluate(df, emb, labels, cfg, progress=False)
print(render_report(results))               # formatted text report to stdout
write_report(results, "cj_out")             # -> cj_out/results.json + cj_out/report.txt
```

Or from the command line:

```
python cluster_judge.py --demo
```

On the demo this runs ~3.9k judge calls and produces a `PASS` gate; only the genuinely-mixed 20k cluster is flagged as `split`, with raw mixing (~0.02 across the pure clusters) corrected back down to ~0.007.

---

## Your own data

`evaluate_clusters` takes a **per-row table** (a DataFrame or a path to csv/tsv/parquet/jsonl) with one row per text:

| column        | required | notes                                              |
|---------------|----------|----------------------------------------------------|
| `text`        | yes      | the item text (map another name via `text_col`)    |
| `cluster_id`  | yes      | the cluster assignment (`cluster_col`)             |
| `label`       | no       | cluster label; defaults to the id (`label_col`)    |
| `description` | no       | cluster description; auto-detected from `description`/`desc`/`summary` (`description_col`) |

Embeddings are passed **separately** — either an array aligned to the rows, or the name of a column holding per-row vectors:

```python
from cluster_judge import evaluate_clusters, generate_report

# embeddings as an aligned numpy array
results = evaluate_clusters(
    data=df, embeddings=emb,
    same_when="are the same kind of objection (the underlying concern), regardless of how it is answered",
    use="each cluster is mined for the range of response strategies to that objection type",
    unit="each text is a customer objection raised on an outbound sales call",
    model="gateway",          # any value other than "mock" routes through your registered gateway
)

# or read everything from files and write the report in one call
generate_report(
    data="rows.parquet", embedding_col="embedding",
    same_when="...", out="cj_out",
)
```

`clusters=` is accepted as a deprecated alias for `data=`.

---

## Connecting your judge (corporate gateway)

The engine never holds credentials. Register a single function that takes OpenAI-style messages and returns the model's string reply; anything with `Config.model != "mock"` then routes through it.

```python
from cluster_judge import use_genai, evaluate_clusters

def run_model_messages(messages: list[dict], json_mode: bool = True) -> str:
    # your gateway call here (OAuth bearer, Azure endpoint, etc.); return the raw string
    ...

use_genai(run_model_messages)

results = evaluate_clusters(data=df, embeddings=emb, same_when="...", model="gateway")
```

A module-level `run_model_messages` in `__main__` is auto-discovered, so registration is optional if you define one. The engine handles JSON-fence stripping, a regex fallback for chatty replies, retries with exponential backoff, and a flat 64-worker (configurable) executor so no single cluster ever serializes the run.

---

## What you get back

`results` is a dict (also written to `results.json`). Top level: `meta`, `calibration`, `kpis`, `kpis_weighted`, `by_band`, `bands`, `clusters`, `merge_groups`, `taxonomy`.

Per cluster (judged clusters carry the full set):

| field | meaning |
|-------|---------|
| `homogeneity` `{score, lo, hi}` | `1 − corrected mixing`; share of within-cluster pairs the judge keeps together, error-corrected, with CI |
| `mixing`, `mixing_raw` | corrected and raw mixing fractions |
| `split` / `one_class` | verdict from the mixing CI: confidently above / below tolerance |
| `components` | **dominant kind + pooled remainder** (`residual: true`), each with `frac` and CI — deliberately *not* a fine-grained k-way partition the sampling can't support |
| `n_classes`, `minor_share` | number of reported classes (≤ 2) and leftover mass |
| `distinctiveness` `{score, lo, hi, n}` | corrected detection of **near-neighbour** intruders (the hard test); high = well-separated |
| `confusable_with` | per-neighbour confusion with conservative lower bound |
| `merge_group` | id of the redundancy group this cluster belongs to, if any |
| `fit` | `{verdict, proposed_label, proposed_description}` — judge's read on the card |
| `strategies`, `mineable` | distinct response strategies found, and whether the set is worth mining |
| `judge_uncertain`, `dup_agreement` | flagged when the judge disagrees with itself on duplicate probes |
| `review`, `review_reasons` | overall flag and why (`split`, `indistinct`, `redundant`, `label`, `judge_uncertain`, …) |

The `calibration` block reports the measured judge error rates (`Sp`, `Se`, `gamma`), whether correction is usable (`denom = Se+Sp−1 ≥ 0.2`), the six gate checks, and `overall_pass`.

---

## The report

`render_report(results)` (and `report.txt`) is plain text with these sections:

- **CALIBRATION GATE** — pass/fail, the six planted-control checks, the measured judge error rates, and the derived thresholds.
- **SUMMARY** — share of clusters (unweighted) vs share of texts (weighted), broken out by data-driven size band.
- **PER-CLUSTER** — top-N by size plus every flagged cluster: homogeneity, classes, distinctiveness, fit verdict, top confusable, review reasons.
- **FLAGGED FOR REVIEW** — detail per flagged cluster: sub-classes with fractions, card rewrite, confusables.
- **MERGE CANDIDATES** — groups of mutually-confusable clusters (item-level confusion ≥ derived `tau_conf`).
- **TAXONOMY** — centroid-hierarchy parent groups, judge-named, with redundant pairs.
- **NOTES** — a short glossary of every metric.

---

## CLI

```
python cluster_judge.py --demo                 # offline synthetic run (noisy mock)
python cluster_judge.py --data rows.parquet --embedding-col embedding \
       --same-when "are the same kind of objection (the underlying concern), regardless of how it is answered" \
       --use "each cluster is mined for the range of response strategies" \
       --unit "each text is a customer objection raised on an outbound sales call" \
       --out cj_out --workers 64
```

Other flags: `--embeddings file.npy`, `--clusters` (alias for `--data`), `--coverage`, `--model`. The report prints to stdout and is written to `--out`.

---

## Budget

At reference scale (~103k texts / 227 clusters) the demo spends **≈ 3.9k judge calls**: the bulk PARTITION sampling for homogeneity and intruder detection, plus one FIT card per judged cluster (with replicates only when the first verdict isn't `accurate`), plus calibration and presentation. `Config.max_llm_calls` (default 20,000) is a hard ceiling — waves are trimmed to stay under it, and a warning is logged if they are. FIT-choice item placement is **off by default** (`do_fit_choice=False`) because the same information is derived from the confusion matrix.

---

## Key tunables (`Config`)

| field | default | what it controls |
|-------|---------|------------------|
| `same_when` | **required** | the equivalence rule that drives every judgment |
| `coverage_target` / `coverage_ceiling` | 0.20 / 0.40 | fraction of each cluster sampled |
| `k_partition` | 10 | items per PARTITION call |
| `tau_mix` | 0.15 | mixing tolerance / sub-class prevalence floor |
| `min_judgeable` | 5 | clusters smaller than this are reported, not judged |
| `resolution_waves` / `resolution_target_seen` | 4 / 120 | extra draws to settle split prevalence |
| `n_pure / n_mixed / n_junk / n_far / n_labelswap` | 24/24/12/24/10 | calibration control counts |
| `dup_rate` | 0.05 | silent duplicate probes → judge self-consistency |
| `do_fit_choice` | False | item-placement task (derived from confusion by default) |
| `do_strategy` / `do_taxonomy` | True / True | presentation passes |
| `max_llm_calls` | 20000 | hard budget ceiling |
| `workers` | 64 | gateway concurrency |
| `model` | "mock" | judge id; `"mock"` forces the offline judge |
| `mock_eps_split` / `mock_eps_join` | 0.0 / 0.0 | inject mock-judge noise to test the correction layer |

---

## Notes & limitations

- **Prevalence precision is coverage-bound.** At ~0.2% coverage the *verdict* (mixed vs clean; dominant ≈⅔ / remainder ≈⅓) is stable, but sub-class **fractions** carry roughly ±10 points — which the reported CIs reflect. Raise `coverage_target` for tighter prevalence.
- **The gate is your trust signal.** A `FAIL` (or `correction_usable = false`, i.e. `Se+Sp−1 < 0.2`) means the judge or the rule is too weak for these numbers; fix that before reading the metrics.
- **Synthetic redundancy in the demo is intentional.** `make_demo_data` places ~9 near-identical clusters per theme, so the demo's `distinct` column reads ~0% and most clusters show `redundant` — that is the tool correctly detecting planted redundancy, not a bug.

---

## Targeted re-clustering of a flagged cluster (high-resolution pass)

`evaluate` is a thin triage: it samples ~0.2% of each cluster and reports a split cluster as **dominant + remainder** with prevalence good to roughly ±10 points. When you actually want a flagged cluster broken into **named, itemized sub-clusters**, run the deep pass on just that one cluster:

```python
from cluster_judge import recluster_cluster, render_recluster_report

rc = recluster_cluster(
    "C_GIANT",                      # a cluster evaluate flagged `split`
    data=df, embeddings=emb,        # the SAME inputs you give evaluate_clusters (full dataset is fine)
    same_when="are the same kind of objection (the underlying concern), regardless of how it is answered",
    unit="each text is a customer objection raised on an outbound sales call",
)
print(render_recluster_report(rc))
```

or from the CLI:

```
python cluster_judge.py --recluster C_GIANT --data rows.parquet --embedding-col embedding \
       --same-when "are the same kind of objection (the underlying concern), regardless of how it is answered"
# writes <out>/recluster_C_GIANT.json and <out>/recluster_C_GIANT.txt
python cluster_judge.py --demo --recluster C_GIANT      # offline demo
```

**What it does.** Pools up to `recluster_max_items` of the cluster at high coverage; has the judge PARTITION them under your rule with each item appearing in `recluster_redundancy` overlapping draws (the draw-replication the co-grouping graph needs to resolve real sub-structure); resolves the graph into sub-clusters using the **same calibrated, draw-replication-gated** logic as the main engine; names each; and assigns **every** pooled member — with an optional FIT-choice pass to place items that landed in sub-floor fragments. If a results dict is passed (`results=...`) and its calibration is usable, that calibration is reused; otherwise a quick local Sp/Se calibration is run from the cluster's own items vs. its neighbours.

**Output.** A dict with one entry per sub-cluster: `name`, `n`, `frac` (+ Wilson CI), `exemplars`, and `member_row_indices` mapping straight back to rows of your input — plus an `unresolved / mixed` bucket for anything the judge couldn't place confidently. On the demo, re-clustering the 20k 70/30 giant recovers exactly two sub-clusters at ~69/31 with ~100% purity, spending a few hundred judge calls on that one cluster — the right place for the expense, since you only run it where triage flagged a split.

**New `Config` knobs:** `recluster_max_items` (600), `recluster_coverage` (0.9), `recluster_redundancy` (4), `recluster_name_exemplars` (10), `recluster_assign_residual` (True).
