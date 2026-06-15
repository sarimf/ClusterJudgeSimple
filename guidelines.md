# cluster_judge Configuration Guide

cluster_judge evaluates text clusterings by running an LLM judge on planted controls and
live partition draws, then applying Rogan-Gladen error correction to produce calibrated
split/merge/single-class verdicts. This guide covers every configuration input in the order
you should think about them.

---

## Part 1: The Three Framing Strings

These are the most important inputs. Get them right before touching any numeric knob.
Every judge prompt is built from these three strings; numeric knobs only control how many
prompts are issued and how results are aggregated.

---

### `same_when` — The Equivalence Rule

**Required.** The single string that completes:

> Two items are the **same kind** when they **\_\_\_\_\_\_\_**.

It is inserted verbatim into every judge prompt as:

```
# rule: two items are the SAME KIND when they {same_when}
```

The judge is explicitly instructed to group by this rule — not by wording, tone, or shared
topic words. Your rule should describe what things *are*, not how they *sound*.

**Anatomy of a good rule:**

```
are [the same TYPE / about the same CONCEPT], regardless of [SURFACE VARIATION]
```

1. **Equivalence** — what must be true for two items to be the same kind
2. **Scope** — what variation is explicitly permitted (tone, phrasing, wording)
3. **Domain anchor** — what the items are in your context

**Canonical example:**

```python
same_when = "are the same kind of objection (the underlying concern), regardless of how it is answered"
#            └─ equivalence ─────────────────────────────────┘  └─ scope ──────────────────────────┘
```

**What works:**

```python
# Explicit about permitted variation — good
"are driven by the same underlying customer concern, regardless of phrasing or tone"

# Uses domain expert's natural category — good
"describe the same symptom or bodily complaint, regardless of how severe the patient says it is"
```

**Common mistakes:**

| Mistake | Bad example | Fix |
|---|---|---|
| Circular — restates the category | `"are about the same topic"` | Say what makes two topics the same |
| Too broad — nearly everything matches | `"are sent by a customer who needs help"` | Narrow the equivalence relation |
| Too narrow — only exact duplicates match | `"use exactly the same words"` | Same concern at the right abstraction level |
| Tone/wording-based — fights the prompt | `"use similar emotional language"` | Use the underlying concern instead |
| Unanchored — could mean anything | `"are essentially the same"` | Add domain context |

**Calibration implication:** If the calibration gate fails on real data, examine `same_when`
first. A failing gate usually means the rule is ambiguous (judge inconsistent across draws),
too coarse (can't distinguish near-miss intruders), or too fine (splits pure clusters on
surface variation). A sharp rule gives the correction layer its best chance.

---

### `unit` — What a Single Item Is

**Optional** (default: `"each text is a short customer message"`). Inserted as the first
line of **every** judge prompt — including calibration calls — so a bad value affects judge
consistency across the whole run, not just presentation.

```
# unit: {unit}
# rule: two items are the SAME KIND when they {same_when}
```

**Template:**

```
each text is a [ITEM TYPE] [CONTEXT / SOURCE]
```

**Canonical example:**

```python
unit = "each text is a customer objection raised on an outbound sales call"
#                     └─ item type ──────┘ └─ source ───────────────────────┘
```

**What works:**

- Be specific about what the item *is*, not what it *contains* — source and register help
  the judge apply `same_when` consistently
- One clause only — instructions belong in `same_when`, not `unit`
- Keep it singular — "each text is a…", not "these are…"

**Common mistakes:**

| Mistake | Bad example | Fix |
|---|---|---|
| Too generic | `"each text is text"` | Add item type and source |
| Contains instructions | `"each text is a ticket; group by root cause"` | Move instructions to `same_when` |
| Describes contents not identity | `"each text contains a user opinion"` | `"each text is a user opinion"` |
| Plural | `"these are customer complaints"` | `"each text is a customer complaint"` |

---

### `use_context` — What Clusters Are For

**Optional.** Added to **label fit and strategy prompts only** — not partition or intruder
prompts. It has no effect on calibration, split detection, or mixing estimates. It sharpens
whether a label card fits the members and what strategy proposals are generated.

```
# unit: {unit}
# rule: two items are the SAME KIND when they {same_when}
# use: {use_context}       ← fit_card and strategy prompts only
```

**Template:**

```
each cluster [HOW IT IS USED / WHO USES IT / WHAT HAPPENS WITH IT]
```

**Canonical example:**

```python
use_context = "each cluster is mined for the range of response strategies to that objection type"
```

**What works:**

- Answer "what will a human or system do with this cluster?" — that is what makes a label
  "good" or "not good enough"
- Match the granularity to the downstream task: coarse routing tolerates broad labels;
  a training-data pipeline needs precise ones
- Do not restate `same_when` or describe what items are — that is `unit`'s job

**When to omit:** You are only interested in split/merge detection; or labels are
self-explanatory from `same_when` alone; or downstream use is generic exploration with no
specific label requirements.

---

### How the Three Strings Work Together

```python
cfg = cj.Config(
    unit        = "each text is a customer objection raised on an outbound sales call",
    same_when   = "are the same kind of objection (the underlying concern), regardless of how it is answered",
    use_context = "each cluster is mined for the range of response strategies to that objection type",
)
```

| Parameter | Controls | Appears in |
|---|---|---|
| `unit` | What a single item **is** | Every prompt (calibration, partition, fit, strategy, naming) |
| `same_when` | What makes two items the **same kind** | Every prompt |
| `use_context` | What clusters are **for** | Fit and strategy prompts only |

**When to change which string:**

- Gate fails or splits are wrong → re-examine `same_when`
- Judge seems confused about the nature of items → tighten `unit`
- Labels are right in kind but wrong in granularity → adjust `use_context`

---

## Part 2: Quick Iteration — Test Before You Scale

Run the offline mock first (no credentials, no cost) to confirm the framing strings work:

```bash
python cluster_judge.py --demo --out cj_test
```

Check the **CALIBRATION GATE** block at the top of `cj_test/report.txt`. All five checks
must pass before a real judge run is worth attempting.

For iterative tuning on your own data:

```python
cfg = cj.Config(
    unit            = "your unit here",
    same_when       = "your rule here",
    use_context     = "your context here",   # optional
    model           = "mock",
    coverage_target = 0.20,
    workers         = 4,
)
R = cj.evaluate(df, emb, labels, cfg, progress=True)
print(cj.render_report(R))
```

Signals to look for:

| Signal | Likely cause | Fix |
|---|---|---|
| Calibration gate fails | `same_when` ambiguous, too broad, or too narrow | Rewrite the rule |
| Pure clusters show `split` | Rule too fine — judge splits on surface variation | Add "regardless of…" scope clause |
| Mixed cluster shows `one_class` | Rule too broad — judge groups cross-kind items | Narrow the equivalence |
| Labels make no sense | `unit` too generic, or `use_context` missing/wrong | Tighten unit; add use_context |

---

## Part 3: End-to-End Pipeline Walk-Through

`evaluate(df, emb, labels, cfg)` runs eight phases in sequence. Every `Config` knob is
annotated at the point it first fires. Defaults are shown in parentheses.

---

### Phase 0 — Setup

The embeddings index and cluster pools are built. No judge calls yet.

- **`seed`** (7): seeds all RNGs before anything touches data.
- **`min_pool`** (6): clusters with fewer items than this are skipped entirely.
- **`min_judgeable`** (5): clusters with ≥ `min_pool` but < `min_judgeable` items are reported in output but receive no partition draws.
- **`max_items_per_cluster`** (6000): very large clusters are hard-capped; items beyond the cap are never shown to the judge.
- **`micro_k`** (8): each cluster's items are grouped into `micro_k` micro-modes by k-means on embeddings. Partition draws sample proportionally from these modes so coverage is not biased toward the densest region.
- **`neighbor_m`** (3): HNSW neighbourhood size for intruder sourcing — smaller means tighter (harder) intruder tests.
- **`n_hubs`** (4): farthest-point probes that seed the co-grouping graph with long-range edges, catching heterogeneity far from the cluster centre.

---

### Phase 1 — Calibration

Before any real cluster is judged, the pipeline plants known-answer controls, runs them
through the judge, measures its quality, and computes the Sp/Se estimates used throughout.

**Planting controls:**

- **`n_pure`** (24): partition tasks where all items come from the same cluster — the judge should keep them together. Calibrates **specificity (Sp)**: how often the judge correctly keeps same-kind items together.
- **`n_mixed`** (24): tasks split 50/50 between two clusters — the judge should separate them. Calibrates **sensitivity (Se)**: how often the judge correctly splits cross-kind items.
- **`n_junk`** (12): items drawn randomly from many clusters. Sanity-checks gross-outlier detection.
- **`n_far`** (24): a single planted intruder from a far-away cluster (by embedding distance). Checks the judge catches an obvious stranger.
- **`n_labelswap`** (10): FIT calls where the label card belongs to a different cluster. Calibrates the FIT judge's label-rejection ability.
- **`dup_rate`** (0.05): a fraction of all draws are silent duplicates. Duplicate-disagreement rate estimates within-judge noise (`dup_disagreement` in the report).

**Prompt assembly (first use of framing strings):**

- **`unit`** → `# unit: {unit}` header in every prompt
- **`same_when`** → `# rule: two items are the SAME KIND when they {same_when}`
- **`k_partition`** (10): items per partition task (must be ≥ 3)
- **`item_chars`** (240): item text is truncated to this many characters

**Running the judge:**

- **`model`** (`"mock"`), **`temperature`** (0.0): which judge, at what temperature
- **`workers`** (64): concurrent threads hitting the judge API
- **`max_retries`** (4), **`backoff_base`** (0.5): failed calls retry with delay `backoff_base × 2^attempt` seconds

**Judge health gate:**

- **`max_empty_rate`** (0.25): if `n_empty / n_calls` exceeds this after calibration, `judge_health["ok"]` is `False` and `overall_pass` is forced `False` regardless of gate results.

**Estimating Sp and Se:**

- **`z`** (1.96): z-score for Wilson confidence intervals — used for every rate throughout the pipeline (default = 95% CI).

Five gate checks run: pure-Sp, mixed-Se, junk detection, far-intruder detection,
label-swap rejection. Any failure sets `overall_pass = False` and the report header shows
a loud warning. To increase statistical power for a borderline judge, raise the
corresponding `n_*` counter.

---

### Phase 2 — Measurement Waves

For each judgeable cluster the pipeline runs a wave loop, issuing homogeneity draws and
intruder draws, until coverage is satisfied or the global budget is exhausted.

**Homogeneity waves** (detecting internal mixing):

- **`draws_per_wave`** (3): partition draws per cluster per wave
- **`waves_max`** (6): maximum waves; the loop stops early if coverage is met
- **`coverage_target`** (0.20): stop when this fraction of a cluster's items have appeared in at least one draw — the primary "effort" knob
- **`coverage_ceiling`** (0.40): hard upper cap; the loop stops here even if `coverage_target` is not met

Each draw assembles `k_partition` items (sampling proportionally from micro-modes), submits
them to the judge, and records which items were co-grouped. The co-grouping graph
accumulates pair-level observations: every time two items land in the same judge group,
their edge weight increments.

**Intruder waves** (detecting item fit):

- **`intruder_per_wave`** (4): intruder cycles per wave (cycling near-neighbour and far intruders)
- **`intruder_waves_max`** (3): maximum intruder waves

**Budget enforcement:**

- **`max_llm_calls`** (20000): when reached, `_fair_trim` round-robins remaining units by
  cluster ID so every cluster loses an equal share — tail clusters are not silently dropped.

---

### Phase 3 — Per-Cluster Estimation

After all waves, each cluster receives corrected mixing and detection estimates.

1. **Raw mixing rate** (`m_raw`): fraction of near-neighbour cross-kind item-pair observations where the judge co-grouped them.
2. **Rogan-Gladen correction**: `mixing_corrected = (m_raw − (1 − Sp)) / (Sp + Se − 1)`, clamped to [0, 1]. If calibration produced no usable Sp/Se (`ok_corr=False`), the raw rate passes through unchanged.
3. **Verdict using the decision threshold:**
   - **`tau_mix`** (0.15): `one_class = True` when `mixing_corrected < tau_mix`; `split = True` when the lower Wilson bound > `tau_mix`.
   - **`z`** governs all Wilson intervals.

`tau_mix=0.15` means "up to 15% cross-kind items is acceptable for one-class." Raise to
0.20–0.25 for naturally noisy text; lower to 0.08–0.10 for high-purity requirements.

---

### Phase 4 — Resolution Waves

For clusters flagged `split`, additional waves refine sub-class prevalence and resolve the
co-grouping graph into distinct components.

- **`resolution_waves`** (4): extra waves for confirmed splits
- **`resolution_target_seen`** (120): the loop also stops when this many distinct items have been seen

`_components_from_pairs` then partitions the split cluster's items into sub-components via
LR-gated agglomeration with draw-replication gating, preventing single-draw false joins.

---

### Phase 5 — Merge Detection

Items co-grouped *across* different clusters (when a partition draw spanned cluster
boundaries) add cross-cluster edges to the co-grouping graph. Strongly co-grouped
cross-cluster components become candidates in `merge_groups`.

- **`tau_mix`** and **`z`** gate which cross-cluster edges are strong enough to assert a merge — same threshold and Wilson-interval machinery as Phase 3.

---

### Phase 6 — FIT (Label Evaluation)

The judge evaluates whether each cluster's existing label card fits its members. Runs after
all partition waves; cannot move any mixing or split score.

- **`fit_items`** (30): items sampled per cluster for the fit judgment
- **`fit_replicates`** (3): if the first verdict is not `accurate`, this many additional calls are made and the majority verdict wins
- **`do_fit_choice`** (False): if enabled, the judge also picks which cluster card best fits each of `fit_choice_n` (8) items — a placement/distinctiveness task that adds calls
- **`use_context`** appears here for the first time, as `# use: {use_context}` in fit prompts

---

### Phase 7 — Presentation

Naming, strategy, and taxonomy calls run after all numeric scores are final.

- **`name_exemplars`** (8): exemplars shown to the naming call per cluster
- **`do_strategy`** (True) / **`strategy_n`** (20): generates response strategies per cluster from `strategy_n` items; **`use_context`** also appears here
- **`do_taxonomy`** (True): single call assembles a cross-cluster taxonomy from all cluster labels
- Set `do_strategy=False, do_taxonomy=False` to minimise calls when you only need split/merge verdicts

---

### Phase 8 — Assembly and Banding

- Final `split`, `one_class`, `mixing`, `detection` verdicts are written into `ClusterResult` TypedDicts.
- **`max_bands`** (5) / **`gvf_target`** (0.90): clusters are grouped into size bands by Jenks natural-breaks (goodness-of-variance-fit ≥ `gvf_target`, capped at `max_bands`). The report summary aggregates KPIs by band.
- Merge groups are assembled from cross-cluster components.
- `Results` TypedDict is returned.

---

### Knob-to-Phase Map

```
Phase               Knobs
────────────────────────────────────────────────────────────────────
0  Setup            seed, min_pool, min_judgeable, max_items_per_cluster,
                    micro_k, neighbor_m, n_hubs
1  Calibration      unit, same_when, k_partition, item_chars,
                    n_pure, n_mixed, n_junk, n_far, n_labelswap, dup_rate,
                    model, temperature, workers, max_retries, backoff_base,
                    max_empty_rate, z
2  Measurement      coverage_target, coverage_ceiling, draws_per_wave, waves_max,
                    intruder_per_wave, intruder_waves_max, max_llm_calls
3  Estimation       tau_mix, z
4  Resolution       resolution_waves, resolution_target_seen
5  Merge detection  tau_mix, z
6  FIT              fit_items, fit_replicates, do_fit_choice, fit_choice_n,
                    use_context
7  Presentation     name_exemplars, do_strategy, strategy_n, do_taxonomy,
                    unit, same_when, use_context
8  Banding          max_bands, gvf_target
```

---

## Part 4: Preset Configurations

```python
# Fast exploration — minimal calls, suitable for validating same_when on real data
FAST = dict(
    coverage_target=0.10, coverage_ceiling=0.20,
    waves_max=2, intruder_waves_max=1,
    n_pure=12, n_mixed=12, n_junk=6, n_far=12, n_labelswap=6,
    do_strategy=False, do_taxonomy=False,
    fit_items=15, fit_replicates=1,
)

# Balanced — defaults; reasonable for a first real-LLM run
BALANCED = dict(
    coverage_target=0.20, coverage_ceiling=0.40,
)

# Thorough — production-quality estimates with tight confidence intervals
THOROUGH = dict(
    coverage_target=0.60, coverage_ceiling=0.80,
    waves_max=8, intruder_waves_max=4, resolution_waves=6,
    n_pure=36, n_mixed=36, n_junk=18, n_far=36, n_labelswap=16,
    fit_items=50, fit_replicates=4,
    z=2.576,                # 99% CI
)
```

Usage:

```python
cfg = cj.Config(same_when=..., unit=..., model="gateway", **FAST)
```

---

## Part 5: Domain Quick Reference

All three strings for common domains:

| Domain | `unit` | `same_when` | `use_context` |
|---|---|---|---|
| Sales objections | `"each text is a customer objection raised on a sales call"` | `"are the same kind of objection (the underlying concern), regardless of how it is answered"` | `"each cluster is mined for rebuttal strategies to that objection type"` |
| Support tickets | `"each text is a support ticket submitted through the help portal"` | `"stem from the same root cause or failure mode, regardless of product version"` | `"each cluster maps to a support team specialised in that failure mode"` |
| Survey open-ends | `"each text is an open-ended response to a post-purchase survey"` | `"express the same underlying sentiment or concern, regardless of phrasing or intensity"` | `"each cluster is presented to the research team as a distinct customer sentiment"` |
| Legal clauses | `"each text is a clause extracted from a commercial contract"` | `"raise the same legal issue or ground for dispute, regardless of how it is framed"` | `"each cluster is reviewed by a paralegal to assess contract risk by issue type"` |
| Medical notes | `"each text is a symptom description written by a patient during intake"` | `"describe the same symptom or clinical concern, regardless of how the patient characterises severity"` | `"each cluster informs a triage category used by the nursing team"` |
| App reviews | `"each text is a review left by a user on the app store"` | `"report on the same product problem or satisfaction driver, regardless of tone"` | `"each cluster becomes a card in the product team's feedback backlog"` |
| Code review | `"each text is a code review comment left by an engineer on a pull request"` | `"identify the same class of defect or code quality issue, regardless of which file is affected"` | `"each cluster defines a lint or review rule to automate"` |
