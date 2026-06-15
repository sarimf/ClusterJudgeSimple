# Writing Effective `same_when` Rules for cluster_judge

## What `same_when` Is

`same_when` is a single string that completes the sentence:

> Two items are the **same kind** when they **\_\_\_\_\_\_\_**.

It is the single most important input to cluster_judge. Every prompt the judge sees — for
homogeneity testing, intruder detection, label fit, item placement, and strategy mining — is
built around this string. It is inserted **verbatim** into every judge call:

```
# rule: two items are the SAME KIND when they {same_when}
```

The judge uses it literally. If the rule is vague, the judge will be inconsistent. If the
rule is sharp, the judge will be reliable and the Rogan-Gladen correction will do its job.

---

## The One Principle

**Define the underlying concern or type, not the surface form.**

The judge is explicitly instructed: *"Group by the rule, not by wording, tone, or shared
topic words."* Your `same_when` should embody that same stance. It should describe what
things *are*, not how they *sound*.

---

## Anatomy of a Good Rule

A good rule has three parts:

1. **The equivalence relation** — what must be true for two items to be the same kind
2. **The scope** — what variation is allowed (what does *not* matter)
3. **The domain anchor** — what the items are in your context

### Template

```
are [the same {TYPE} / about the same {CONCEPT}], regardless of [SURFACE VARIATION]
```

### The canonical example (from the codebase)

```python
same_when = "are the same kind of objection (the underlying concern), regardless of how it is answered"
```

- **Equivalence**: same kind of objection / underlying concern
- **Scope**: variation in how it is answered does not matter
- **Domain anchor**: objection (implied: customer objection on a sales call)

---

## What Makes a Rule Work Well

### Be explicit about what variation to ignore

If two items differ in tone, phrasing, or strategy but share the same root concern, say so:

```python
# Good — explicitly excludes surface variation
"are driven by the same underlying customer concern, regardless of phrasing or tone"

# Weak — leaves the judge to guess whether phrasing variation counts
"are about the same thing"
```

### Use your domain's natural categories

The rule should reflect how an expert in your domain would group items — not how a language
model would cluster text by topic overlap. Think: *"What would a domain expert say makes
two of these interchangeable?"*

```python
# Medical complaints: same symptom, regardless of severity wording
"describe the same symptom or bodily complaint, regardless of how severe the patient says it is"

# Support tickets: same root cause, regardless of product version
"stem from the same root cause or failure mode, regardless of which product version is mentioned"

# News articles: same event, regardless of outlet framing
"report on the same event or development, regardless of editorial framing or angle"
```

### Phrase it as a completion of "two items are the same kind when they…"

The string is grammatically inserted after "they". Test it by reading:

> "Two items are the same kind when they **[your rule]**."

It must read as a coherent relative clause.

---

## Common Mistakes

### Circular rules

A rule that just restates the category name gives the judge nothing to work with:

```python
# Bad — circular: "the same topic" tells the judge nothing
same_when = "are about the same topic"

# Better — what makes two topics the same?
same_when = "address the same underlying question or concern a user is trying to resolve"
```

### Too broad

If nearly everything is the same kind under the rule, homogeneity scores will be
artificially high and split detection will fail:

```python
# Too broad — almost all customer messages could qualify
same_when = "are sent by a customer who needs help"

# Appropriately scoped
same_when = "reflect the same type of billing problem (the root issue, not the specific amount)"
```

### Too narrow

If the rule is so precise that only exact duplicates qualify, you will get mostly singletons
and the pipeline cannot find meaningful clusters:

```python
# Too narrow — almost no two texts will match
same_when = "use exactly the same words to describe the same problem"

# Better — same concern at the right level of abstraction
same_when = "express the same objection to committing budget this quarter"
```

### Wording- or tone-based rules

The PARTITION prompt explicitly tells the judge to *ignore* wording and tone. If your rule
is based on those, you are fighting the prompt:

```python
# Bad — the judge is told to ignore this
same_when = "use similar emotional language"

# Good — the underlying concern
same_when = "reflect the same emotional need or concern, regardless of how strongly it is expressed"
```

### Forgetting to anchor to your domain

A rule that could apply to anything is harder for the judge to apply consistently:

```python
# Unanchored — could mean anything
same_when = "are essentially the same"

# Anchored — the judge knows what domain it is working in
same_when = "are the same type of legal objection raised during contract negotiation, regardless of the specific clause involved"
```

---

## Calibration Implications

cluster_judge measures judge quality on planted controls (pure clusters, mixed draws, junk
plants, intruders, swapped labels) and uses the results to calibrate Sp (specificity) and
Se (sensitivity). These corrections are only meaningful if the judge can reliably apply
your rule to the planted items.

**If the calibration gate fails, re-examine `same_when` first.** A failing gate usually
means one of:

1. The rule is ambiguous — the judge applies it inconsistently across draws
2. The rule is too coarse — the judge cannot distinguish near-miss intruders from genuine members
3. The rule is too fine — the judge splits pure clusters because of surface variation it is told to ignore

A sharp, well-anchored rule gives the calibration controls their best chance of passing.

---

## `unit` — What a Single Item Is

### What it does

`unit` is inserted as the first comment in **every** judge prompt:

```
# unit: {unit}
# rule: two items are the SAME KIND when they {same_when}
```

It appears in partition tasks (homogeneity testing, intruder detection) as well as label
fit, naming, and strategy prompts. Because it is in every prompt, including calibration
calls, a bad `unit` string affects judge consistency across the whole run — not just
presentation.

The default is `"each text is a short customer message"`. This is a safe fallback but
rarely the most informative choice.

### Template

```
each text is a [ITEM TYPE] [CONTEXT / SOURCE]
```

### The canonical example

```python
unit = "each text is a customer objection raised on an outbound sales call"
```

- **Item type**: customer objection
- **Context/source**: raised on an outbound sales call

### What makes a good `unit` string

**Be specific about what the item is, not what it contains.**

The judge needs to know the nature of the item to apply `same_when` consistently. "A
customer objection raised on a sales call" tells the judge more than "a short sentence" —
it sets the register, the source, and what variability to expect.

```python
# Too generic — the judge has no domain context
unit = "each text is a piece of text"

# Good — tells the judge what this item is in your world
unit = "each text is a one-sentence user complaint submitted via the app's feedback form"
```

**Keep it to one clause describing the item.** `unit` is not the place for instructions;
`same_when` carries the rule.

```python
# Bad — instructions belong in same_when, not unit
unit = "each text is a support ticket; group tickets by root cause"

# Good — pure description
unit = "each text is a support ticket submitted by a customer"
```

**Mention the source or context when it matters for interpretation.**

If two texts could mean different things depending on where they came from, the source
anchors the judge:

```python
# Ambiguous without source
unit = "each text is a user comment"

# Unambiguous — the source tells the judge what register to expect
unit = "each text is a comment left on a product listing by a verified purchaser"
```

### Common mistakes

| Mistake | Example | Fix |
|---|---|---|
| Too generic | `"each text is text"` | Add item type and source |
| Contains instructions | `"each text is a ticket; classify by category"` | Move instructions to `same_when` |
| Describes what items CONTAIN not what they ARE | `"each text contains a user opinion"` | `"each text is a user opinion"` |
| Plural form | `"these are customer complaints"` | Keep it singular: `"each text is a customer complaint"` |

### Patterns by domain

| Domain | Example `unit` |
|---|---|
| Sales objections | `"each text is a customer objection raised on an outbound sales call"` |
| Support tickets | `"each text is a support ticket submitted through the help portal"` |
| Survey responses | `"each text is an open-ended response to a post-purchase satisfaction survey"` |
| App reviews | `"each text is a review left by a user on the app store"` |
| Legal documents | `"each text is a clause extracted from a commercial contract"` |
| Medical notes | `"each text is a symptom description written by a patient during intake"` |
| News articles | `"each text is the headline and lead paragraph of a published news article"` |
| Code comments | `"each text is a code review comment left by an engineer on a pull request"` |

---

## `use_context` — What Clusters Are For

### What it does

`use_context` is added as a `# use:` comment in **label fit and strategy prompts only**:

```
# unit: {unit}
# rule: two items are the SAME KIND when they {same_when}
# use: {use_context}          ← only in fit_card and strategy prompts
```

It does **not** appear in partition prompts (homogeneity testing, intruder detection) and
therefore has **no effect on calibration, split detection, or mixing estimates**. It only
changes how the judge evaluates whether a label card fits the cluster members, and what
strategy proposals it generates.

`use_context` is optional. Omitting it is fine — the fit and strategy prompts simply run
without the extra context line. Add it when the downstream use of the clusters changes what
a "good" label looks like.

### Template

```
each cluster [HOW IT IS USED / WHO USES IT / WHAT HAPPENS WITH IT]
```

### The canonical example

```python
use_context = "each cluster is mined for the range of response strategies to that objection type"
```

This tells the judge that the purpose of a cluster is to anchor a playbook of responses —
so a label should be precise enough to distinguish response strategies, not just broadly
thematic.

### What makes a good `use_context` string

**Answer: what will a human or system do with this cluster?**

```python
# Vague — doesn't tell the judge anything about downstream use
use_context = "clusters are used internally"

# Clear — the judge knows what "useful" means for a label
use_context = "each cluster is used to route incoming tickets to the right support team"
```

**Match the granularity to the downstream task.**

If clusters will be used for coarse routing, a broad label is fine. If they will power a
detailed playbook or taxonomy, labels need to be precise. `use_context` shifts the judge's
calibration:

```python
# Coarse routing — broad labels are acceptable
use_context = "each cluster maps to a tier-1 support queue"

# Fine-grained playbook — precise labels are required
use_context = "each cluster is used to generate targeted email rebuttals for the sales team"
```

**Do not repeat `same_when` or describe what items are** (that is `unit`'s job):

```python
# Redundant with same_when
use_context = "each cluster contains objections of the same type"

# Redundant with unit
use_context = "each cluster contains customer messages"

# Good — describes the cluster's PURPOSE, not its contents
use_context = "each cluster informs the training data used to fine-tune a rebuttal model"
```

### When to omit it

You can omit `use_context` when:

- The cluster labels are self-explanatory from `same_when` alone
- You are only interested in split/merge detection, not label quality
- The downstream use is generic (exploration, QA, analytics) with no specific label requirements

### Patterns by domain

| Domain | Example `use_context` |
|---|---|
| Sales playbook | `"each cluster is mined for the range of response strategies to that objection type"` |
| Support routing | `"each cluster maps to a support team specialised in that failure mode"` |
| Product taxonomy | `"each cluster becomes a node in the product feedback taxonomy shown to the PM team"` |
| Content moderation | `"each cluster defines a violation category used to label training data"` |
| Survey analysis | `"each cluster is presented to the research team as a distinct customer sentiment"` |
| Legal review | `"each cluster is reviewed by a paralegal to assess contract risk by issue type"` |

---

## Putting It All Together

The three strings work as a team. A concrete example with all three set:

```python
cfg = cj.Config(
    unit        = "each text is a customer objection raised on an outbound sales call",
    same_when   = "are the same kind of objection (the underlying concern), regardless of how it is answered",
    use_context = "each cluster is mined for the range of response strategies to that objection type",
    model       = "mock",
)
```

| Parameter | Controls | Affects |
|---|---|---|
| `unit` | What a single item **is** | All prompts (calibration, partition, fit, strategy) |
| `same_when` | What makes two items the **same kind** | All prompts |
| `use_context` | What clusters are **for** | Label fit and strategy prompts only |

Change `unit` if the judge seems confused about the nature of the data. Change `same_when`
if calibration fails or splits are wrong. Change `use_context` if labels are right in kind
but wrong in granularity for your downstream task.

---

## Quick Reference: Patterns by Domain

| Domain | `unit` | `same_when` | `use_context` |
|---|---|---|---|
| Sales objections | `"each text is a customer objection raised on a sales call"` | `"are the same kind of objection (the underlying concern), regardless of how it is answered"` | `"each cluster is mined for rebuttal strategies to that objection type"` |
| Support tickets | `"each text is a support ticket submitted through the help portal"` | `"stem from the same root cause or failure mode, regardless of product version"` | `"each cluster maps to a support team specialised in that failure mode"` |
| Survey open-ends | `"each text is an open-ended response to a post-purchase survey"` | `"express the same underlying sentiment or concern, regardless of phrasing or intensity"` | `"each cluster is presented to the research team as a distinct customer sentiment"` |
| Legal filings | `"each text is a clause extracted from a commercial contract"` | `"raise the same legal issue or ground for dispute, regardless of how it is framed"` | `"each cluster is reviewed by a paralegal to assess contract risk by issue type"` |
| Medical notes | `"each text is a symptom description written by a patient during intake"` | `"describe the same symptom or clinical concern, regardless of how the patient characterises its severity"` | `"each cluster informs a triage category used by the nursing team"` |
| App reviews | `"each text is a review left by a user on the app store"` | `"report on the same product problem or satisfaction driver, regardless of tone"` | `"each cluster becomes a card in the product team's feedback backlog"` |
| Code review | `"each text is a code review comment left by an engineer on a pull request"` | `"identify the same class of defect or code quality issue, regardless of which file is affected"` | `"each cluster defines a lint or review rule to automate"` |

---

## Numeric Calibration Levers

Beyond the three framing strings, `Config` has ~30 numeric knobs. Most can be left at
their defaults. This section explains the ones that matter, grouped by what they control.

---

### 1. Measurement depth — how much judging to do

These are the most impactful dials after `same_when`. They govern how many partition draws
the pipeline makes and therefore how precise the mixing/detection estimates are.

| Parameter | Default | What it does |
|---|---|---|
| `coverage_target` | `0.20` | Fraction of items in each cluster that must appear in at least one partition draw before the wave loop stops. The main "effort" knob. |
| `coverage_ceiling` | `0.40` | Hard upper cap on coverage — the loop stops here even if `coverage_target` is not yet met. Prevents runaway costs on huge clusters. |
| `waves_max` | `6` | Maximum homogeneity waves per cluster. Each wave adds `draws_per_wave` partition draws. |
| `draws_per_wave` | `3` | Partition draws per cluster per homogeneity wave. |
| `intruder_waves_max` | `3` | Maximum intruder-detection waves. |
| `intruder_per_wave` | `4` | Intruder-detection cycles per wave (cycles near neighbours + far). |
| `resolution_waves` | `4` | Extra homogeneity waves added for clusters that are confirmed splits. More waves → more precise sub-class prevalence estimates. |
| `resolution_target_seen` | `120` | Target number of distinct items to have seen before stopping resolution waves. |

**Rule of thumb:** `coverage_target=0.20` is suitable for exploration. Raise to `0.50`–`0.80`
for production runs where you need tight confidence intervals. `coverage_ceiling` should be
at most `2 × coverage_target` to avoid wasting budget on clusters where early waves already
resolved the verdict.

```python
# Quick smoke test — minimal calls
cfg = cj.Config(same_when=..., coverage_target=0.10, coverage_ceiling=0.20,
                waves_max=2, intruder_waves_max=1)

# Production — tight estimates
cfg = cj.Config(same_when=..., coverage_target=0.60, coverage_ceiling=0.80)
```

---

### 2. Partition design — what each judge call measures

| Parameter | Default | What it does |
|---|---|---|
| `k_partition` | `10` | Number of items shuffled into each PARTITION prompt. Larger k gives more co-occurrence data per call but is harder for the judge. Must be ≥ 3. |
| `min_judgeable` | `5` | Clusters smaller than this are reported but not judged (no partition draws run). |
| `min_pool` | `6` | Minimum pool size for sampling; clusters with fewer items are skipped entirely. |
| `neighbor_m` | `3` | HNSW neighbourhood size used when sampling near-neighbour intruders. Smaller = tighter neighbourhood = harder intruder test. |
| `n_hubs` | `4` | Farthest-point probes that seed the co-grouping graph with long-range edges. Increase for very heterogeneous clusters. |
| `item_chars` | `240` | Characters of each item text shown to the judge. Truncate further if your model has a very small context or items are verbose. |

For most datasets `k_partition=10` is good. Drop to `7`–`8` if your judge struggles with
long items or if `k_partition × item_chars` approaches the model's effective context window.

---

### 3. Planted calibration controls

The pipeline measures judge quality by planting known-answer controls into every run. The
`n_*` parameters control how many of each type are planted. Reducing them cuts cost but
weakens the calibration gate.

| Parameter | Default | What it controls |
|---|---|---|
| `n_pure` | `24` | Pure kNN-ball partitions (all items from same cluster). Used to estimate **specificity** (Sp) — how often the judge correctly keeps same-kind items together. |
| `n_mixed` | `24` | 50/50 mixed-cluster partitions. Used to estimate **sensitivity** (Se) — how often the judge correctly splits cross-kind items. |
| `n_junk` | `12` | Partitions containing random items from many clusters. Sanity check that the judge can spot obvious outliers. |
| `n_far` | `24` | Partitions containing a planted intruder drawn from a far-away cluster (by embedding distance). Checks that the judge catches obvious intruders. |
| `n_labelswap` | `10` | FIT calls where the label card belongs to a *different* cluster. Calibrates the FIT judge's ability to reject wrong labels. |
| `dup_rate` | `0.05` | Fraction of partition draws that are silent duplicates of earlier draws. Duplicate disagreement rate estimates within-judge noise. |

**Minimum viable calibration:** `n_pure=12, n_mixed=12, n_junk=6, n_far=12, n_labelswap=6`
for a quick run. Below these values the Rogan-Gladen correction becomes unreliable.

**The five calibration gate checks** are:
1. Pure-Sp passes (judge keeps pure items together)
2. Mixed-Se passes (judge splits mixed draws)
3. Junk detection passes
4. Far-intruder detection passes
5. Label-swap rejection passes

If any single check fails, `overall_pass` is `False` and every downstream number is
flagged as untrustworthy. Increasing the corresponding `n_*` counter gives the gate more
statistical power to detect a borderline judge.

---

### 4. Decision threshold

| Parameter | Default | What it does |
|---|---|---|
| `tau_mix` | `0.15` | Corrected-mixing tolerance. A cluster is declared `one_class` when its Rogan-Gladen-corrected mixing rate falls below this value. Raise it to accept more heterogeneity; lower it to demand purer clusters. |
| `z` | `1.96` | z-score used for Wilson confidence intervals throughout (default = 95%). Change to `1.645` for 90% CIs (fewer calls needed to pass) or `2.576` for 99% (stricter). |

`tau_mix=0.15` means "up to 15% cross-kind items is acceptable for one-class." If your
domain has naturally noisy text that even humans would disagree on, raising to `0.20`–`0.25`
avoids over-splitting. If you need very clean clusters, lower to `0.08`–`0.10`.

---

### 5. FIT — label evaluation

FIT tasks ask the judge to evaluate whether the existing cluster label card fits the
members. They run after all partition waves finish.

| Parameter | Default | What it does |
|---|---|---|
| `fit_items` | `30` | Items sampled per cluster for the label-fit judgment. Fewer items = faster but noisier verdict. |
| `fit_replicates` | `3` | Extra FIT calls when the first verdict is not `accurate`. Majority vote across replicates. |
| `do_fit_choice` | `False` | Enable item-placement task: the judge picks which cluster card best fits each sampled item. Adds calls but sharpens label distinctiveness. |
| `fit_choice_n` | `8` | Items shown per fit-choice call when `do_fit_choice=True`. |

FIT results appear in `ClusterResult["fit"]` and drive the `render_report` label-quality
column. They never move the mixing or split scores.

---

### 6. Presentation switches

These control whether the pipeline generates extra outputs after scoring is complete. They
add LLM calls but do not affect any numeric estimate.

| Parameter | Default | What it does |
|---|---|---|
| `do_strategy` | `True` | Generate a list of response strategies per cluster (calls the judge once per cluster). Set `False` to skip. |
| `strategy_n` | `20` | Items shown to the strategy generation call. |
| `do_taxonomy` | `True` | Generate a cross-cluster taxonomy after all clusters are scored. Set `False` to skip. |
| `name_exemplars` | `8` | Exemplars shown to the naming call when proposing a cluster label. |
| `max_bands` | `5` | Maximum size-bands in the report summary. |
| `gvf_target` | `0.90` | Goodness-of-variance-fit target for choosing band boundaries. |

Turn both `do_strategy=False, do_taxonomy=False` when you only care about split/merge
detection and want to minimise calls.

---

### 7. Budget & infrastructure

| Parameter | Default | What it does |
|---|---|---|
| `max_llm_calls` | `20000` | Hard budget cap. The pipeline stops issuing new calls once this is reached. Clusters whose waves were cut short are flagged `under_sampled` in the report. |
| `workers` | `64` | Thread pool size for concurrent judge calls. Match to your API's rate limit; set `1` for fully sequential (useful for deterministic tests). |
| `max_retries` | `4` | Retries per failed judge call with exponential backoff. |
| `backoff_base` | `0.5` | Base delay in seconds between retries (`delay = backoff_base × 2^attempt`). |
| `max_empty_rate` | `0.25` | Judge health gate: if the fraction of calls returning empty responses exceeds this, `overall_pass` is forced `False`. Lower to be more sensitive to infrastructure failures. |

---

### Preset configurations

Three ready-to-use configurations for common situations:

```python
# Fast exploration — ~300 calls for a 10-cluster dataset
FAST = dict(coverage_target=0.10, coverage_ceiling=0.20,
            waves_max=2, intruder_waves_max=1,
            n_pure=12, n_mixed=12, n_junk=6, n_far=12, n_labelswap=6,
            do_strategy=False, do_taxonomy=False,
            fit_items=15, fit_replicates=1)

# Balanced — default-ish, reasonable for a first real run
BALANCED = dict(coverage_target=0.20, coverage_ceiling=0.40)   # all other defaults

# Thorough — production-quality estimates
THOROUGH = dict(coverage_target=0.60, coverage_ceiling=0.80,
                waves_max=8, intruder_waves_max=4, resolution_waves=6,
                n_pure=36, n_mixed=36, n_junk=18, n_far=36, n_labelswap=16,
                fit_items=50, fit_replicates=4,
                z=2.576)                                          # 99% CI
```

---

## Testing Your Rule

Before running a full evaluation, test your rule with the offline demo:

```bash
python cluster_judge.py --demo --out cj_test
```

Then read the **CALIBRATION GATE** section at the top of `cj_test/report.txt`. If all five
checks pass, the judge is reliably applying a rule of that shape. Substitute your real data
and rule once the gate is green.

For iterative development, run with low coverage and the mock judge first (no credentials,
no cost):

```python
cfg = cj.Config(
    same_when       = "your rule here",
    model           = "mock",
    coverage_target = 0.20,
    workers         = 4,
)
R = cj.evaluate(df, emb, labels, cfg, progress=True)
print(cj.render_report(R))
```

Check that:

- The calibration gate passes (`overall_pass: True`)
- Pure clusters show `one_class: True`
- Known-mixed clusters show `split: True` (if any exist in your data)
- The proposed labels from FIT make sense under your rule

If the gate fails on real data but the mock passes, the problem is the rule's interaction
with your specific text — the judge cannot reliably distinguish the planted controls given
how your items are written. Tighten the scope clause or add a domain anchor.

---

## End-to-End Pipeline Walk-Through

A single call to `evaluate(df, emb, labels, cfg)` runs eight phases in sequence. Every
knob is annotated at the point it first fires.

---

### Phase 0 — Setup

- **`seed`**: seeds all RNGs before anything touches data.
- **`min_pool`** (6): clusters with fewer items than this are skipped entirely.
- **`min_judgeable`** (5): clusters with ≥ `min_pool` but < `min_judgeable` items are *reported* but receive no partition draws.
- **`max_items_per_cluster`** (6000): very large clusters are hard-capped; items beyond the cap are never shown to the judge.
- **`micro_k`** (8): each cluster's items are grouped into `micro_k` micro-modes by k-means on embeddings. Partition draws sample from these modes so coverage is not biased toward the densest region.
- **`neighbor_m`** (3) + **`n_hubs`** (4): an HNSW approximate-nearest-neighbour index is built over all embeddings. `neighbor_m` controls neighbourhood size for intruder sourcing. `n_hubs` seeds the co-grouping graph with `n_hubs` farthest-point probes per cluster, giving the graph long-range edges that catch heterogeneity far from the centre.

---

### Phase 1 — Calibration

Before any real cluster is judged, the pipeline plants known-answer controls and runs them
through the judge to measure its quality and enable Rogan-Gladen error correction.

**Planting controls:**

- **`n_pure`** (24): partition tasks where all items come from the *same* cluster. The judge should keep them together. Calibrates **specificity (Sp)**.
- **`n_mixed`** (24): tasks split 50/50 between two clusters. The judge should separate them. Calibrates **sensitivity (Se)**.
- **`n_junk`** (12): items drawn randomly from many clusters. Sanity-checks gross-outlier detection.
- **`n_far`** (24): a single planted intruder from a far-away cluster. Checks the judge catches an obvious stranger.
- **`n_labelswap`** (10): FIT calls where the label card belongs to a *different* cluster. Calibrates the FIT judge's label-rejection ability.
- **`dup_rate`** (0.05): a fraction of all draws are silent duplicates of earlier draws. Duplicate-disagreement rate estimates within-judge noise.

Every prompt is assembled with:

- **`unit`** → `# unit: {unit}` header line in every prompt
- **`same_when`** → `# rule: two items are the SAME KIND when they {same_when}`
- **`k_partition`** (10): items per partition task
- **`item_chars`** (240): each item text is truncated to this many characters

**Running the judge:**

- **`workers`** (64): concurrent threads
- **`model`**, **`temperature`** (0.0): which judge, at what temperature
- **`max_retries`** (4) + **`backoff_base`** (0.5): failed calls retry with delay `backoff_base × 2^attempt` seconds

**Judge health gate:**

- **`max_empty_rate`** (0.25): if `n_empty / n_calls` exceeds this, `judge_health["ok"]` is `False` and `overall_pass` is forced `False`.

**Sp / Se estimation:**

- **`z`** (1.96): Wilson confidence intervals at this z-score (95% CI). Used for every rate throughout the pipeline.

Five gate checks run (pure-Sp, mixed-Se, junk, far-intruder, label-swap). Any failure sets `overall_pass = False`.

---

### Phase 2 — Measurement Waves

For each judgeable cluster the pipeline runs a wave loop, issuing homogeneity and intruder draws until coverage is satisfied or the budget is hit.

**Homogeneity waves:**

- **`draws_per_wave`** (3): partition draws per cluster per wave
- **`waves_max`** (6): maximum waves; stops early if coverage is met
- **`coverage_target`** (0.20): stop when this fraction of a cluster's items have appeared in at least one draw
- **`coverage_ceiling`** (0.40): hard upper cap regardless of whether `coverage_target` is met
- **`k_partition`** (10): items per draw; also the unit of co-occurrence data added to the co-grouping graph

Each draw result adds item-pair co-grouping observations to the graph: pairs in the same
judge-returned group get an edge increment; pairs separated get nothing.

**Intruder waves:**

- **`intruder_per_wave`** (4): intruder cycles per wave
- **`intruder_waves_max`** (3): maximum intruder waves

**Budget enforcement:**

- **`max_llm_calls`** (20000): when reached, `_fair_trim` round-robins remaining units by cluster ID so every cluster loses an equal share rather than tail clusters losing everything.

---

### Phase 3 — Per-Cluster Estimation

After all waves, for each cluster:

1. **Raw mixing rate** (`m_raw`): fraction of near-neighbour cross-kind item-pair observations where the judge co-grouped them.
2. **Rogan-Gladen correction**: `mixing_corrected = (m_raw − (1 − Sp)) / (Sp + Se − 1)`, clamped to [0, 1]. If calibration produced no usable Sp/Se (`ok_corr=False`), the raw rate passes through unchanged.
3. **Verdict:**
   - **`tau_mix`** (0.15): `one_class = True` when `mixing_corrected < tau_mix`; `split = True` when the lower Wilson bound > `tau_mix`.
   - **`z`** governs the Wilson intervals used in the comparison.

---

### Phase 4 — Resolution Waves

For clusters flagged `split`, additional waves refine sub-class prevalence and resolve the
co-grouping graph into components.

- **`resolution_waves`** (4): extra waves added for confirmed splits
- **`resolution_target_seen`** (120): the loop also stops when this many distinct items have been seen
- Same `draws_per_wave`, `k_partition`, `coverage_target/ceiling` mechanics apply

`_components_from_pairs` then partitions the split cluster's items into sub-components via
LR-gated agglomeration with draw-replication gating (prevents single-draw false joins).

---

### Phase 5 — Merge Detection

Items co-grouped *across* different clusters (because a partition draw spanned cluster
boundaries) add cross-cluster edges to the co-grouping graph. Strongly co-grouped
cross-cluster components produce merge candidates in `merge_groups`.

- **`tau_mix`** and **`z`** gate which cross-cluster edges are strong enough to assert a merge — same threshold, same Wilson-interval machinery.

---

### Phase 6 — FIT (Label Evaluation)

The judge evaluates whether each cluster's label card fits its members.

- **`fit_items`** (30): items sampled per cluster for the fit judgment
- **`fit_replicates`** (3): if the first verdict is not `accurate`, this many additional calls are made and the majority verdict wins
- **`do_fit_choice`** (False): if enabled, the judge also picks which cluster card best fits each of `fit_choice_n` (8) items — a placement/distinctiveness task
- **`use_context`** appears *only here* (and in strategy prompts) as `# use: {use_context}` — the first phase it matters
- **`n_labelswap`** calibration informs the FIT judge's specificity estimate for label rejection

FIT results appear in `ClusterResult["fit"]` but never move any mixing or split score.

---

### Phase 7 — Presentation

Naming, strategy, and taxonomy calls run after all numeric scores are final and cannot
change any estimate.

- **`name_exemplars`** (8): exemplars shown per cluster to the naming call
- **`do_strategy`** (True) / **`strategy_n`** (20): if enabled, the judge generates response strategies per cluster from `strategy_n` items; **`use_context`** appears in this prompt
- **`do_taxonomy`** (True): a single call assembles a cross-cluster taxonomy from all cluster labels
- **`unit`**, **`same_when`**, **`use_context`** all appear in naming and strategy prompts

---

### Phase 8 — Assembly and Banding

- Final `split`, `one_class`, `mixing`, `detection` verdicts are written into `ClusterResult` TypedDicts.
- **`max_bands`** (5) / **`gvf_target`** (0.90): clusters are grouped into size bands by Jenks natural-breaks (goodness-of-variance-fit ≥ `gvf_target`, capped at `max_bands`). The report summary aggregates KPIs by band.
- Merge groups are assembled from cross-cluster components.
- `Results` TypedDict is returned.

---

### Knob-to-Phase Map

```
Phase               Key knobs
────────────────────────────────────────────────────────────────────
0  Setup            seed, min_pool, min_judgeable, max_items_per_cluster,
                    micro_k, neighbor_m, n_hubs
1  Calibration      n_pure, n_mixed, n_junk, n_far, n_labelswap, dup_rate,
                    unit, same_when, k_partition, item_chars,
                    workers, model, temperature, max_retries, backoff_base,
                    max_empty_rate, z
2  Measurement      draws_per_wave, waves_max, coverage_target, coverage_ceiling,
                    intruder_per_wave, intruder_waves_max,
                    k_partition, max_llm_calls
3  Estimation       tau_mix, z  (+ Sp/Se from calibration)
4  Resolution       resolution_waves, resolution_target_seen
5  Merge detection  tau_mix, z
6  FIT              fit_items, fit_replicates, do_fit_choice, fit_choice_n,
                    use_context
7  Presentation     name_exemplars, do_strategy, strategy_n, do_taxonomy,
                    unit, same_when, use_context
8  Banding          max_bands, gvf_target
```
