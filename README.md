# cluster_judge

A reference-free **LLM-as-judge** evaluator that measures one thing: **how well-separated your text clusters are from their nearest neighbours**.

The headline output is a single number — **weighted distinctiveness** — plus a calibration gate that tells you how much to trust it.

---

## What it measures

For each cluster the judge is given `k` items, one of which is secretly planted from a neighbouring cluster. **Distinctiveness** is the corrected rate at which the LLM correctly isolates the planted item as a singleton. A cluster scores high when the LLM can reliably tell its items apart from its neighbours under your equivalence rule.

The headline metric is the **size-weighted fraction of clusters** that pass a 50% detection threshold — so large clusters that fail pull the score down more than small ones.

---

## Install

Python ≥ 3.10 and:

```
pip install numpy pandas
```

`tqdm` is optional (progress bars).

---

## Usage

```python
from cluster_judge import use_genai, evaluate, print_report, Config

# 1. Register your LLM gateway
@use_genai
def call_llm(messages: list[dict], json_mode: bool = True) -> str:
    # messages is a list of {"role": ..., "content": ...} dicts (OpenAI format)
    # return the raw response string
    response = your_client.chat(messages, ...)
    return response.text

# 2. Configure
cfg = Config(
    same_when="are the same kind of objection, regardless of how it is answered",
    unit="each text is a customer objection raised on an outbound sales call",
)

# 3. Evaluate
results = evaluate(
    data=df,          # DataFrame with 'text' and 'cluster_id' columns
                      # or path to .csv / .tsv / .parquet / .jsonl
    embeddings=emb,   # aligned numpy array (N × D float32), or use embedding_col=
    config=cfg,
)

# 4. Print results
print_report(results)
```

`print_report` writes a summary line and a per-cluster table to stdout. `results` is a plain dict you can inspect or serialise directly.

---

## Output dict

```python
results["kpi"]["weighted_distinct"]   # float 0–1, the headline number
results["kpi"]["n_distinct"]          # int, clusters that passed the threshold
results["kpi"]["n_judged"]            # int, clusters that were evaluated

results["calibration"]["gate_ok"]     # bool — trust the numbers if True
results["calibration"]["gamma"]       # float, chance-isolation rate γ
results["calibration"]["far_rate"]    # float, far-intruder detection rate

results["clusters"]                   # list of per-cluster dicts:
#   cluster_id, label, size,
#   score (corrected detection rate), lo, hi (95% CI bounds),
#   n_draws, distinct (bool)
```

---

## How the calibration gate works

Two kinds of planted controls are run before measurement:

| control | what it tests |
|---------|---------------|
| **Pure** (all items from one cluster) | Measures **γ** — the rate the LLM accidentally isolates a truly-same item. Used to correct all detection rates. |
| **Far** (home items + one from a distant cluster) | Verifies the judge can detect obvious intruders. Gate threshold: ≥ 70%. |

**PASS** means the correction is meaningful and scores are trustworthy.  
**FAIL** means the judge or `same_when` rule is too weak — fix it before reading scores.

Detection rates are corrected for chance isolation via the Rogan–Gladen formula:

```
corrected = (raw_detection − γ) / (1 − γ)
```

---

## Config reference

| field | default | what it controls |
|-------|---------|------------------|
| `same_when` | **required** | equivalence rule every judgment is made under |
| `unit` | `"each text is a short customer message"` | describes what each item is |
| `k_partition` | 10 | items per LLM call (k−1 home + 1 intruder) |
| `neighbor_m` | 3 | near-neighbour clusters used as intruder sources |
| `n_draws` | 24 | intruder draws per cluster |
| `coverage_target` | 0.5 | fraction of each cluster included in the sampling pool |
| `min_judgeable` | 5 | clusters smaller than this are skipped |
| `n_cal_pure` | 60 | pure calibration draws → γ |
| `n_cal_far` | 48 | far calibration draws → gate |
| `workers` | 64 | gateway concurrency |
| `max_retries` | 4 | retries per failed gateway call |
| `backoff` | 0.5 | initial retry delay in seconds (doubles each retry) |
| `item_chars` | 1024 | max characters per item shown to the LLM |
| `model` | `""` | passed through to `meta`; use in your gateway to route to a specific model |
| `seed` | 7 | numpy RNG seed for reproducible sampling |
