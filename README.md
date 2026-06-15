# cluster_judge

A reference-free **LLM-as-judge** evaluator that measures one thing: **how well-separated your text clusters are from their nearest neighbours**.

The headline output is a single number — **weighted distinctiveness** — plus a calibration gate that tells you how much to trust it. Everything is written to a self-contained HTML page.

---

## What it measures

For each cluster the judge is given `k` items, one of which is secretly planted from a neighbouring cluster. **Distinctiveness** is the corrected rate at which the LLM correctly isolates the planted item as a singleton. A cluster scores high when the LLM can reliably tell its items apart from its neighbours under your equivalence rule.

The headline metric is the **size-weighted fraction of clusters** that pass a 50% detection threshold — so large clusters that fail pull the score down more than small ones.

---

## Install

Python ≥ 3.10 and:

```
pip install numpy pandas scikit-learn
```

`tqdm` is optional (progress bars).

---

## Quickstart (offline demo)

```
python cluster_judge.py --demo
```

Runs on synthetic data with a mock judge, prints the score to stdout, and writes `demo_report.html`.

```python
from cluster_judge import make_demo_data, evaluate, write_html, Config

df, emb = make_demo_data()
cfg = Config(same_when="are the same kind of objection (the underlying concern)")
results = evaluate(df, emb, config=cfg, progress=False)
write_html(results, "report.html")
```

---

## Your own data

```python
from cluster_judge import evaluate, write_html, Config

cfg = Config(
    same_when="are the same kind of objection (the underlying concern), regardless of how it is answered",
    unit="each text is a customer objection raised on an outbound sales call",
    model="gateway",   # anything other than "mock" routes through your registered gateway
)

results = evaluate(
    data=df,              # DataFrame or path to csv/tsv/parquet/jsonl
    embeddings=emb,       # aligned numpy array, or use embedding_col=
    config=cfg,
)
write_html(results, "report.html")
```

Or from the CLI:

```
python cluster_judge.py \
  --data rows.parquet --embedding-col embedding \
  --same-when "are the same kind of objection" \
  --unit "each text is a customer objection" \
  --model gateway \
  --out report.html
```

---

## Connecting your judge

Register a single function that takes OpenAI-style messages and returns a string:

```python
from cluster_judge import use_genai, evaluate

def run_model_messages(messages: list[dict], json_mode: bool = True) -> str:
    # your gateway call here; return the raw string
    ...

use_genai(run_model_messages)
results = evaluate(data=df, embeddings=emb, config=cfg)
```

A module-level `run_model_messages` in `__main__` is auto-discovered, so registration is optional if you define one there.

---

## Output

`results` is a dict with three sections:

| key | contents |
|-----|----------|
| `kpi` | `weighted_distinct_rate`, `weighted_distinct_score`, `n_distinct`, `n_judged`, `threshold` |
| `calibration` | `Sp`, `Se`, `gamma`, `overall_pass`, per-check results, judge health |
| `clusters` | per-cluster `distinctiveness {score, lo, hi, n}`, `distinct` flag, `size`, `label` |

`write_html(results, path)` writes the self-contained HTML report.

---

## How the calibration gate works

Before measuring anything, the LLM is run on planted controls of known composition:

| control | tests |
|---------|-------|
| Pure (within-cluster kNN) | Same-kind items stay together → **Sp** (specificity) |
| Mixed (half A + half far B) | Different-kind items are separated → **Se** (sensitivity) |
| Junk (random word soup) | Nonsense is isolated (sanity check) |
| Far intruder (distant cluster) | Easy intruder is detected → **γ** (chance isolation rate) |

The gate PASS means the judge is reliable enough for the error correction to be meaningful. A FAIL means the numbers are not trustworthy — fix your `same_when` rule or gateway first.

Detection rates are corrected for chance isolation via the Rogan–Gladen formula:

```
corrected = (raw_detection − γ) / (1 − γ)
```

---

## Key tunables (`Config`)

| field | default | what it controls |
|-------|---------|------------------|
| `same_when` | **required** | the equivalence rule every judgment is made under |
| `unit` | `"each text is a short customer message"` | describes what each item is |
| `k_partition` | 10 | items per PARTITION call (including the one intruder) |
| `neighbor_m` | 3 | near-neighbour clusters used as intruder sources |
| `intruder_per_wave` | 4 | intruder draws per cluster per wave |
| `intruder_waves_max` | 3 | total measurement waves |
| `coverage_target` | 0.20 | fraction of each cluster sampled |
| `min_judgeable` | 5 | clusters smaller than this are skipped |
| `n_pure / n_mixed / n_junk / n_far` | 24/24/12/24 | calibration control counts |
| `max_llm_calls` | 20000 | hard budget ceiling |
| `workers` | 64 | gateway concurrency |
| `model` | `"mock"` | judge id; `"mock"` uses the offline judge |
| `mock_eps_split / mock_eps_join` | 0.0/0.0 | inject mock-judge noise to test the correction layer |
