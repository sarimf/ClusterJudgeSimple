"""cluster_judge — weighted distinctiveness evaluator for text clusterings.

Plants one item from a neighbouring cluster among k−1 home items and asks
the LLM to sort them by kind. Distinctiveness = corrected rate the planted
item is isolated. Headline: size-weighted fraction of clusters that pass.
Output: HTML report.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

log = logging.getLogger("cluster_judge")

_gateway_fn: Optional[Callable] = None


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    same_when: str = ""
    unit: str = "each text is a short customer message"
    k_partition: int = 10      # items per PARTITION call (includes 1 intruder)
    neighbor_m: int = 3        # near-neighbour clusters as intruder sources
    n_draws: int = 12          # intruder draws per cluster
    coverage_target: float = 0.25
    min_judgeable: int = 5
    n_cal_pure: int = 30       # pure calibration draws → γ
    n_cal_far: int = 24        # far calibration draws → gate
    workers: int = 64
    max_retries: int = 4
    backoff: float = 0.5
    item_chars: int = 1024
    z: float = 1.96
    model: str = ""
    seed: int = 7

    def __post_init__(self):
        if not self.same_when:
            raise ValueError("same_when is required")


def use_genai(fn: Callable) -> Callable:
    """Register gateway: fn(messages, json_mode=True) -> str."""
    global _gateway_fn
    _gateway_fn = fn
    return fn


# ── LLM client ────────────────────────────────────────────────────────────────

class Client:
    def __init__(self, cfg: Config):
        if _gateway_fn is None:
            raise RuntimeError("No gateway registered — call use_genai(fn) first.")
        self.cfg = cfg
        self._lock = threading.Lock()
        self.n_calls = 0

    def call(self, items: List[str]) -> Optional[dict]:
        with self._lock:
            self.n_calls += 1
        prompt = (
            f"# unit: {self.cfg.unit}\n"
            f"# rule: two items are the SAME KIND when they {self.cfg.same_when}\n"
            "Sort the items into groups of the same kind under the rule. Singletons are allowed. "
            "Use every index exactly once.\nITEMS:\n"
            + "\n".join(f"{i+1}. {t}" for i, t in enumerate(items))
            + '\nReturn STRICT JSON: {"groups": [[1,4],[2],[3,5]]}'
        )
        msgs = [{"role": "system", "content": "Reply with STRICT JSON only."},
                {"role": "user",   "content": prompt}]
        delay = self.cfg.backoff
        for att in range(self.cfg.max_retries + 1):
            try:
                v = _parse_json(_gateway_fn(msgs, json_mode=True))
                if v:
                    return v
            except Exception as e:
                if att == self.cfg.max_retries:
                    log.warning("call failed: %s", e)
            time.sleep(delay)
            delay *= 2
        return None


_JSON_RE = re.compile(r"\{.*\}", re.S)


def _parse_json(s) -> Optional[dict]:
    if isinstance(s, dict):
        return s
    if not isinstance(s, str):
        return None
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", s)
    try:
        return json.loads(s)
    except Exception:
        m = _JSON_RE.search(s)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return None


def _valid_groups(v: Optional[dict], k: int) -> Optional[List[List[int]]]:
    if not v:
        return None
    try:
        gs = [[int(x) for x in g] for g in v.get("groups", [])]
    except Exception:
        return None
    return gs if sorted(x for g in gs for x in g) == list(range(1, k + 1)) else None


# ── Dispatch ──────────────────────────────────────────────────────────────────

def _dispatch(tasks: List[List[str]], client: Client, cfg: Config,
              desc: str = "") -> List[Optional[dict]]:
    """Run item lists in parallel; return aligned list of results."""
    results: List[Optional[dict]] = [None] * len(tasks)
    if not tasks:
        return results
    bar = tqdm(total=len(tasks), desc=desc) if tqdm else None
    with ThreadPoolExecutor(max_workers=min(cfg.workers, len(tasks))) as ex:
        futs = {ex.submit(lambda i=i: (i, client.call(tasks[i]))): None
                for i in range(len(tasks))}
        for f in as_completed(futs):
            i, res = f.result()
            results[i] = res
            if bar:
                bar.update(1)
    if bar:
        bar.close()
    return results


# ── Data helpers ──────────────────────────────────────────────────────────────

def _load_df(path: str) -> pd.DataFrame:
    p = path.lower()
    if p.endswith(".parquet"):  return pd.read_parquet(path)
    if p.endswith(".tsv"):      return pd.read_csv(path, sep="\t")
    if p.endswith(".jsonl"):    return pd.read_json(path, lines=True)
    if p.endswith(".json"):     return pd.read_json(path)
    return pd.read_csv(path)


def _coerce(data, embeddings=None, *, text_col="text", cluster_col="cluster_id",
            label_col="label", embedding_col=None):
    df = data if isinstance(data, pd.DataFrame) else _load_df(data)
    df = df.rename(columns={k: v for k, v in {text_col: "text", cluster_col: "cluster_id"}.items()
                             if k in df.columns and k != v}).copy().reset_index(drop=True)
    if "text" not in df.columns or "cluster_id" not in df.columns:
        raise ValueError("need text and cluster_id columns")
    labels = {str(cid): str(g[label_col].iloc[0]) if label_col in df.columns else str(cid)
              for cid, g in df.groupby("cluster_id")}
    if embeddings is not None:
        emb = np.asarray(embeddings, dtype=np.float32)
    elif embedding_col and embedding_col in df.columns:
        emb = np.vstack(df[embedding_col]).astype(np.float32)
    else:
        raise ValueError("embeddings required: pass embeddings= or embedding_col=")
    if len(emb) != len(df):
        raise ValueError(f"embeddings length {len(emb)} != rows {len(df)}")
    return df, emb, labels


def _normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (x / n).astype(np.float32)


def _sample(idxs: np.ndarray, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    target = max(cfg.min_judgeable, int(round(len(idxs) * cfg.coverage_target)))
    return idxs if len(idxs) <= target else rng.choice(idxs, size=target, replace=False)


# ── Math ──────────────────────────────────────────────────────────────────────

def _wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 1.0
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def _rg(raw: float, gamma: float) -> float:
    """Rogan-Gladen: correct raw detection rate for chance isolation γ."""
    return min(1.0, max(0.0, (raw - gamma) / (1 - gamma))) if gamma < 1 else raw


# ── Main evaluate ─────────────────────────────────────────────────────────────

def evaluate(data, embeddings=None, *, same_when: str = "", unit: str = "",
             model: str = "", config: Optional[Config] = None,
             text_col="text", cluster_col="cluster_id",
             label_col="label", embedding_col=None) -> dict:

    cfg = config or Config(same_when=same_when or "are the same kind",
                           unit=unit or "each text is a short customer message",
                           model=model)
    if same_when: cfg.same_when = same_when
    if unit:      cfg.unit = unit
    if not cfg.same_when:
        raise ValueError("same_when is required")

    df, emb, labels = _coerce(data, embeddings, text_col=text_col, cluster_col=cluster_col,
                               label_col=label_col, embedding_col=embedding_col)
    rng = np.random.default_rng(cfg.seed)

    cats   = pd.Categorical(df["cluster_id"].astype(str))
    codes  = np.asarray(cats.codes)
    cid_of = {i: str(c) for i, c in enumerate(cats.categories)}
    K      = len(cats.categories)
    texts  = df["text"].astype(str).to_numpy()
    emb_n  = _normalize(emb)

    # cluster sizes + coverage pools
    sizes = {cid_of[c]: int((codes == c).sum()) for c in range(K)}
    pools = {cid_of[c]: _sample(np.where(codes == c)[0], cfg, rng) for c in range(K)}
    judged = {cid: p for cid, p in pools.items() if len(p) >= cfg.min_judgeable}
    log.info("%d clusters total, %d judged", K, len(judged))

    # centroids → near / far neighbour tables (by cluster code)
    cent = _normalize(np.vstack([emb_n[codes == c].mean(0) for c in range(K)]))
    sims = cent @ cent.T
    np.fill_diagonal(sims, -2.0)                              # exclude self
    nb      = np.argsort(-sims, axis=1)                       # most→least similar
    near_nb = nb[:, :min(cfg.neighbor_m, K - 1)]             # top-m near (self excluded)
    far_nb  = nb[:, max(1, K // 2):-1]                       # far half (self is last at -2)
    code_of = {cid_of[c]: c for c in range(K)}

    def clip(i): return texts[i][: cfg.item_chars]

    def plant_call(home_pool, src_pool):
        """k-1 home items + 1 planted intruder; return (items, plant_pos)."""
        k = cfg.k_partition
        home = rng.choice(home_pool, size=min(k - 1, len(home_pool)), replace=False).tolist()
        p    = int(rng.choice(src_pool))
        pos  = int(rng.integers(len(home) + 1))
        ids  = home[:pos] + [p] + home[pos:]
        return [clip(i) for i in ids], pos + 1   # plant_pos is 1-indexed

    # ── Calibration ───────────────────────────────────────────────────────────
    # pure: k items from same cluster → measure γ (chance isolation rate)
    # far:  k-1 home + 1 far item    → gate check (must be detected ≥ 70%)
    judged_list = list(judged)
    cal_tasks: List[List[str]] = []
    cal_meta:  List[dict] = []

    for _ in range(cfg.n_cal_pure):
        cid = judged_list[int(rng.integers(len(judged_list)))]
        if len(judged[cid]) < cfg.k_partition:
            continue
        idxs = rng.choice(judged[cid], size=cfg.k_partition, replace=False).tolist()
        cal_tasks.append([clip(i) for i in idxs])
        cal_meta.append({"ctx": "pure"})

    for _ in range(cfg.n_cal_far):
        cid = judged_list[int(rng.integers(len(judged_list)))]
        if len(judged[cid]) < cfg.k_partition - 1 or not len(far_nb[code_of[cid]]):
            continue
        src = cid_of[int(rng.choice(far_nb[code_of[cid]]))]
        if not len(pools[src]):
            continue
        items, pp = plant_call(judged[cid], pools[src])
        cal_tasks.append(items)
        cal_meta.append({"ctx": "far", "pp": pp})

    log.info("calibration: %d calls…", len(cal_tasks))
    cal_res = _dispatch(cal_tasks, Client(cfg), cfg, desc="calibrating")

    iso_d = iso_n = far_d = far_n = 0
    for meta, v, task in zip(cal_meta, cal_res, cal_tasks):
        gs = _valid_groups(v, len(task))
        if gs is None:
            continue
        if meta["ctx"] == "pure":
            iso_d += sum(1 for g in gs if len(g) == 1)
            iso_n += len(task)
        else:
            far_d += int(any(g == [meta["pp"]] for g in gs))
            far_n += 1

    gamma    = iso_d / iso_n if iso_n else 0.02
    far_rate = far_d / far_n if far_n else 0.0
    gate_ok  = far_rate >= 0.7
    cal = {"gamma": round(gamma, 3), "far_rate": round(far_rate, 3), "gate_ok": gate_ok}
    log.info("calibration done: γ=%.3f  far_rate=%.2f  gate=%s",
             gamma, far_rate, "PASS" if gate_ok else "FAIL")

    # ── Measurement ───────────────────────────────────────────────────────────
    client = Client(cfg)
    meas_tasks: List[List[str]] = []
    meas_cids:  List[str] = []
    meas_ppos:  List[int] = []

    for cid, pool in judged.items():
        if len(pool) < cfg.k_partition - 1:
            continue
        nears = [cid_of[int(x)] for x in near_nb[code_of[cid]]]
        nears = [c for c in nears if len(pools.get(c, [])) > 0]
        if not nears:
            continue
        for _ in range(cfg.n_draws):
            src = nears[int(rng.integers(len(nears)))]
            items, pp = plant_call(pool, pools[src])
            meas_tasks.append(items)
            meas_cids.append(cid)
            meas_ppos.append(pp)

    log.info("measurement: %d calls across %d clusters…", len(meas_tasks), len(judged))
    meas_res = _dispatch(meas_tasks, client, cfg, desc="measuring")

    det = {cid: [0, 0] for cid in judged}   # [detected, total]
    for cid, pp, v, task in zip(meas_cids, meas_ppos, meas_res, meas_tasks):
        gs = _valid_groups(v, len(task))
        if gs is None:
            continue
        det[cid][0] += int(any(g == [pp] for g in gs))
        det[cid][1] += 1

    # ── Assembly ──────────────────────────────────────────────────────────────
    THRESHOLD = 0.5
    clusters: List[dict] = []
    w_num = w_den = 0.0

    for cid in sorted(judged, key=lambda c: -sizes[c]):
        d, n = det[cid]
        raw, lo, hi = _wilson(d, n, cfg.z)
        score    = round(_rg(raw, gamma), 3)
        distinct = score >= THRESHOLD
        size     = sizes[cid]
        clusters.append({"cluster_id": cid, "label": labels.get(cid, cid), "size": size,
                         "score": score, "lo": round(_rg(lo, gamma), 3),
                         "hi": round(_rg(hi, gamma), 3), "n_draws": n, "distinct": distinct})
        w_num += size * float(distinct)
        w_den += size

    return {
        "kpi": {"weighted_distinct": round(w_num / w_den, 3) if w_den else None,
                "threshold": THRESHOLD,
                "n_distinct": sum(c["distinct"] for c in clusters),
                "n_judged": len(clusters)},
        "calibration": cal,
        "meta": {"model": cfg.model, "n_texts": len(df), "n_clusters": K,
                 "n_llm_calls": client.n_calls, "k_partition": cfg.k_partition,
                 "same_when": cfg.same_when, "unit": cfg.unit},
        "clusters": clusters,
    }


# ── HTML report ───────────────────────────────────────────────────────────────

def render_html(results: dict) -> str:
    kpi = results.get("kpi", {})
    cal = results.get("calibration", {})
    meta = results.get("meta", {})
    clusters = results.get("clusters", [])

    score    = kpi.get("weighted_distinct")
    gate     = cal.get("gate_ok", False)
    pct      = f"{score * 100:.1f}%" if score is not None else "—"
    color    = "#2a7a3b" if score and score >= 0.75 else "#b85c00" if score and score >= 0.5 else "#c0392b"
    g_color  = "#2a7a3b" if gate else "#c0392b"
    g_label  = "PASS" if gate else "FAIL"

    def p(x): return f"{x * 100:.0f}%" if x is not None else "—"

    rows = ""
    for c in clusters:
        ok = c["distinct"]
        ci = f"{c['lo']*100:.0f}–{c['hi']*100:.0f}%" if c["n_draws"] else "—"
        badge = (f"<span style='color:#2a7a3b;font-weight:bold'>✓</span>"
                 if ok else f"<span style='color:#c0392b;font-weight:bold'>✗</span>")
        bg = "#f0faf2" if ok else "#fff8f8"
        rows += (f"<tr style='background:{bg}'><td>{c['cluster_id']}</td>"
                 f"<td>{c['label']}</td><td style='text-align:right'>{c['size']:,}</td>"
                 f"<td style='text-align:right'>{p(c['score'])}</td>"
                 f"<td style='text-align:right;color:#666'>{ci}</td>"
                 f"<td style='text-align:right;color:#888'>{c['n_draws']}</td>"
                 f"<td style='text-align:center'>{badge}</td></tr>\n")

    gate_warn = ("""<div style='background:#fff3cd;border:1px solid #ffc107;padding:1em;
border-radius:6px;margin-bottom:1.5em'><strong>⚠ Calibration gate FAILED.</strong>
The judge could not reliably detect obvious far-cluster intruders (rate
""" + p(cal.get("far_rate")) + """ &lt; 70%). The distinctiveness numbers below
may not be trustworthy. Check your <code>same_when</code> rule and judge gateway.</div>"""
                if not gate else "")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cluster Distinctiveness</title>
<style>
  body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
        max-width:900px;margin:2em auto;padding:0 1.5em;color:#1a1a1a;line-height:1.6}}
  h1{{font-size:1.5em;font-weight:700;margin-bottom:.2em}}
  h2{{font-size:1.1em;font-weight:600;margin-top:2em;border-bottom:1px solid #e0e0e0;padding-bottom:.3em}}
  .hl{{display:flex;gap:2.5em;margin:1.5em 0;flex-wrap:wrap;align-items:flex-start}}
  .big{{font-size:4em;font-weight:800;color:{color};line-height:1}}
  .sub{{color:#555;font-size:.9em;margin-top:.3em}}
  .badge{{display:inline-block;padding:.3em .9em;border-radius:4px;font-weight:700;
           background:{'#e8f5eb' if gate else '#fdecea'};color:{g_color}}}
  table{{border-collapse:collapse;width:100%;font-size:.9em;margin-top:.6em}}
  th{{text-align:left;padding:.5em .7em;background:#f5f5f5;border-bottom:2px solid #ddd;font-weight:600}}
  td{{padding:.4em .7em;border-bottom:1px solid #eee}}
  tr:last-child td{{border-bottom:none}}
  .meta{{color:#666;font-size:.85em;margin-bottom:.8em}}
  .box{{background:#f9f9f9;border-left:4px solid #ccc;padding:.9em 1.1em;
        border-radius:0 6px 6px 0;margin:.8em 0}}
  .box p{{margin:.3em 0}}
  code{{background:#eef;padding:.1em .3em;border-radius:3px;font-size:.9em}}
</style>
</head>
<body>
<h1>Cluster Distinctiveness Report</h1>
<p class="meta">
  {meta.get('n_clusters','?')} clusters · {meta.get('n_texts','?'):,} texts ·
  {kpi.get('n_judged','?')} judged · model: <code>{meta.get('model','?')}</code> ·
  {meta.get('n_llm_calls','?')} LLM calls
</p>
{gate_warn}
<div class="hl">
  <div>
    <div class="big">{pct}</div>
    <div class="sub">Weighted distinctiveness<br>
      <span style="color:#888;font-size:.9em">
        {kpi.get('n_distinct','?')} of {kpi.get('n_judged','?')} clusters
        pass the {int(kpi.get('threshold',0.5)*100)}% threshold
      </span>
    </div>
  </div>
  <div>
    <div class="sub">Calibration gate</div>
    <div class="badge">{g_label}</div>
    <div class="sub" style="margin-top:.5em">
      γ = {p(cal.get('gamma'))} &nbsp;·&nbsp; far detection = {p(cal.get('far_rate'))}
    </div>
  </div>
</div>

<h2>What this number means</h2>
<div class="box">
  <p><strong>{pct}</strong> of your text corpus lives in clusters that are
  well-separated from their nearest neighbours, as judged by the LLM under
  your equivalence rule.</p>
  <p>A cluster scores high when an item from a neighbouring cluster, secretly
  planted among its members, is reliably spotted and isolated by the LLM.
  A low score means the cluster blurs into its neighbours — the LLM cannot
  tell them apart under your rule.</p>
  <p>Weighting is by cluster size, so a large failing cluster pulls the score
  down more than a small one.</p>
</div>

<h2>How it is computed</h2>
<div class="box">
  <p><strong>Step 1 — Calibration.</strong> The LLM is run on two kinds of planted
  controls: (a) <em>pure</em> draws — <code>k</code> items from the same cluster
  — to measure <strong>γ</strong>, the rate the LLM accidentally isolates a
  truly-same item; (b) <em>far</em> draws — <code>k−1</code> home items plus
  one from a distant cluster — to verify the judge can detect obvious intruders
  (gate threshold: ≥ 70%).</p>
  <p><strong>Step 2 — Intruder detection.</strong> For each cluster, one item
  from a near-neighbour cluster is planted among <code>k={meta.get('k_partition','?')}</code> home items and the LLM
  sorts them by kind. Detected = planted item is a singleton group.</p>
  <p><strong>Step 3 — Correction.</strong> Raw detection rates are corrected for
  chance isolation: <code>corrected = (raw − γ) / (1 − γ)</code>.</p>
  <p><strong>Step 4 — Aggregation.</strong> Each cluster's corrected rate is
  thresholded at {int(kpi.get('threshold',0.5)*100)}%. The headline is the
  size-weighted fraction above that threshold.</p>
</div>

<h2>How much to trust it</h2>
<div class="box">
  <p><strong>Calibration gate: <span style="color:{g_color}">{g_label}</span>.</strong>
  PASS = the judge reliably detects obvious intruders, so the correction is
  meaningful. FAIL = the judge or the rule is too weak; fix before reading scores.</p>
  <p><strong>Per-cluster 95% confidence intervals</strong> (lo–hi column) reflect
  how many intruder draws were made per cluster (<code>n</code>). Narrow the
  intervals by raising <code>n_draws</code> or <code>coverage_target</code>.</p>
  <p><strong>γ = {p(cal.get('gamma'))}</strong> is the chance isolation rate —
  how often the LLM mistakenly isolates a truly-same item. Low γ means
  little correction is needed; high γ means results depend heavily on it.</p>
</div>

<h2>Per-cluster results</h2>
<table>
  <tr><th>ID</th><th>Label</th><th style="text-align:right">Size</th>
      <th style="text-align:right">Score</th><th style="text-align:right">95% CI</th>
      <th style="text-align:right">n draws</th><th style="text-align:center">Distinct</th></tr>
  {rows}
</table>
</body>
</html>"""


def write_html(results: dict, path: str) -> None:
    import os
    if os.path.isdir(path):
        path = os.path.join(path, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_html(results))
    log.info("report written to %s", path)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    p = argparse.ArgumentParser(description="Cluster distinctiveness evaluator")
    p.add_argument("--data", required=True)
    p.add_argument("--embeddings", help=".npy file")
    p.add_argument("--embedding-col")
    p.add_argument("--same-when", required=True)
    p.add_argument("--unit", default="each text is a short customer message")
    p.add_argument("--model", default="")
    p.add_argument("--out", default="report.html")
    p.add_argument("--workers", type=int, default=64)
    p.add_argument("--coverage", type=float, default=0.25)
    args = p.parse_args()

    emb = np.load(args.embeddings) if args.embeddings else None
    cfg = Config(same_when=args.same_when, unit=args.unit, model=args.model,
                 workers=args.workers, coverage_target=args.coverage)
    results = evaluate(args.data, emb, config=cfg, embedding_col=args.embedding_col)

    kpi = results["kpi"]; cal = results["calibration"]
    wd  = kpi["weighted_distinct"]
    print(f"\nWeighted distinctiveness: {wd * 100:.1f}%" if wd is not None else "\nWeighted distinctiveness: —")
    print(f"  ({kpi['n_distinct']} of {kpi['n_judged']} clusters pass the "
          f"{int(kpi['threshold']*100)}% threshold)")
    print(f"Calibration: gate={'PASS' if cal['gate_ok'] else 'FAIL'}  "
          f"γ={cal['gamma']:.3f}  far_rate={cal['far_rate']:.2f}")

    write_html(results, args.out)
    print(f"HTML report: {args.out}")


if __name__ == "__main__":
    main()
