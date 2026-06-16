"""cluster_judge — weighted distinctiveness evaluator for text clusterings.

Plants one item from a neighbouring cluster among k−1 home items and asks
the LLM to sort them by kind. Distinctiveness = corrected rate the planted
item is isolated. Headline: size-weighted fraction of clusters that pass.
Output: HTML report.
"""
from __future__ import annotations


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
    n_draws: int = 24          # intruder draws per cluster
    coverage_target: float = 0.5
    min_judgeable: int = 5
    n_cal_pure: int = 60       # pure calibration draws → γ
    n_cal_far: int = 48        # far calibration draws → gate
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
                 "n_llm_calls": client.n_calls, "same_when": cfg.same_when, "unit": cfg.unit},
        "clusters": clusters,
    }


# ── Text report ───────────────────────────────────────────────────────────────

def print_report(results: dict) -> None:
    kpi = results.get("kpi", {})
    cal = results.get("calibration", {})
    clusters = results.get("clusters", [])

    wd   = kpi.get("weighted_distinct")
    gate = cal.get("gate_ok", False)
    pct  = f"{wd * 100:.1f}%" if wd is not None else "—"

    def p(x): return f"{x * 100:.0f}%" if x is not None else "—"

    print(f"\n{'Weighted distinctiveness:':26s} {pct}  "
          f"({kpi.get('n_distinct','?')} of {kpi.get('n_judged','?')} clusters "
          f"pass the {int(kpi.get('threshold', 0.5) * 100)}% threshold)")
    print(f"{'Calibration gate:':26s} {'PASS' if gate else 'FAIL'}   "
          f"γ={p(cal.get('gamma'))}  far detection={p(cal.get('far_rate'))}")
    if not gate:
        print("  ⚠  Gate FAILED — judge cannot reliably detect obvious intruders."
              " Fix same_when or gateway before trusting scores.")

    if clusters:
        id_w  = max(len(c["cluster_id"]) for c in clusters)
        lab_w = max(len(c["label"])      for c in clusters)
        id_w  = max(id_w,  7)
        lab_w = max(lab_w, 5)
        hdr = (f"\n{'Cluster':<{id_w}}  {'Label':<{lab_w}}  {'Size':>6}"
               f"  {'Score':>6}  {'95% CI':>9}  {'n':>4}  Distinct")
        print(hdr)
        print("-" * len(hdr))
        for c in clusters:
            ci = (f"{c['lo']*100:.0f}–{c['hi']*100:.0f}%"
                  if c["n_draws"] else "—")
            mark = "✓" if c["distinct"] else "✗"
            print(f"{c['cluster_id']:<{id_w}}  {c['label']:<{lab_w}}  "
                  f"{c['size']:>6,}  {p(c['score']):>6}  {ci:>9}  "
                  f"{c['n_draws']:>4}  {mark}")

