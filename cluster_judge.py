"""cluster_judge v3 — reference-free LLM-as-judge evaluator for text clusterings.

Measurement core = two primitives:
  PARTITION  sort k shuffled items into groups of the same kind under the user's
             equivalence rule (`same_when`); output is index lists only.
  FIT        judge a cluster card (label + description) against member evidence.

Every metric is a sampling design + deterministic aggregation over these:
  homogeneity & substructure    <- PARTITION on cluster items (overlapping draws
                                   -> co-grouping graph -> components)
  distinctiveness & redundancy  <- PARTITION with one planted near-neighbour
                                   intruder (detection rates + confusion matrix)
  calibration & bias correction <- PARTITION/FIT on planted controls of known
                                   composition (Rogan-Gladen correction, data-
                                   driven thresholds, the pass/fail gate)
Naming, strategies and the taxonomy are presentation-layer calls that run after
measurement and can never move a score. Detection tasks never see labels.

Single judge model behind a corporate gateway: register once with
    use_genai(run_model_messages)   # (messages: list[dict], json_mode=True) -> str
`model="mock"` runs the built-in offline judge (optionally noisy via
Config.mock_eps_* so the correction layer itself can be tested end to end).
Output = results dict -> results.json + a formatted text report (no HTML).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypedDict

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

log = logging.getLogger("cluster_judge")
_log_lock = threading.Lock()

__all__ = ["generate_report", "evaluate_clusters", "evaluate_from_files", "write_report",
           "render_report", "configure_logging", "use_genai", "Config", "LLMClient",
           "make_demo_data", "load_inputs", "evaluate",
           "recluster_cluster", "render_recluster_report", "main",
           "Results", "ClusterResult", "Band"]


# ===========================================================================
# Result schema (documents the dict evaluate() returns; see README "Results schema")
# ===========================================================================
class Band(TypedDict):
    band: str
    lo: int
    hi: int


class ClusterResult(TypedDict, total=False):
    cluster_id: str
    label: str
    description: str
    size: int
    band: str
    coverage: float
    judged: bool
    n_draws: int
    homogeneity: Dict[str, float]
    mixing: float
    mixing_raw: float
    components: List[Dict[str, Any]]
    n_classes: int
    minor_share: float
    split: bool
    one_class: bool
    distinctiveness: Optional[Dict[str, float]]
    far_detect: Optional[float]
    confusable_with: List[Dict[str, Any]]
    merge_group: Optional[int]
    fit: Dict[str, Any]
    placement_accuracy: Optional[float]
    strategies: List[str]
    mineable: Optional[bool]
    judge_uncertain: bool
    dup_agreement: Optional[float]
    parse_fail: int
    review: bool
    review_reasons: List[str]
    note: str


class Results(TypedDict):
    meta: Dict[str, Any]
    calibration: Dict[str, Any]
    kpis: Dict[str, Optional[float]]
    kpis_weighted: Dict[str, Optional[float]]
    by_band: Dict[str, Dict[str, Any]]
    bands: List[Band]
    clusters: List[ClusterResult]
    merge_groups: List[List[str]]
    taxonomy: Optional[Dict[str, Any]]


# ===========================================================================
# Config
# ===========================================================================
@dataclass
class Config:
    # -- task framing (same_when is REQUIRED) --
    unit: str = "each text is a short customer message"
    same_when: Optional[str] = None          # "two items are the same kind when they ..."
    use_context: Optional[str] = None        # what clusters are used for (identity tasks only)
    # -- sampling --
    coverage_target: float = 0.20
    coverage_ceiling: float = 0.40
    min_pool: int = 6
    max_items_per_cluster: int = 6000
    micro_k: int = 8                         # micro-modes for representative pooling
    item_chars: int = 1024
    desc_chars: int = 120
    # -- PARTITION designs --
    k_partition: int = 10
    n_hubs: int = 4                          # farthest-point probes giving the co-graph long range
    draws_per_wave: int = 3                  # homogeneity draws per cluster per wave
    waves_max: int = 6
    intruder_per_wave: int = 4               # cycles near neighbours + far
    intruder_waves_max: int = 3
    neighbor_m: int = 3
    min_judgeable: int = 5                   # smaller clusters are reported, not judged
    resolution_waves: int = 4                # extra homog waves for confirmed splits
    resolution_target_seen: int = 120        # seen-items target for split prevalence
    # -- targeted re-clustering of a single flagged cluster (high-resolution pass) --
    recluster_max_items: int = 600           # cap on items pooled for the deep pass
    recluster_coverage: float = 0.9          # fraction of the cluster to pool (up to the cap)
    recluster_redundancy: int = 4            # PARTITION draws each pooled item appears in
    recluster_name_exemplars: int = 10       # exemplars shown to the naming call per sub-cluster
    recluster_assign_residual: bool = True   # FIT-choice items in sub-floor fragments onto named subs
    # -- decisions --
    tau_mix: float = 0.15                    # mixing tolerance / sub-class prevalence floor
    z: float = 1.96
    # -- calibration (planted controls) --
    k_pure: int = 6                            # items per pure-control draw (cross-cluster kNN)
    n_pure: int = 24
    n_mixed: int = 24
    n_junk: int = 12
    n_far: int = 24
    n_labelswap: int = 10
    dup_rate: float = 0.05                   # silent duplicates -> judge-noise estimate
    # -- FIT --
    fit_items: int = 30
    fit_replicates: int = 3                  # extra replicates when first verdict != accurate
    do_fit_choice: bool = False              # placement is derived-only by default
    fit_choice_n: int = 8
    # -- presentation --
    do_strategy: bool = True
    strategy_n: int = 20
    do_taxonomy: bool = True
    name_exemplars: int = 8
    # -- bands --
    max_bands: int = 5
    gvf_target: float = 0.90
    # -- budget / concurrency --
    max_llm_calls: int = 20000
    workers: int = 64
    max_retries: int = 4
    backoff_base: float = 0.5
    max_empty_rate: float = 0.25             # health gate: if more judge calls than this come back empty, distrust the run
    # -- judge --
    model: str = "mock"
    temperature: float = 0.0
    seed: int = 7
    # -- mock noise (to exercise the correction layer in tests/demo) --
    mock_eps_split: float = 0.0              # P(falsely split a true group) per group per draw
    mock_eps_join: float = 0.0               # P(falsely join two groups) per draw

    def __post_init__(self):
        """Fail fast on incoherent knob combinations rather than degrading silently."""
        if not 0 < self.coverage_target <= 1:
            raise ValueError(f"coverage_target must be in (0, 1], got {self.coverage_target}")
        if not 0 < self.coverage_ceiling <= 1:
            raise ValueError(f"coverage_ceiling must be in (0, 1], got {self.coverage_ceiling}")
        if self.coverage_target > self.coverage_ceiling:
            raise ValueError(f"coverage_target ({self.coverage_target}) must be <= "
                             f"coverage_ceiling ({self.coverage_ceiling})")
        if self.k_partition < 3:
            raise ValueError(f"k_partition must be >= 3 (partition/intruder designs need it), "
                             f"got {self.k_partition}")
        if self.k_pure < 3 or self.k_pure > self.k_partition:
            raise ValueError(f"k_pure must be in [3, k_partition={self.k_partition}], "
                             f"got {self.k_pure}")
        if self.min_judgeable < 2:
            raise ValueError(f"min_judgeable must be >= 2, got {self.min_judgeable}")
        if not 0 <= self.tau_mix <= 1:
            raise ValueError(f"tau_mix must be in [0, 1], got {self.tau_mix}")
        if not 0 <= self.max_empty_rate <= 1:
            raise ValueError(f"max_empty_rate must be in [0, 1], got {self.max_empty_rate}")
        if self.workers < 1:
            raise ValueError(f"workers must be >= 1, got {self.workers}")
        if self.max_llm_calls < 1:
            raise ValueError(f"max_llm_calls must be >= 1, got {self.max_llm_calls}")


# ===========================================================================
# Gateway wiring
# ===========================================================================
_GATEWAY: Dict[str, Optional[Callable]] = {"messages_fn": None}


def configure_logging(level=logging.INFO):
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    log.handlers[:] = [h]
    log.setLevel(level)
    log.propagate = False   # prevent double-printing when root logger also has a handler (e.g. Jupyter)
    return log


def use_genai(messages_fn: Callable) -> Callable:
    """Register the gateway: run_model_messages(messages: list[dict], json_mode=True) -> str.
    Any Config.model other than "mock" then routes through it."""
    _GATEWAY["messages_fn"] = messages_fn
    return messages_fn


def _discover_gateway() -> Optional[Callable]:
    if _GATEWAY["messages_fn"] is not None:
        return _GATEWAY["messages_fn"]
    import builtins, sys
    for ns in (getattr(sys.modules.get("__main__"), "__dict__", {}), vars(builtins)):
        fn = ns.get("run_model_messages")
        if callable(fn):
            return fn
    return None


def _make_client(cfg: Config) -> "LLMClient":
    if cfg.model == "mock":
        return LLMClient(cfg)
    gw = _discover_gateway()
    if gw is not None:
        log.info("judge: gateway via run_model_messages (workers=%d)", cfg.workers)
        return LLMClient(cfg, messages_fn=gw)
    log.warning("no gateway registered (use_genai); falling back to offline mock")
    return LLMClient(cfg)


# ===========================================================================
# Units, prompts, client, executor
# ===========================================================================
@dataclass
class Unit:
    uid: int
    kind: str            # partition | fit_card | fit_choice | name | strategy
    cid: str
    payload: dict
    truth: dict = field(default_factory=dict)   # ground truth / bookkeeping; NEVER in prompts


def _clip(t: Any, n: int) -> str:
    t = str(t).replace("\n", " ").strip()
    return t if len(t) <= n else t[: n - 1] + "…"


def _numbered(items: Sequence[str]) -> str:
    return "\n".join(f"{i + 1}. {t}" for i, t in enumerate(items))


def prompt_for(u: Unit, cfg: Config) -> str:
    p = u.payload
    head = f"# unit: {cfg.unit}\n# rule: two items are the SAME KIND when they {cfg.same_when}\n"
    if u.kind == "partition":
        return (head +
                "Sort the items into groups of the same kind under the rule. Singletons are allowed. "
                "Group by the rule, not by wording, tone, or shared topic words. Use every index exactly once.\n"
                f"ITEMS:\n{_numbered(p['items'])}\n"
                'Return STRICT JSON: {"groups": [[1,4],[2],[3,5]]}')
    if u.kind == "fit_card":
        use = f"# use: {cfg.use_context}\n" if cfg.use_context else ""
        return (head + use +
                "A cluster of items carries this card. Judge whether the card covers the members below. "
                "Do NOT assume it is good. 'accurate' = right breadth; 'too_broad' = vaguer or wider than "
                "the members; 'too_narrow' = misses kinds that are present.\n"
                f"LABEL: {p['label']}\nDESCRIPTION: {p['description'] or '(none)'}\n"
                f"MEMBERS:\n{_numbered(p['items'])}\n"
                'Return STRICT JSON: {"verdict":"accurate|too_broad|too_narrow",'
                '"proposed_label":"<=8 words","proposed_description":"<=25 words"}')
    if u.kind == "fit_choice":
        cards = "\n".join(f"{i + 1}. {c['label']} — {_clip(c['description'], cfg.desc_chars)}"
                          for i, c in enumerate(p["cards"]))
        return (head +
                "Which card does this item belong to under the rule? Pick the NUMBER, or \"none\".\n"
                f"ITEM: {p['item']}\nCARDS:\n{cards}\n"
                'Return STRICT JSON: {"choice": 1}')
    if u.kind == "name":
        return (head +
                "All items below are the SAME KIND under the rule. Name that kind, specifically "
                "(<=6 words; not 'miscellaneous').\n"
                f"ITEMS:\n{_numbered(p['items'])}\n"
                'Return STRICT JSON: {"name":"..."}')
    if u.kind == "strategy":
        use = f"# use: {cfg.use_context}\n" if cfg.use_context else ""
        return (head + use +
                f"These items are one kind. LABEL: {p['label']} — {_clip(p['description'], cfg.desc_chars)}\n"
                "List the DISTINCT response strategies an agent could use, and whether the set yields "
                "a usable range to mine.\n"
                f"ITEMS:\n{_numbered(p['items'])}\n"
                'Return STRICT JSON: {"strategies":["<=8 words"],"mineable":true}')
    return "{}"


_JSON_RE = re.compile(r"\{.*\}", re.S)


def _parse_json(s: Any) -> Optional[dict]:
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
                return None
    return None


class LLMClient:
    """Threaded judge client. mock unless a messages_fn gateway is supplied."""

    def __init__(self, cfg: Config, messages_fn: Optional[Callable] = None):
        self.cfg = cfg
        self.fn = messages_fn
        self.mock = messages_fn is None
        self.n_calls = 0
        self.n_empty = 0                       # calls that returned empty/unparseable after retries
        self._lock = threading.Lock()
        self._mock_rng = random.Random(cfg.seed * 7919 + 13)

    def complete(self, u: Unit) -> dict:
        with self._lock:
            self.n_calls += 1
        if self.mock:
            with self._lock:                       # rng is not thread-safe
                return _mock(u, self.cfg, self._mock_rng)
        msgs = [{"role": "system", "content": "You are a careful evaluator. Reply with STRICT JSON only."},
                {"role": "user", "content": prompt_for(u, self.cfg)}]
        delay = self.cfg.backoff_base
        v = None
        for att in range(self.cfg.max_retries + 1):
            try:
                v = _parse_json(self.fn(msgs, json_mode=True))
                if v:
                    return v
            except Exception as e:
                v = None
                if att == self.cfg.max_retries:
                    log.warning("unit %d (%s) failed after retries: %s", u.uid, u.kind, e)
            time.sleep(delay)
            delay *= 2
        with self._lock:                       # dead/garbage response: feeds the run-level health gate
            self.n_empty += 1
        return {}


def run_units(units: List[Unit], client: LLMClient, cfg: Config,
              desc: str = "judging", progress: bool = True,
              executor: Optional[ThreadPoolExecutor] = None,
              call_log: Optional[List[dict]] = None) -> Dict[int, dict]:
    """One flat queue, one pool: no cluster ever serializes the run. Pass a shared `executor`
    to reuse one pool across all of a run's waves instead of creating one per call."""
    out: Dict[int, dict] = {}
    if not units:
        return out
    bar = tqdm(total=len(units), desc=desc, unit="call") if (progress and tqdm) else None
    own = executor is None
    ex = executor or ThreadPoolExecutor(max_workers=max(1, min(cfg.workers, len(units))))
    try:
        def _make_task(u):
            def task():
                res = client.complete(u)
                if call_log is not None:
                    with _log_lock:
                        call_log.append({"uid": u.uid, "unit": u, "prompt": prompt_for(u, cfg), "result": dict(res)})
                return u.uid, res
            return task

        futs = {ex.submit(_make_task(u)): None for u in units}
        for f in as_completed(futs):
            uid, res = f.result()
            out[uid] = res
            if bar:
                bar.update(1)
    finally:
        if own:
            ex.shutdown(wait=True)
        if bar:
            bar.close()
    return out


def _fair_trim(units: List[Unit], room: int) -> List[Unit]:
    """Trim a wave to `room` units round-robin across clusters, so the budget ceiling can't
    starve whichever clusters happen to be appended last. A prefix cut (`units[:room]`) would
    drop the tail clusters entirely; this takes one unit from each cluster in turn instead."""
    if room <= 0:
        return []
    if len(units) <= room:
        return list(units)
    by_cid: Dict[str, List[Unit]] = defaultdict(list)
    for u in units:
        by_cid[u.cid].append(u)
    kept: List[Unit] = []
    queues = list(by_cid.values())
    while len(kept) < room:
        progressed = False
        for q in queues:
            if q and len(kept) < room:
                kept.append(q.pop(0))
                progressed = True
        if not progressed:
            break
    return kept


# ===========================================================================
# Mock judge (ground-truth themes + injectable error rates)
# ===========================================================================
def _mock(u: Unit, cfg: Config, rng: random.Random) -> dict:
    t = u.truth
    if u.kind == "partition":
        themes = t.get("themes")
        if themes is None:
            return {"groups": [[i + 1] for i in range(len(u.payload["items"]))]}
        groups: Dict[str, List[int]] = {}
        for i, th in enumerate(themes, start=1):
            groups.setdefault(str(th), []).append(i)
        gs = list(groups.values())
        if cfg.mock_eps_split > 0:                       # false split of a true group
            out = []
            for g in gs:
                if len(g) >= 2 and rng.random() < cfg.mock_eps_split:
                    cut = rng.randrange(1, len(g))
                    out += [g[:cut], g[cut:]]
                else:
                    out.append(g)
            gs = out
        if cfg.mock_eps_join > 0 and len(gs) >= 2 and rng.random() < cfg.mock_eps_join:
            i, j = rng.sample(range(len(gs)), 2)         # false join of two groups
            merged = gs[i] + gs[j]
            gs = [g for k_, g in enumerate(gs) if k_ not in (i, j)] + [merged]
        return {"groups": gs}
    if u.kind == "fit_card":
        ths = [str(x) for x in t.get("themes", [])]
        if not ths:
            return {"verdict": "accurate", "proposed_label": u.payload["label"], "proposed_description": ""}
        dom = max(sorted(set(ths)), key=ths.count)   # sorted -> tie-break is hash-seed independent
        share = ths.count(dom) / len(ths)
        lab_th = str(t.get("label_theme"))
        verdict = "accurate" if (lab_th == dom and share >= 0.65) else "too_broad"
        return {"verdict": verdict, "proposed_label": f"theme {dom}",
                "proposed_description": f"objections of theme {dom}"}
    if u.kind == "fit_choice":
        it = str(t.get("item_theme"))
        for i, th in enumerate(t.get("card_themes", []), start=1):
            if str(th) == it:
                return {"choice": i}
        return {"choice": "none"}
    if u.kind == "name":
        ths = [str(x) for x in t.get("themes", ["?"])]
        return {"name": f"theme {max(sorted(set(ths)), key=ths.count)}"}
    if u.kind == "strategy":
        size = int(t.get("size", 0))
        k = 1 + int(size >= 20) + int(size >= 500) + int(size >= 5000)
        return {"strategies": [f"strategy {i + 1}" for i in range(k)], "mineable": k >= 2}
    return {}


# ===========================================================================
# Data prep: IO, coercion, pooling, neighbours, bands
# ===========================================================================
def load_inputs(path: str) -> pd.DataFrame:
    p = str(path).lower()
    if p.endswith(".parquet"):
        return pd.read_parquet(path)
    if p.endswith(".tsv"):
        return pd.read_csv(path, sep="\t")
    if p.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    if p.endswith(".json"):
        return pd.read_json(path)
    return pd.read_csv(path)


def _coerce_inputs(data, embeddings=None, *, text_col="text", cluster_col="cluster_id",
                   label_col="label", description_col=None, embedding_col=None, theme_col=None):
    df = data if isinstance(data, pd.DataFrame) else load_inputs(data)
    ren = {}
    if text_col != "text" and text_col in df.columns:
        ren[text_col] = "text"
    if cluster_col != "cluster_id" and cluster_col in df.columns:
        ren[cluster_col] = "cluster_id"
    if theme_col and theme_col in df.columns:
        ren[theme_col] = "_theme"
    df = df.rename(columns=ren).copy()
    if "text" not in df.columns or "cluster_id" not in df.columns:
        raise ValueError("inputs need text and cluster_id columns (or map them via *_col)")
    dcol = description_col or next((c for c in ("description", "desc", "summary") if c in df.columns), None)
    labels: Dict[str, dict] = {}
    for cid, g in df.groupby("cluster_id"):
        cid = str(cid)
        lab = str(g[label_col].iloc[0]) if label_col in df.columns else cid
        desc = str(g[dcol].iloc[0]) if dcol else ""
        labels[cid] = {"label": lab, "description": "" if desc in ("nan", "None") else desc}
        if "_theme" in df.columns:
            labels[cid]["_theme"] = g["_theme"].mode().iloc[0]
    if embeddings is not None:
        emb = np.asarray(embeddings, dtype=np.float32)
    elif embedding_col and embedding_col in df.columns:
        emb = np.vstack(df[embedding_col].to_list()).astype(np.float32)
    else:
        raise ValueError("embeddings required: pass embeddings=<array> or embedding_col=<col>")
    if len(emb) != len(df):
        raise ValueError(f"embeddings length {len(emb)} != rows {len(df)}")
    log.info("loaded %d rows, %d clusters, embeddings %s",
             len(df), df["cluster_id"].nunique(), "×".join(str(d) for d in emb.shape))
    return df, emb, labels


def _normalize(emb: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(emb, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (emb / n).astype(np.float32)


def _pool(emb_n: np.ndarray, idxs: np.ndarray, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """Coverage-driven, mode-proportional (representative) sample of a cluster."""
    n = len(idxs)
    target = int(min(max(cfg.min_pool, round(n * cfg.coverage_target)),
                     n, cfg.max_items_per_cluster, max(cfg.min_pool, round(n * cfg.coverage_ceiling))))
    if n <= target:
        return np.asarray(idxs)
    if n >= cfg.micro_k * 4:
        from sklearn.cluster import MiniBatchKMeans
        km = MiniBatchKMeans(n_clusters=cfg.micro_k, random_state=cfg.seed, n_init=3).fit(emb_n[idxs])
        sel: List[int] = []
        arr = np.asarray(idxs)
        for m in range(cfg.micro_k):
            mem = arr[km.labels_ == m]
            if len(mem) == 0:
                continue
            take = min(len(mem), max(1, round(target * len(mem) / n)))
            sel += list(rng.choice(mem, size=take, replace=False))
        sel = sel[:target]
        if len(sel) < target:
            rest = np.setdiff1d(arr, np.asarray(sel))
            sel += list(rng.choice(rest, size=target - len(sel), replace=False))
        return np.asarray(sel)
    return rng.choice(np.asarray(idxs), size=target, replace=False)


def _fit_bands(sizes: List[int], cfg: Config) -> List[dict]:
    """Data-driven size bands: singletons isolated; 1-D k-means on log10(size) to GVF target."""
    uniq = sorted(set(sizes))
    bands: List[dict] = []
    if 1 in uniq:
        bands.append({"band": "1", "lo": 1, "hi": 1})
        uniq = [u for u in uniq if u > 1]
    if not uniq:
        return bands
    xs = np.log10(np.array([s for s in sizes if s > 1], dtype=float))
    best = None
    sst = float(((xs - xs.mean()) ** 2).sum()) or 1.0
    for k in range(1, min(cfg.max_bands - len(bands), len(set(xs.tolist()))) + 1):
        cuts = np.quantile(xs, np.linspace(0, 1, k + 1)[1:-1]) if k > 1 else np.array([])
        cent = None
        for _ in range(30):
            edges = np.concatenate([[-np.inf], cuts, [np.inf]])
            lab = np.digitize(xs, edges[1:-1])
            cent = np.array([xs[lab == j].mean() if (lab == j).any() else np.nan for j in range(k)])
            cent = np.where(np.isnan(cent), np.nanmean(cent), cent)
            new = (cent[:-1] + cent[1:]) / 2 if k > 1 else np.array([])
            if len(new) == len(cuts) and np.allclose(new, cuts):
                break
            cuts = new
        lab = np.digitize(xs, np.concatenate([[-np.inf], cuts, [np.inf]])[1:-1])
        ssw = sum(float(((xs[lab == j] - xs[lab == j].mean()) ** 2).sum()) for j in range(k) if (lab == j).any())
        gvf = 1 - ssw / sst
        best = (k, cuts)
        if gvf >= cfg.gvf_target:
            break
    k, cuts = best
    edges = [2] + [int(round(10 ** c)) + 1 for c in cuts] + [max(sizes) + 1]
    for lo, hi in zip(edges[:-1], edges[1:]):
        mem = [s for s in sizes if lo <= s < hi]
        if not mem:
            continue
        a, b = min(mem), max(mem)
        lbl = _fmt_size(a) if a == b else f"{_fmt_size(a)}–{_fmt_size(b)}"
        bands.append({"band": lbl, "lo": a, "hi": b})
    return bands


def _fmt_size(n: int) -> str:
    return f"{round(n / 1000)}k" if n >= 1000 else str(n)


def _band_of(size: int, bands: List[dict]) -> str:
    for b in bands:
        if b["lo"] <= size <= b["hi"]:
            return b["band"]
    return bands[-1]["band"] if bands else "all"


def _wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


# ===========================================================================
# Per-cluster state and draw construction
# ===========================================================================
@dataclass
class CState:
    cid: str
    code: int
    size: int
    pool: np.ndarray
    judged: bool = True
    hubs: List[int] = field(default_factory=list)
    order: List[int] = field(default_factory=list)
    cursor: int = 0
    mix_draws: List[float] = field(default_factory=list)         # per-draw mixing fraction
    pair: Dict[Tuple[int, int], List[int]] = field(default_factory=dict)   # (i,j) -> [co, tot]
    seen: set = field(default_factory=set)
    parse_fail: int = 0
    homog_done: bool = False
    det: Dict[str, List[int]] = field(default_factory=dict)      # stratum -> [detected, n]
    intr_done: bool = False
    dup_agree: List[float] = field(default_factory=list)
    hub_i: int = 0


def _homog_block(st: CState, k: int, prng: random.Random) -> List[int]:
    n = len(st.order)
    k = min(k, n)
    if st.cursor == 0 and st.order:
        prng.shuffle(st.order)                                  # fresh pass -> pairs recur across sweeps
    if st.cursor + k <= n:
        block = st.order[st.cursor:st.cursor + k]
    else:
        block = st.order[st.cursor:] + st.order[:(st.cursor + k) % n]
    st.cursor = (st.cursor + max(1, k // 2)) % n               # 50% overlap chains the co-graph
    return block


def _groups_valid(v: dict, k: int) -> Optional[List[List[int]]]:
    try:
        gs = [[int(x) for x in g] for g in v.get("groups", [])]
    except Exception:
        return None
    flat = sorted(x for g in gs for x in g)
    return gs if flat == list(range(1, k + 1)) else None


def _pair_pattern(gs: List[List[int]], k: int) -> set:
    gid = {}
    for gi, g in enumerate(gs):
        for x in g:
            gid[x] = gi
    return {(a, b) for a, b in combinations(range(1, k + 1), 2) if gid[a] == gid[b]}


# ===========================================================================
# Evaluate
# ===========================================================================
@dataclass
class _Eval:
    """Shared state + unit builders for one evaluate() run, threaded through the phases."""
    df: pd.DataFrame
    cfg: Config
    client: "LLMClient"
    rng: Any
    prng: random.Random
    cid_of: Dict[int, str]
    K: int
    text_arr: np.ndarray
    theme_arr: Optional[np.ndarray]
    emb_n: np.ndarray
    cent: np.ndarray
    nb_order: np.ndarray
    far_pool: np.ndarray
    labels: Dict[str, dict]
    states: Dict[str, CState]
    judged: List[CState]
    uid: List[int] = field(default_factory=lambda: [0])
    calls_by_kind: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    executor: Optional[ThreadPoolExecutor] = None

    def U(self, kind, cid, payload, truth) -> Unit:
        self.uid[0] += 1
        self.calls_by_kind[kind] += 1
        return Unit(self.uid[0], kind, cid, payload, truth)

    def items_of(self, ids: Sequence[int]) -> List[str]:
        return [_clip(self.text_arr[i], self.cfg.item_chars) for i in ids]

    def themes_of(self, ids: Sequence[int]) -> Optional[List[str]]:
        return [str(self.theme_arr[i]) for i in ids] if self.theme_arr is not None else None

    def part_unit(self, cid: str, ids: List[int], ctx: str, plant_pos: Optional[int] = None,
                  plant_src: Optional[str] = None, extra: Optional[dict] = None) -> Unit:
        truth = {"ctx": ctx, "item_ids": ids, "themes": self.themes_of(ids),
                 "plant_pos": plant_pos, "plant_src": plant_src}
        if extra:
            truth.update(extra)
        return self.U("partition", cid, {"items": self.items_of(ids)}, truth)

    def homog_unit(self, st: CState) -> Unit:
        ids = _homog_block(st, self.cfg.k_partition - (1 if st.hubs else 0), self.prng)
        if st.hubs:                              # rotating hub = long-range graph connector
            hub = st.hubs[st.hub_i % len(st.hubs)]
            st.hub_i += 1
            pos = self.prng.randrange(len(ids) + 1)
            ids = ids[:pos] + [hub] + ids[pos:]
        return self.part_unit(st.cid, ids, "homog")

    def guard(self, units: List[Unit]) -> List[Unit]:
        room = max(0, self.cfg.max_llm_calls - self.client.n_calls)
        if len(units) > room:
            log.warning("budget ceiling: trimming wave %d -> %d units (fair round-robin by cluster)",
                        len(units), room)
        return _fair_trim(units, room)

    def run(self, units: List[Unit], desc: str, progress: bool, call_log: Optional[List[dict]] = None) -> Dict[int, dict]:
        return run_units(units, self.client, self.cfg, desc=desc, progress=progress,
                         executor=self.executor, call_log=call_log)


def _setup_eval(df: pd.DataFrame, emb: np.ndarray, labels: Dict[str, dict], cfg: Config,
                client: "LLMClient") -> _Eval:
    """Build per-cluster state, coverage pools, centroids, and the neighbour map for one run."""
    rng = np.random.default_rng(cfg.seed)
    prng = random.Random(cfg.seed)
    df = df.reset_index(drop=True)
    cats = pd.Categorical(df["cluster_id"].astype(str))
    codes = cats.codes
    cid_of = {i: str(c) for i, c in enumerate(cats.categories)}
    K = len(cats.categories)
    text_arr = df["text"].astype(str).to_numpy()
    theme_arr = df["_theme"].to_numpy() if "_theme" in df.columns else None
    emb_n = _normalize(np.asarray(emb, dtype=np.float32))
    for cid in cid_of.values():
        labels.setdefault(cid, {"label": cid, "description": ""})
        labels[cid].setdefault("description", "")

    # centroids + neighbour map
    log.info("  computing cluster centroids and neighbour map …")
    cent = np.zeros((K, emb_n.shape[1]), dtype=np.float32)
    for c in range(K):
        cent[c] = emb_n[codes == c].mean(axis=0)
    cent = _normalize(cent)
    sims = cent @ cent.T
    np.fill_diagonal(sims, -2)
    nb_order = np.argsort(-sims, axis=1)

    # states + pools (+ farthest-point hubs: long-range probes for the co-grouping graph)
    log.info("  building coverage pools for %d clusters (coverage target %.0f%%) …",
             K, cfg.coverage_target * 100)
    states: Dict[str, CState] = {}
    for c in range(K):
        idxs = np.where(codes == c)[0]
        st = CState(cid=cid_of[c], code=c, size=len(idxs), pool=_pool(emb_n, idxs, cfg, rng))
        st.judged = len(st.pool) >= cfg.min_judgeable
        if cfg.n_hubs > 0 and len(st.pool) >= cfg.k_partition * 2:
            sub = st.pool if len(st.pool) <= 512 else rng.choice(st.pool, size=512, replace=False)
            hubs = [int(sub[0])]
            d = 1 - emb_n[sub] @ emb_n[hubs[0]]
            for _ in range(cfg.n_hubs - 1):
                hubs.append(int(sub[int(np.argmax(d))]))
                d = np.minimum(d, 1 - emb_n[sub] @ emb_n[hubs[-1]])
            st.hubs = hubs
            st.order = [int(x) for x in rng.permutation(np.setdiff1d(st.pool, np.asarray(hubs)))]
        else:
            st.order = list(map(int, rng.permutation(st.pool)))
        states[st.cid] = st
    judged = [s for s in states.values() if s.judged]
    for st in judged:
        if K < 2 or len(st.pool) < cfg.k_partition - 1:
            st.intr_done = True               # no intruder design possible
    far_pool = nb_order[:, K // 2:-1] if K > 3 else nb_order[:, :-1]   # self excluded (last)
    return _Eval(df=df, cfg=cfg, client=client, rng=rng, prng=prng, cid_of=cid_of, K=K,
                 text_arr=text_arr, theme_arr=theme_arr, emb_n=emb_n, cent=cent, nb_order=nb_order,
                 far_pool=far_pool, labels=labels, states=states, judged=judged)


def _build_cal_calls(units: List[Unit], log_entries: List[dict], res: Dict[int, dict], cfg: Config) -> List[dict]:
    """Build structured per-call records for the calibration review HTML."""
    log_by_uid = {e["uid"]: e for e in log_entries}
    records = []
    for u in units:
        entry = log_by_uid.get(u.uid, {})
        result = res.get(u.uid, {})
        ctx = u.truth.get("ctx", "?")
        rec = {
            "uid": u.uid,
            "ctrl_type": ctx,
            "kind": u.kind,
            "cid": u.cid,
            "prompt": entry.get("prompt", ""),
        }

        if u.kind == "partition":
            items = u.payload["items"]
            k = len(items)
            gs = _groups_valid(result, k)

            if ctx in ("pure", "pure_cluster"):
                passed = gs is not None and len(gs) == 1
                rec.update({
                    "items": items,
                    "expected": "all items in one group",
                    "passed": passed,
                    "n_groups_returned": len(gs) if gs else None,
                    "groups_returned": [[items[i-1] for i in g] for g in gs] if gs else None,
                })

            elif ctx == "mixed":
                side = u.truth.get("side", [])
                a_items = [{"text": items[i], "group": "A"} for i, s in enumerate(side) if s == 0]
                b_items = [{"text": items[i], "group": "B"} for i, s in enumerate(side) if s == 1]
                pairs = _pair_pattern(gs, k) if gs else set()
                ct = sum(1 for a, b in combinations(range(1, k+1), 2) if side[a-1] != side[b-1])
                cj = sum(1 for a, b in combinations(range(1, k+1), 2) if side[a-1] != side[b-1] and (min(a,b), max(a,b)) in pairs)
                passed = ct > 0 and gs is not None and (cj / ct) < 0.5
                rec.update({
                    "items_a": a_items,
                    "items_b": b_items,
                    "expected": "items split into two groups (sources A vs B)",
                    "passed": passed,
                    "n_groups_returned": len(gs) if gs else None,
                    "groups_returned": [[items[i-1] for i in g] for g in gs] if gs else None,
                })

            elif ctx in ("junk", "farcal"):
                plant_pos = u.truth.get("plant_pos")
                plant_src = u.truth.get("plant_src", "?")
                home_items = [items[i] for i in range(k) if i + 1 != plant_pos]
                planted = items[plant_pos - 1] if plant_pos else None
                passed = gs is not None and any(g == [plant_pos] for g in gs)
                rec.update({
                    "home_items": home_items,
                    "planted_item": planted,
                    "plant_src": plant_src,
                    "expected": "planted item isolated as its own group",
                    "passed": passed,
                    "n_groups_returned": len(gs) if gs else None,
                    "groups_returned": [[items[i-1] for i in g] for g in gs] if gs else None,
                })

        elif u.kind == "fit_card":  # ctx == "swap"
            verdict = result.get("verdict", "")
            proposed = result.get("proposed_label", "")
            rec.update({
                "label_shown": u.payload["label"],
                "description_shown": u.payload.get("description", ""),
                "items": u.payload["items"],
                "expected": "label rejected — verdict should NOT be 'accurate'",
                "passed": verdict != "accurate" and bool(verdict),
                "verdict_returned": verdict,
                "proposed_label": proposed,
            })

        records.append(rec)
    return records


def _run_calibration(ev: _Eval, progress: bool) -> dict:
    """Planted-control wave (pure/mixed/junk/far intruders + label swaps) -> Sp/Se + gate."""
    cfg, prng, rng = ev.cfg, ev.prng, ev.rng
    cal_units: List[Unit] = []
    pool_ok = [s for s in ev.judged if len(s.pool) >= cfg.k_partition]
    pool_pure = [s for s in ev.judged if len(s.pool) >= cfg.k_pure]
    n_items = len(ev.df)
    if pool_ok and ev.K >= 2:
        _cid_col = ev.df["cluster_id"].astype(str).to_numpy()
        # Batch all cross-cluster similarity lookups into one matmul (BLAS-friendly)
        centers = [prng.randint(0, n_items - 1) for _ in range(cfg.n_pure)]
        sim_matrix = ev.emb_n[centers] @ ev.emb_n.T   # (n_pure, n_items)
        for i, center in enumerate(centers):
            order = np.argsort(-sim_matrix[i])
            ids = [int(x) for x in order if int(x) != center][:cfg.k_pure]
            prng.shuffle(ids)
            cal_units.append(ev.part_unit(str(_cid_col[center]), ids, "pure"))
        for _ in range(cfg.n_pure):                       # within-cluster kNN ball -> informational only
            st = prng.choice(pool_pure)
            center = int(prng.choice(list(st.pool)))
            s = ev.emb_n[st.pool] @ ev.emb_n[center]
            ids = [int(x) for x in st.pool[np.argsort(-s)[:cfg.k_pure]]]
            prng.shuffle(ids)
            cal_units.append(ev.part_unit(st.cid, ids, "pure_cluster"))
        for _ in range(cfg.n_mixed):                      # half A + half far B -> expect a split
            a = prng.choice(pool_ok)
            b_code = int(prng.choice(ev.far_pool[a.code]))
            b = ev.states[ev.cid_of[b_code]]
            if not len(b.pool):
                continue
            ka = cfg.k_partition // 2
            ia = [int(x) for x in rng.choice(a.pool, size=min(ka, len(a.pool)), replace=False)]
            ib = [int(x) for x in rng.choice(b.pool, size=min(cfg.k_partition - len(ia), len(b.pool)), replace=False)]
            ids = ia + ib
            side = [0] * len(ia) + [1] * len(ib)
            order = list(range(len(ids)))
            prng.shuffle(order)
            cal_units.append(ev.part_unit(a.cid, [ids[i] for i in order], "mixed",
                                          extra={"side": [side[i] for i in order]}))
        words = " ".join(_clip(ev.text_arr[i], 80) for i in rng.choice(len(ev.text_arr), size=20)).split()
        for _ in range(cfg.n_junk):                       # shuffled-word junk plant -> must isolate
            st = prng.choice(pool_ok)
            home = [int(x) for x in rng.choice(st.pool, size=cfg.k_partition - 1, replace=False)]
            junk = " ".join(prng.choice(words) for _ in range(12))
            pos = prng.randrange(len(home) + 1)
            ids = home[:pos] + [-1] + home[pos:]
            u = ev.U("partition", st.cid,
                     {"items": ev.items_of(home[:pos]) + [junk] + ev.items_of(home[pos:])},
                     {"ctx": "junk", "item_ids": ids, "plant_pos": pos + 1, "plant_src": "junk",
                      "themes": (ev.themes_of(home[:pos]) + ["JUNK"] + ev.themes_of(home[pos:])) if ev.theme_arr is not None else None})
            cal_units.append(u)
        for _ in range(cfg.n_far):                        # far intruder -> easy detection floor
            st = prng.choice(pool_ok)
            b = ev.states[ev.cid_of[int(prng.choice(ev.far_pool[st.code]))]]
            if not len(b.pool):
                continue
            home = [int(x) for x in rng.choice(st.pool, size=cfg.k_partition - 1, replace=False)]
            plant = int(prng.choice(list(b.pool)))
            pos = prng.randrange(len(home) + 1)
            ids = home[:pos] + [plant] + home[pos:]
            cal_units.append(ev.part_unit(st.cid, ids, "farcal", plant_pos=pos + 1, plant_src=b.cid))
        for _ in range(cfg.n_labelswap):                  # foreign card -> FIT must flag
            st = prng.choice(pool_ok)
            others = [s for s in pool_ok if s.cid != st.cid and ev.labels[s.cid]["label"] != ev.labels[st.cid]["label"]]
            if not others:
                break
            o = prng.choice(others)
            ids = [int(x) for x in rng.choice(st.pool, size=min(cfg.fit_items, len(st.pool)), replace=False)]
            cal_units.append(ev.U("fit_card", st.cid,
                                  {"label": ev.labels[o.cid]["label"], "description": ev.labels[o.cid]["description"],
                                   "items": ev.items_of(ids)},
                                  {"ctx": "swap", "themes": ev.themes_of(ids),
                                   "label_theme": ev.labels[o.cid].get("_theme")}))
    cal_units = ev.guard(cal_units)
    cal_log: List[dict] = []
    cal_res = ev.run(cal_units, "calibrating", progress, call_log=cal_log)
    cal = _calibrate(cal_units, cal_res, cfg)
    cal["calls"] = _build_cal_calls(cal_units, cal_log, cal_res, cfg)
    return cal


def _run_measurement(ev: _Eval, cal: dict, master: Dict[int, Unit], progress: bool) -> None:
    """Homogeneity + intruder waves with early stopping, ingested into per-cluster state."""
    cfg, prng, rng = ev.cfg, ev.prng, ev.rng
    th = cal["thresholds"]
    n_waves = max(cfg.waves_max, cfg.intruder_waves_max)
    for w in range(n_waves):
        wave: List[Unit] = []
        for st in ev.judged:
            if not st.homog_done and w < cfg.waves_max:
                for _ in range(cfg.draws_per_wave):
                    wave.append(ev.homog_unit(st))
            if (not st.intr_done and w < cfg.intruder_waves_max and ev.K >= 2
                    and len(st.pool) >= cfg.k_partition - 1):
                near = [int(x) for x in ev.nb_order[st.code][:-1]][:cfg.neighbor_m]
                cyc = near + ["far"]
                for j in range(cfg.intruder_per_wave):
                    src = cyc[(w * cfg.intruder_per_wave + j) % len(cyc)]
                    if src == "far":
                        b = ev.states[ev.cid_of[int(prng.choice(ev.far_pool[st.code]))]]
                        tag = "far"
                    else:
                        b = ev.states[ev.cid_of[src]]
                        tag = b.cid
                    if not len(b.pool):
                        continue
                    home = [int(x) for x in rng.choice(st.pool, size=cfg.k_partition - 1, replace=False)]
                    plant = int(prng.choice(list(b.pool)))
                    pos = prng.randrange(len(home) + 1)
                    ids = home[:pos] + [plant] + home[pos:]
                    wave.append(ev.part_unit(st.cid, ids, "intr", plant_pos=pos + 1, plant_src=tag))
        dups = [ev.U("partition", u.cid, u.payload, dict(u.truth, dup_of=u.uid))
                for u in wave if prng.random() < cfg.dup_rate]
        wave = ev.guard(wave + dups)
        if not wave:
            break
        res = ev.run(wave, f"wave {w + 1}", progress)
        for u in wave:
            master[u.uid] = u
        _ingest_wave(wave, res, ev.states, master, cfg)
        _update_decisions(ev.judged, cfg, th)
        if all(s.homog_done and s.intr_done for s in ev.judged):
            break


def _run_resolution(ev: _Eval, cal: dict, est: Dict[str, dict], master: Dict[int, Unit],
                    progress: bool) -> None:
    """Buy extra homogeneity draws where the split / one-class verdict is still unsettled.

    Gating on the split flag alone would be circular: a fragmented minority can suppress the
    flag, so the run would never buy the draws that resolve it (the component PREVALENCE of a
    confirmed split needs |seen| to stabilise). The corrected-mixing CI straddling tau_mix is
    the non-circular trigger."""
    cfg = ev.cfg

    def _needs(st: CState) -> bool:
        e = est[st.cid]
        unsettled = not e["split"] and not e["one_class"]
        return ((e["split"] or unsettled)
                and len(st.seen) < cfg.resolution_target_seen
                and len(st.pool) > len(st.seen))

    for _ in range(cfg.resolution_waves):
        todo = [st for st in ev.judged if _needs(st)]
        if not todo:
            break
        wave = [ev.homog_unit(st) for st in todo for _ in range(cfg.draws_per_wave)]
        wave = ev.guard(wave)
        if not wave:
            break
        res = ev.run(wave, "resolution", progress)
        _ingest_wave(wave, res, ev.states, master, cfg)
        for st in todo:
            est[st.cid] = _estimate_cluster(st, cfg, cal)


def _run_fit(ev: _Eval, mg_of: Dict[str, int],
             progress: bool) -> Tuple[Dict[str, List[dict]], Dict[str, List[int]]]:
    """Label-fit cards (replicated on non-accurate verdicts) and optional placement choice."""
    cfg, rng, prng = ev.cfg, ev.rng, ev.prng
    fit_units: List[Unit] = []
    for st in ev.judged:
        ids = [int(x) for x in rng.choice(st.pool, size=min(cfg.fit_items, len(st.pool)), replace=False)]
        fit_units.append(ev.U("fit_card", st.cid,
                              {"label": ev.labels[st.cid]["label"], "description": ev.labels[st.cid]["description"],
                               "items": ev.items_of(ids)},
                              {"themes": ev.themes_of(ids), "label_theme": ev.labels[st.cid].get("_theme")}))
        if cfg.do_fit_choice and ev.K >= 2:
            near = [ev.cid_of[int(x)] for x in ev.nb_order[st.code][:-1]][:cfg.fit_choice_n - 1]
            cards = [{"cid": c, "label": ev.labels[c]["label"], "description": ev.labels[c]["description"]}
                     for c in [st.cid] + near]
            for i in rng.choice(st.pool, size=min(8, len(st.pool)), replace=False):
                order = list(range(len(cards)))
                prng.shuffle(order)
                sh = [cards[o] for o in order]
                fit_units.append(ev.U("fit_choice", st.cid,
                                      {"item": _clip(ev.text_arr[int(i)], cfg.item_chars), "cards": sh},
                                      {"home_pos": order.index(0) + 1, "cids": [c["cid"] for c in sh],
                                       "item_theme": str(ev.theme_arr[int(i)]) if ev.theme_arr is not None else None,
                                       "card_themes": [ev.labels[c["cid"]].get("_theme") for c in sh]}))
    fit_units = ev.guard(fit_units)
    fit_res = ev.run(fit_units, "fit", progress)
    rep_units: List[Unit] = []
    for u in fit_units:
        if u.kind != "fit_card":
            continue
        v = fit_res.get(u.uid, {})
        if v.get("verdict", "accurate") != "accurate" and cfg.fit_replicates > 1:
            st = ev.states[u.cid]
            for _ in range(cfg.fit_replicates - 1):
                ids = [int(x) for x in rng.choice(st.pool, size=min(cfg.fit_items, len(st.pool)), replace=False)]
                rep_units.append(ev.U("fit_card", u.cid, dict(u.payload, items=ev.items_of(ids)),
                                      {"themes": ev.themes_of(ids), "label_theme": ev.labels[u.cid].get("_theme")}))
    rep_units = ev.guard(rep_units)
    rep_res = ev.run(rep_units, "fit replicates", progress)
    fit_verdicts: Dict[str, List[dict]] = defaultdict(list)
    for u in fit_units + rep_units:
        if u.kind == "fit_card":
            fit_verdicts[u.cid].append((fit_res | rep_res).get(u.uid, {}))
    choice_stats: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for u in fit_units:
        if u.kind != "fit_choice":
            continue
        v = fit_res.get(u.uid, {})
        ch = v.get("choice")
        ok = False
        if isinstance(ch, int) and 1 <= ch <= len(u.truth["cids"]):
            chosen = u.truth["cids"][ch - 1]
            ok = (ch == u.truth["home_pos"]) or (mg_of.get(chosen) is not None
                                                 and mg_of.get(chosen) == mg_of.get(u.cid))
        choice_stats[u.cid][0] += int(ok)
        choice_stats[u.cid][1] += 1
    return fit_verdicts, choice_stats


def _run_presentation(ev: _Eval, est: Dict[str, dict], merge_groups: List[List[str]],
                      progress: bool) -> Tuple[Dict[Tuple[str, int], str], Dict[str, dict], Optional[dict]]:
    """Sub-theme names, strategy mining, and the label taxonomy. Never moves a score."""
    cfg, rng, prng = ev.cfg, ev.rng, ev.prng
    pres: List[Unit] = []
    for st in ev.judged:
        e = est[st.cid]
        if e["split"]:
            for ci, (comp, is_res) in enumerate(e["_comp_items"][:6]):
                if is_res:
                    continue                    # residual is a mixed bag; no single name
                ex = [int(x) for x in (comp if len(comp) <= cfg.name_exemplars
                                       else rng.choice(comp, size=cfg.name_exemplars, replace=False))]
                pres.append(ev.U("name", st.cid, {"items": ev.items_of(ex)},
                                 {"comp": ci, "themes": ev.themes_of(ex)}))
        if cfg.do_strategy:
            ids = [int(x) for x in rng.choice(st.pool, size=min(cfg.strategy_n, len(st.pool)), replace=False)]
            pres.append(ev.U("strategy", st.cid,
                             {"label": ev.labels[st.cid]["label"], "description": ev.labels[st.cid]["description"],
                              "items": ev.items_of(ids)}, {"size": st.size}))
    taxonomy = None
    parents_members: List[List[str]] = []
    if cfg.do_taxonomy and ev.K >= 4:
        from sklearn.cluster import AgglomerativeClustering
        P = int(min(24, max(2, ev.K // 8)))
        plab = AgglomerativeClustering(n_clusters=P).fit_predict(ev.cent)
        for p in range(P):
            members = [ev.cid_of[c] for c in range(ev.K) if plab[c] == p]
            parents_members.append(members)
            ex: List[int] = []
            for cid in members[:4]:
                pl = ev.states[cid].pool
                ex += [int(x) for x in pl[:2]]
            pres.append(ev.U("name", "__tax__", {"items": ev.items_of(ex[: cfg.name_exemplars])},
                             {"parent": p, "themes": ev.themes_of(ex[: cfg.name_exemplars])}))
    pres = ev.guard(pres)
    pres_res = ev.run(pres, "naming", progress)
    comp_names: Dict[Tuple[str, int], str] = {}
    strategies: Dict[str, dict] = {}
    parent_names: Dict[int, str] = {}
    for u in pres:
        v = pres_res.get(u.uid, {})
        if u.kind == "name" and u.cid == "__tax__":
            parent_names[u.truth["parent"]] = str(v.get("name", f"group {u.truth['parent']}"))
        elif u.kind == "name":
            comp_names[(u.cid, u.truth["comp"])] = str(v.get("name", f"sub-theme {u.truth['comp'] + 1}"))
        elif u.kind == "strategy":
            strategies[u.cid] = {"strategies": list(v.get("strategies", []))[:8],
                                 "mineable": bool(v.get("mineable", False))}
    if cfg.do_taxonomy and parents_members:
        red = [[a, b] for grp in merge_groups for a, b in combinations(grp, 2)][:60]
        taxonomy = {"parents": [{"name": parent_names.get(i, f"group {i}"), "members": m}
                                for i, m in enumerate(parents_members)],
                    "redundant_pairs": red}
    return comp_names, strategies, taxonomy


def evaluate(df: pd.DataFrame, emb: np.ndarray, labels: Dict[str, dict], cfg: Config,
             client: Optional[LLMClient] = None, progress: bool = True) -> "Results":
    """Reference-free LLM-as-judge evaluation of a clustering. Orchestrates the phases; each is
    an independently testable function over a shared _Eval context. All waves dispatch through
    one thread pool."""
    if not cfg.same_when:
        raise ValueError("Config.same_when is required — the equivalence rule drives every judgment")
    client = client or _make_client(cfg)
    n_rows = len(df)
    n_clusters = df["cluster_id"].nunique()
    log.info("inputs: %d items across %d clusters", n_rows, n_clusters)
    log.info("config: model=%s  workers=%d  k_partition=%d  k_pure=%d  coverage=%.0f%%",
             cfg.model, cfg.workers, cfg.k_partition, cfg.k_pure, cfg.coverage_target * 100)
    log.info("building cluster pools and neighbour map …")
    ev = _setup_eval(df, emb, labels, cfg, client)
    n_judged = len(ev.judged)
    n_small = n_clusters - n_judged
    avg_pool = sum(len(s.pool) for s in ev.judged) / n_judged if n_judged else 0
    log.info("setup done: %d clusters eligible for judging, %d too small (<%d items); "
             "avg pool size %.0f", n_judged, n_small, cfg.min_judgeable, avg_pool)
    log.info("starting calibration (%d pure + %d pure-cluster + %d mixed + %d junk + "
             "%d far + %d label-swap controls) …",
             cfg.n_pure, cfg.n_pure, cfg.n_mixed, cfg.n_junk, cfg.n_far, cfg.n_labelswap)
    with ThreadPoolExecutor(max_workers=max(1, cfg.workers)) as executor:
        ev.executor = executor
        cal = _run_calibration(ev, progress)
        gate = "PASS" if cal.get("overall_pass") else "FAIL"
        log.info("calibration done: gate=%s  Sp=%.2f  Se=%.2f  correction=%s",
                 gate, cal.get("Sp", 0), cal.get("Se", 0),
                 "usable" if cal.get("ok_corr") else "WEAK (Se+Sp-1<0.2)")
        th = cal["thresholds"]
        master: Dict[int, Unit] = {}
        log.info("starting measurement (%d clusters to judge) …", n_judged)
        _run_measurement(ev, cal, master, progress)
        # per-cluster estimates, then resolution waves where the verdict is unsettled
        est = {st.cid: _estimate_cluster(st, cfg, cal) for st in ev.judged}
        _run_resolution(ev, cal, est, master, progress)
        merge_groups, tau_conf = _merge_groups(ev.judged, est, cfg, cal)
        mg_of: Dict[str, int] = {}
        for gi, grp in enumerate(merge_groups):
            for cid in grp:
                mg_of[cid] = gi
        fit_verdicts, choice_stats = _run_fit(ev, mg_of, progress)
        comp_names, strategies, taxonomy = _run_presentation(ev, est, merge_groups, progress)
    return _assemble(ev.df, ev.states, ev.judged, est, ev.labels, cal, th, tau_conf, merge_groups,
                     mg_of, fit_verdicts, choice_stats, comp_names, strategies, taxonomy,
                     ev.calls_by_kind, client, cfg)


# ===========================================================================
# Ingestion, decisions, calibration math, estimation
# ===========================================================================
def _ingest_wave(wave: List[Unit], res: Dict[int, dict], states: Dict[str, CState],
                 master: Dict[int, Unit], cfg: Config) -> None:
    for u in wave:
        st = states[u.cid]
        k = len(u.payload["items"])
        gs = _groups_valid(res.get(u.uid, {}), k)
        if gs is None:
            st.parse_fail += 1
            continue
        if "dup_of" in u.truth:                            # judge-noise probe
            mv = res.get(u.truth["dup_of"], {})
            mgs = _groups_valid(mv, k)
            if mgs is not None:
                a, b = _pair_pattern(gs, k), _pair_pattern(mgs, k)
                tot = k * (k - 1) // 2
                st.dup_agree.append(1 - len(a ^ b) / tot)
            continue
        ctx = u.truth["ctx"]
        if ctx == "homog":
            ids = u.truth["item_ids"]
            co_pairs = _pair_pattern(gs, k)
            tot = k * (k - 1) // 2
            st.mix_draws.append(1 - len(co_pairs) / tot)
            for a, b in combinations(range(1, k + 1), 2):
                key = (min(ids[a - 1], ids[b - 1]), max(ids[a - 1], ids[b - 1]))
                rec = st.pair.setdefault(key, [0, 0, set()])
                rec[1] += 1
                if (a, b) in co_pairs:
                    rec[0] += 1
                    if len(rec[2]) < 8:
                        rec[2].add(u.uid)              # distinct draws that co-grouped this pair
            st.seen.update(ids)
        elif ctx == "intr":
            p = u.truth["plant_pos"]
            detected = any(g == [p] for g in gs)
            rec = st.det.setdefault(u.truth["plant_src"], [0, 0])
            rec[0] += int(detected)
            rec[1] += 1


def _mix_ci(st: CState, cfg: Config) -> Tuple[float, float, float]:
    d = st.mix_draws
    if not d:
        return 0.0, 0.0, 1.0
    m = float(np.mean(d))
    if len(d) < 3:
        return m, 0.0, 1.0
    half = cfg.z * float(np.std(d, ddof=1)) / math.sqrt(len(d))
    return m, max(0.0, m - half), min(1.0, m + half)


def _near_counts(st: CState) -> Tuple[int, int]:
    det = n = 0
    for s, (d, t) in st.det.items():
        if s != "far":
            det += d
            n += t
    return det, n


def _update_decisions(judged: List[CState], cfg: Config, th: dict) -> None:
    for st in judged:
        m, lo, hi = _mix_ci(st, cfg)
        nd = len(st.mix_draws)
        if nd >= 2 * cfg.draws_per_wave and hi < th["tau_mix_raw"]:
            st.homog_done = True                     # confidently clean: stop
        elif nd >= 4 * cfg.draws_per_wave and lo > th["tau_mix_raw"]:
            st.homog_done = True                     # confirmed mixed: extra draws bought substructure resolution
        det, n = _near_counts(st)
        if n >= 6:
            _, dlo, dhi = _wilson(det, n, cfg.z)
            if dhi < th["det_bar_raw"] or dlo > th["det_bar_raw"]:
                st.intr_done = True


def _calibrate(units: List[Unit], res: Dict[int, dict], cfg: Config) -> dict:
    same_co = [0, 0]      # co, tot on truly-same pairs        -> Sp
    diff_split = [0, 0]   # split, tot on truly-different pairs -> Se
    iso = [0, 0]          # singletons among truly-same items   -> gamma (chance isolation)
    pure_one = [0, 0]           # cross-cluster kNN gate check
    pure_cluster_one = [0, 0]   # within-cluster kNN informational only
    mixed_det = [0, 0]
    junk = [0, 0]
    farc = [0, 0]
    swap = [0, 0]
    pfail = 0
    for u in units:
        v = res.get(u.uid, {})
        if u.kind == "fit_card":
            swap[1] += 1
            swap[0] += int(v.get("verdict", "accurate") != "accurate")
            continue
        k = len(u.payload["items"])
        gs = _groups_valid(v, k)
        if gs is None:
            pfail += 1
            continue
        co = _pair_pattern(gs, k)
        ctx = u.truth["ctx"]
        if ctx == "pure":
            pure_one[1] += 1
            pure_one[0] += int(len(gs) == 1)
            tot = k * (k - 1) // 2
            same_co[0] += len(co)
            same_co[1] += tot
            iso[0] += sum(1 for g in gs if len(g) == 1)
            iso[1] += k
        elif ctx == "pure_cluster":
            # within-cluster kNN: informational only, does NOT contribute to Sp
            pure_cluster_one[1] += 1
            pure_cluster_one[0] += int(len(gs) == 1)
        elif ctx == "mixed":
            side = u.truth["side"]
            cj = ct = 0
            for a, b in combinations(range(1, k + 1), 2):
                if side[a - 1] == side[b - 1]:
                    same_co[1] += 1
                    same_co[0] += int((a, b) in co)
                else:
                    ct += 1
                    diff_split[1] += 1
                    joined = (a, b) in co
                    diff_split[0] += int(not joined)
                    cj += int(joined)
            mixed_det[1] += 1
            mixed_det[0] += int(ct > 0 and cj / ct < 0.5)
        elif ctx == "junk":
            junk[1] += 1
            junk[0] += int(any(g == [u.truth["plant_pos"]] for g in gs))
        elif ctx == "farcal":
            farc[1] += 1
            farc[0] += int(any(g == [u.truth["plant_pos"]] for g in gs))
    Sp = same_co[0] / same_co[1] if same_co[1] else 0.9
    Se = diff_split[0] / diff_split[1] if diff_split[1] else 0.9
    gamma = iso[0] / iso[1] if iso[1] else 0.02
    denom = Se + Sp - 1
    ok_corr = denom >= 0.2
    rates = lambda c: (c[0] / c[1]) if c[1] else None
    checks = {
        "pure_kept_whole": {"score": rates(pure_one), "want": ">=0.70",
                            "pass": (rates(pure_one) or 0) >= 0.70, "n": pure_one[1],
                            "gate": True},
        "cluster_purity": {"score": rates(pure_cluster_one), "want": "informational",
                           "pass": True, "n": pure_cluster_one[1],
                           "gate": False},
        "mixture_split": {"score": rates(mixed_det), "want": ">=0.70",
                          "pass": (rates(mixed_det) or 0) >= 0.70, "n": mixed_det[1],
                          "gate": True},
        "junk_isolated": {"score": rates(junk), "want": ">=0.80",
                          "pass": (rates(junk) or 0) >= 0.80, "n": junk[1],
                          "gate": True},
        "far_intruder_found": {"score": rates(farc), "want": ">=0.80",
                               "pass": (rates(farc) or 0) >= 0.80, "n": farc[1],
                               "gate": True},
        "swapped_label_flagged": {"score": rates(swap), "want": ">=0.70",
                                  "pass": (rates(swap) or 0) >= 0.70, "n": swap[1],
                                  "gate": True},
        "correction_usable": {"score": round(denom, 3), "want": "Se+Sp-1>=0.20",
                              "pass": ok_corr, "n": same_co[1] + diff_split[1],
                              "gate": True},
    }
    far_rate = rates(farc) or 0.9
    th = {
        "tau_edge": (Sp + (1 - Se)) / 2,
        "tau_mix_raw": (cfg.tau_mix * denom + (1 - Sp)) if ok_corr else cfg.tau_mix,
        "det_bar_corr": 0.5,
        "det_bar_raw": 0.5 * (1 - gamma) + gamma,
        "far_rate": far_rate,
    }
    return {"Sp": Sp, "Se": Se, "gamma": gamma, "denom": denom, "ok_corr": ok_corr,
            "parse_fail": pfail, "checks": checks, "thresholds": th,
            "overall_pass": all(c["pass"] for c in checks.values() if c.get("gate", True)),
            "n_calls": len(units)}


def _judge_health(client: "LLMClient", cfg: Config) -> dict:
    """Infrastructure guard: the share of judge calls that came back empty/unparseable.
    Distinct from the calibration gate (which measures judge *quality* on planted controls)
    — this catches a *dead or failing* gateway, where a run can otherwise complete on almost
    no data and still print confident numbers. Mock runs never fail, so health is always ok."""
    n, empty = client.n_calls, client.n_empty
    rate = (empty / n) if n else 0.0
    ok = rate <= cfg.max_empty_rate
    if not ok:
        log.error("judge health: %d/%d calls (%.0f%%) returned empty/unparseable — above the "
                  "%.0f%% ceiling; treating this run as untrustworthy",
                  empty, n, 100 * rate, 100 * cfg.max_empty_rate)
    return {"n_calls": n, "n_empty": empty, "empty_rate": round(rate, 3),
            "max_empty_rate": cfg.max_empty_rate, "ok": ok}


def _rg_mix(m: float, cal: dict) -> float:
    """Rogan-Gladen: observed mixing -> true mixing, given pair-level Se/Sp."""
    if not cal["ok_corr"]:
        return m
    return min(1.0, max(0.0, (m - (1 - cal["Sp"])) / cal["denom"]))


def _rg_det(d: float, cal: dict) -> float:
    """Detection corrected for chance isolation."""
    g = cal["gamma"]
    return min(1.0, max(0.0, (d - g) / (1 - g))) if g < 1 else d


def _components_from_pairs(seen: set, pair: Dict[Tuple[int, int], list],
                           tau_edge: float, sp: float, se: float) -> List[List[int]]:
    """Greedy max-LR agglomeration of seen items into same-kind components, gated by:
      rate    pooled cross co-rate >= tau_edge (calibrated midpoint between Sp and 1-Se)
      LR      pooled log-likelihood ratio >= log 9 under this run's measured Sp/Se
      draws   comp-comp merges need cross-boundary co-evidence from >= 2 DISTINCT draws.
              A single false-join draw of k items mints a whole clique of co=1 pairs that
              all share ONE draw-id, so the union of draw-ids across the boundary is 1 and
              the merge is blocked. Genuine same-kind structure is co-grouped in several
              independent draws (re-shuffled passes), so its draw-id union grows past 1.
      Singleton attachment is exempt (misplacing one item is a bounded ~1/n_seen
      prevalence error; orphaning it is certain information loss)."""
    sp = min(0.99, max(0.01, sp))
    se = min(0.99, max(0.01, se))
    lr_same, lr_diff = math.log(sp / (1 - se)), math.log((1 - sp) / se)
    LR_MIN = math.log(9)
    comps: Dict[int, set] = {i: {i} for i in seen}
    # comp-pair aggregates: (ca,cb) -> [co, tot, union_of_co_draw_ids]
    agg: Dict[Tuple[int, int], list] = {}
    for (a, b), rec in pair.items():
        co, tot = rec[0], rec[1]
        dr = set(rec[2]) if len(rec) > 2 else set()
        agg[(min(a, b), max(a, b))] = [co, tot, dr]
    while True:
        best = None
        for (ca, cb), (co, tot, dr) in agg.items():
            if not tot:
                continue
            rate = co / tot
            lr = co * lr_same + (tot - co) * lr_diff
            if rate < tau_edge or lr < LR_MIN:
                continue
            if min(len(comps[ca]), len(comps[cb])) > 1 and len(dr) < 2:
                continue
            if best is None or lr > best[0]:
                best = (lr, ca, cb)
        if best is None:
            break
        _, ca, cb = best
        comps[ca] |= comps.pop(cb)
        merged: Dict[Tuple[int, int], list] = {}
        for (x, y), v in agg.items():
            if cb in (x, y):
                other = y if x == cb else x
                if other == ca:
                    continue
                k2 = (min(ca, other), max(ca, other))
            else:
                k2 = (x, y)
            r = merged.setdefault(k2, [0, 0, set()])
            r[0] += v[0]
            r[1] += v[1]
            r[2] |= v[2]
        agg = merged
    return sorted((sorted(c) for c in comps.values()), key=len, reverse=True)


def _estimate_cluster(st: CState, cfg: Config, cal: dict) -> dict:
    m, lo, hi = _mix_ci(st, cfg)
    homog = {"score": round(1 - _rg_mix(m, cal), 3),
             "lo": round(1 - _rg_mix(hi, cal), 3), "hi": round(1 - _rg_mix(lo, cal), 3)}
    comp_items = _components_from_pairs(st.seen, st.pair, cal["thresholds"]["tau_edge"],
                                        cal["Sp"], cal["Se"])
    n_seen = max(1, len(st.seen))
    # Component reconstruction is high-variance at low coverage: a single pure theme
    # routinely shatters into several above-floor fragments, and reporting those as
    # distinct sub-themes would overstate structure ("4 kinds" when there are 2). The
    # honest, decision-useful resolution is dominant-vs-remainder: the largest component
    # is the named major; everything else pools into one residual ("mixed / other").
    # Whether the cluster actually splits is decided separately by corrected mixing.
    dom = comp_items[0] if comp_items else sorted(st.seen)
    dom_share = len(dom) / n_seen
    residual = [i for c in comp_items[1:] for i in c]
    res_share = len(residual) / n_seen
    components, comp_payload = [], []          # parallel: (items, is_residual)
    if dom_share >= cfg.tau_mix:
        _, slo, shi = _wilson(len(dom), n_seen)
        components.append({"frac": round(dom_share, 3), "lo": round(slo, 3), "hi": round(shi, 3),
                           "n": len(dom), "residual": False})
        comp_payload.append((sorted(dom), False))
    if res_share >= cfg.tau_mix and residual:
        _, slo, shi = _wilson(len(residual), n_seen)
        components.append({"frac": round(res_share, 3), "lo": round(slo, 3), "hi": round(shi, 3),
                           "n": len(residual), "residual": True})
        comp_payload.append((sorted(residual), True))
    components.sort(key=lambda d: -d["frac"])
    comp_payload.sort(key=lambda t: -len(t[0]))
    # The split verdict keys off CORRECTED MIXING, not component count: mixing is the
    # low-variance pair statistic, while the component partition is a high-variance
    # refinement that fragments even a pure cluster at low coverage. Split when the
    # mixing CI lower bound clears tau_mix (confident real mixing exceeds tolerance);
    # one_class when the CI upper bound is below it. Components only carry meaning under
    # a split verdict, so a homogeneous cluster is reported as a single class regardless
    # of incidental graph fragmentation.
    mix_corr = _rg_mix(m, cal)                  # m, lo, hi already from _mix_ci at the top
    split = _rg_mix(lo, cal) > cfg.tau_mix
    one_class = _rg_mix(hi, cal) <= cfg.tau_mix
    if not split:                               # collapse fragments; the verdict is "one kind"
        allitems = sorted(st.seen)
        components = [{"frac": 1.0, "lo": round(_wilson(len(allitems), n_seen)[1], 3),
                       "hi": 1.0, "n": len(allitems), "residual": False}]
        comp_payload = [(allitems, False)]
    n_classes = len(components)
    minor = 1 - sum(d["frac"] for d in components)
    # distinctiveness + per-neighbour confusion
    det, n = _near_counts(st)
    if n:
        d, dl, dh = _wilson(det, n, cfg.z)
        distinct = {"score": round(_rg_det(d, cal), 3), "lo": round(_rg_det(dl, cal), 3),
                    "hi": round(_rg_det(dh, cal), 3), "n": n}
    else:
        distinct = None
    far = st.det.get("far")
    far_rate = round(_rg_det(far[0] / far[1], cal), 3) if far and far[1] else None
    confus = []
    for s, (d_, t_) in st.det.items():
        if s == "far" or t_ < 2:
            continue
        _, _, dh_ = _wilson(d_, t_, cfg.z)          # conservative: confusion lower bound
        confus.append({"cluster_id": s, "confusion": round(1 - _rg_det(d_ / t_, cal), 3),
                       "lo": round(1 - _rg_det(dh_, cal), 3), "n": t_})
    confus.sort(key=lambda x: -x["confusion"])
    ju = bool((len(st.dup_agree) >= 2 and float(np.mean(st.dup_agree)) < 0.7)
              or (len(st.dup_agree) == 1 and st.dup_agree[0] < 0.5))
    return {"mixing_raw": round(m, 3), "mixing": round(mix_corr, 3), "homogeneity": homog,
            "n_draws": len(st.mix_draws),
            "components": components, "_comp_items": comp_payload,
            "n_classes": n_classes, "minor_share": round(max(0.0, minor), 3),
            "split": split, "one_class": one_class,
            "distinctiveness": distinct, "far_detect": far_rate,
            "confusable_with": confus, "judge_uncertain": ju,
            "parse_fail": st.parse_fail,
            "dup_agreement": round(float(np.mean(st.dup_agree)), 3) if st.dup_agree else None}


def _merge_groups(judged: List[CState], est: Dict[str, dict], cfg: Config,
                  cal: dict) -> Tuple[List[List[str]], float]:
    # threshold lives in the SAME statistic space as the edges (confusion lower bound):
    #   negative control = far strata lo's (should be ~0); positive anchor = lo of full
    #   confusion (0 detections) at the typical per-stratum n. tau = their midpoint.
    lo_far: List[float] = []
    n_near: List[int] = []
    for st in judged:
        far = st.det.get("far")
        if far and far[1]:
            _, _, dh = _wilson(far[0], far[1], cfg.z)
            lo_far.append(1 - _rg_det(dh, cal))
        n_near += [t for s, (d, t) in st.det.items() if s != "far"]
    n_typ = int(np.median(n_near)) if n_near else 3
    _, _, dh0 = _wilson(0, max(1, n_typ), cfg.z)
    lo_full = 1 - _rg_det(dh0, cal)
    neg = float(np.quantile(lo_far, 0.95)) if lo_far else 0.05
    tau_conf = min(0.9, max(0.15, (neg + lo_full) / 2))
    pair_conf: Dict[Tuple[str, str], List[Tuple[float, int]]] = defaultdict(list)
    for st in judged:
        for c in est[st.cid]["confusable_with"]:
            a, b = sorted((st.cid, c["cluster_id"]))
            pair_conf[(a, b)].append((c["lo"], c["n"]))
    parent: Dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (a, b), ent in pair_conf.items():
        los = [e[0] for e in ent]
        # both measured directions must clear tau independently; a single direction
        # needs n>=3 — so one noise-missed plant can never fabricate a merge edge
        ok = (min(los) >= tau_conf) if len(ent) >= 2 else (los[0] >= tau_conf and ent[0][1] >= 3)
        if ok:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    groups: Dict[str, List[str]] = defaultdict(list)
    for cid in list(parent):
        groups[find(cid)].append(cid)
    return sorted([sorted(g) for g in groups.values() if len(g) >= 2], key=len, reverse=True), tau_conf


# ===========================================================================
# Targeted re-clustering of a single flagged cluster (high-resolution pass)
# ===========================================================================
def _dense_partition_draws(order: List[int], k: int, redundancy: int,
                           prng: random.Random) -> List[List[int]]:
    """Cover `order` with k-item PARTITION blocks so each item appears in ~`redundancy`
    draws. Each pass is an independent shuffle, so a given within-kind pair gets several
    chances to be co-observed (the draw-replication the component graph needs), while a
    one-off false-join only ever appears in its single block."""
    draws: List[List[int]] = []
    n = len(order)
    if n <= k:                                  # whole cluster fits in one block
        return [list(order)] * max(1, redundancy)
    for _ in range(max(1, redundancy)):
        perm = list(order)
        prng.shuffle(perm)
        step = max(1, k // 2)                    # 50% overlap chains the co-graph within a pass
        i = 0
        while i < n:
            block = perm[i:i + k]
            if len(block) < k:                   # wrap the tail so every item is in a full block
                block = block + perm[:k - len(block)]
            draws.append(block)
            i += step
    return draws


def _mini_calibrate(cluster_idx: np.ndarray, neigh_idx: Optional[np.ndarray],
                    emb_n: np.ndarray, text_arr: np.ndarray, theme_arr, cfg: Config,
                    client: "LLMClient", prng: random.Random, rng: np.random.Generator,
                    progress: bool) -> dict:
    """Lightweight Sp/Se calibration for a standalone pass: pure draws from the cluster
    itself (truly-same pairs) and mixed draws splicing in far items (truly-different
    pairs). Returns the same thresholds dict the main calibrator does."""
    units: List[Unit] = []
    uid = [0]

    def U(payload, truth):
        uid[0] += 1
        return Unit(uid[0], "partition", "_cal_", payload, truth)

    def items_of(ids):
        return [_clip(text_arr[i], cfg.item_chars) for i in ids]

    def themes_of(ids):
        return [str(theme_arr[i]) for i in ids] if theme_arr is not None else None

    k = cfg.k_partition
    pool = cluster_idx
    for _ in range(cfg.n_pure):
        if len(pool) < k:
            break
        center = int(prng.choice(list(pool)))
        s = emb_n[pool] @ emb_n[center]
        ids = [int(x) for x in pool[np.argsort(-s)[:k]]]
        prng.shuffle(ids)
        units.append(U({"items": items_of(ids)}, {"ctx": "pure", "themes": themes_of(ids)}))
    if neigh_idx is not None and len(neigh_idx):
        for _ in range(cfg.n_mixed):
            ka = k // 2
            if len(pool) < ka or len(neigh_idx) < k - ka:
                break
            ia = [int(x) for x in rng.choice(pool, size=ka, replace=False)]
            ib = [int(x) for x in rng.choice(neigh_idx, size=k - ka, replace=False)]
            ids = ia + ib
            side = [0] * len(ia) + [1] * len(ib)
            o = list(range(len(ids)))
            prng.shuffle(o)
            units.append(U({"items": items_of([ids[i] for i in o])},
                           {"ctx": "mixed", "side": [side[i] for i in o],
                            "themes": themes_of([ids[i] for i in o])}))
    res = run_units(units, client, cfg, desc="recluster-cal", progress=progress)
    return _calibrate(units, res, cfg)


def recluster_cluster(cluster_id: str, data=None, embeddings=None, *, results: dict = None,
                      config: Optional[Config] = None, same_when=None, unit=None, use=None,
                      text_col="text", cluster_col="cluster_id", label_col="label",
                      description_col=None, embedding_col=None, theme_col=None,
                      client: Optional["LLMClient"] = None, progress: bool = True) -> dict:
    """Resolve ONE flagged cluster into named, itemized sub-clusters at high coverage.

    Run this only on clusters that `evaluate` flagged `split` — it is the expensive,
    high-resolution counterpart to the thin verdict pass. It pools up to
    `recluster_max_items` of the cluster, has the judge PARTITION them under `same_when`
    with each item appearing in several overlapping draws, resolves the co-grouping graph
    into sub-clusters (calibrated + draw-replication gated, exactly as the main engine),
    names each, and assigns every pooled member — with an optional FIT-choice pass to
    place items that landed in sub-floor fragments.

    `data`/`embeddings` are the SAME inputs you pass to evaluate_clusters (the full
    dataset is fine; only rows with this cluster_id are used). Returns a dict; pass it to
    render_recluster_report() for text."""
    cfg = config or Config()
    if same_when is not None:
        cfg.same_when = same_when
    if unit is not None:
        cfg.unit = unit
    if use is not None:
        cfg.use_context = use
    if not cfg.same_when:
        raise ValueError("Config.same_when is required — the equivalence rule drives the split")

    df, emb, labels = _coerce_inputs(data, embeddings, text_col=text_col, cluster_col=cluster_col,
                                     label_col=label_col, description_col=description_col,
                                     embedding_col=embedding_col, theme_col=theme_col)
    df = df.reset_index(drop=True)
    cid_str = df["cluster_id"].astype(str)      # _coerce_inputs has already renamed cluster_col -> cluster_id
    mask = (cid_str == str(cluster_id)).to_numpy()
    if not mask.any():
        raise ValueError(f"cluster_id {cluster_id!r} not found")
    cluster_idx = np.where(mask)[0]
    emb_n = _normalize(np.asarray(emb, dtype=np.float32))
    text_arr = df["text"].astype(str).to_numpy()
    theme_arr = df["_theme"].to_numpy() if "_theme" in df.columns else None
    client = client or _make_client(cfg)
    prng = random.Random(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    # dense pool of THIS cluster
    n_total = len(cluster_idx)
    target = int(min(cfg.recluster_max_items, max(cfg.min_pool, round(n_total * cfg.recluster_coverage))))
    if n_total <= target:
        pool = cluster_idx.copy()
    else:
        # representative: micro-kmeans proportional, same idea as _pool but to the bigger cap
        from sklearn.cluster import MiniBatchKMeans
        mk = min(cfg.micro_k * 3, max(2, target // 8))
        km = MiniBatchKMeans(n_clusters=mk, random_state=cfg.seed, n_init=3).fit(emb_n[cluster_idx])
        sel: List[int] = []
        for m in range(mk):
            mem = cluster_idx[km.labels_ == m]
            if len(mem) == 0:
                continue
            take = min(len(mem), max(1, round(target * len(mem) / n_total)))
            sel += [int(x) for x in rng.choice(mem, size=take, replace=False)]
        pool = np.asarray(sel[:target]) if sel else rng.choice(cluster_idx, size=target, replace=False)
    pool = np.asarray(sorted(int(x) for x in pool))

    # neighbour pool (other clusters' items) for mini-calibration mixed draws
    neigh_idx = np.where(~mask)[0]
    if len(neigh_idx) > 2000:
        neigh_idx = rng.choice(neigh_idx, size=2000, replace=False)
    cal = (results or {}).get("calibration") if results else None
    if not (cal and cal.get("ok_corr")):
        cal = _mini_calibrate(pool, neigh_idx if len(neigh_idx) else None, emb_n, text_arr,
                              theme_arr, cfg, client, prng, rng, progress)
    th = cal["thresholds"]

    # dense PARTITION sampling over the pool
    order = list(map(int, pool))
    blocks = _dense_partition_draws(order, cfg.k_partition, cfg.recluster_redundancy, prng)
    uid = [0]
    units: List[Unit] = []
    for b in blocks:
        uid[0] += 1
        units.append(Unit(uid[0], "partition", str(cluster_id),
                          {"items": [_clip(text_arr[i], cfg.item_chars) for i in b]},
                          {"ctx": "homog", "item_ids": list(b),
                           "themes": [str(theme_arr[i]) for i in b] if theme_arr is not None else None}))
    units = units[: max(0, cfg.max_llm_calls - client.n_calls)]
    res = run_units(units, client, cfg, desc=f"recluster {cluster_id}", progress=progress)

    # accumulate co-grouping pairs (with draw-ids), same bookkeeping as _ingest_wave
    pair: Dict[Tuple[int, int], list] = {}
    seen: set = set()
    parse_fail = 0
    for u in units:
        k = len(u.payload["items"])
        gs = _groups_valid(res.get(u.uid, {}), k)
        if gs is None:
            parse_fail += 1
            continue
        ids = u.truth["item_ids"]
        co = _pair_pattern(gs, k)
        for a, b in combinations(range(1, k + 1), 2):
            key = (min(ids[a - 1], ids[b - 1]), max(ids[a - 1], ids[b - 1]))
            rec = pair.setdefault(key, [0, 0, set()])
            rec[1] += 1
            if (a, b) in co:
                rec[0] += 1
                if len(rec[2]) < 8:
                    rec[2].add(u.uid)
        seen.update(ids)

    comp_items = _components_from_pairs(seen, pair, th["tau_edge"], cal["Sp"], cal["Se"])
    n_seen = max(1, len(seen))
    majors = [c for c in comp_items if len(c) / n_seen >= cfg.tau_mix]
    fragments = [c for c in comp_items if len(c) / n_seen < cfg.tau_mix]
    residual = [i for c in fragments for i in c]

    # name each major sub-cluster
    name_units: List[Unit] = []
    for ci, comp in enumerate(majors):
        ex = comp if len(comp) <= cfg.recluster_name_exemplars else \
            [int(x) for x in rng.choice(comp, size=cfg.recluster_name_exemplars, replace=False)]
        uid[0] += 1
        name_units.append(Unit(uid[0], "name", str(cluster_id),
                               {"items": [_clip(text_arr[i], cfg.item_chars) for i in ex]},
                               {"comp": ci, "themes": [str(theme_arr[i]) for i in ex] if theme_arr is not None else None}))
    name_units = name_units[: max(0, cfg.max_llm_calls - client.n_calls)]
    name_res = run_units(name_units, client, cfg, desc="naming subs", progress=progress)
    names = {u.truth["comp"]: str(name_res.get(u.uid, {}).get("name", f"sub-cluster {u.truth['comp'] + 1}"))
             for u in name_units}

    # assign sub-floor-fragment items onto the named subs via FIT-choice (optional)
    assigned_residual: Dict[int, int] = {}
    if cfg.recluster_assign_residual and residual and majors:
        cards = [{"label": names.get(ci, f"sub-cluster {ci + 1}"),
                  "description": ""} for ci in range(len(majors))]
        ch_units: List[Unit] = []
        for it in residual:
            o = list(range(len(cards)))
            prng.shuffle(o)
            uid[0] += 1
            ch_units.append(Unit(uid[0], "fit_choice", str(cluster_id),
                                 {"item": _clip(text_arr[it], cfg.item_chars),
                                  "cards": [cards[j] for j in o]},
                                 {"item_id": it, "order": o,
                                  "card_themes": None}))
        ch_units = ch_units[: max(0, cfg.max_llm_calls - client.n_calls)]
        ch_res = run_units(ch_units, client, cfg, desc="placing residual", progress=progress)
        for u in ch_units:
            ch = ch_res.get(u.uid, {}).get("choice")
            if isinstance(ch, int) and 1 <= ch <= len(u.truth["order"]):
                assigned_residual[u.truth["item_id"]] = u.truth["order"][ch - 1]

    # assemble sub-clusters
    members_by_sub: Dict[int, List[int]] = {ci: list(comp) for ci, comp in enumerate(majors)}
    for it, ci in assigned_residual.items():
        members_by_sub.setdefault(ci, []).append(it)
    placed_residual = set(assigned_residual)
    leftover = [i for i in residual if i not in placed_residual]

    subs = []
    for ci, comp in enumerate(majors):
        mem = members_by_sub.get(ci, [])
        _, lo, hi = _wilson(len(mem), n_seen)
        ex = [int(x) for x in (comp if len(comp) <= 5 else comp[:5])]
        subs.append({"sub_id": ci, "name": names.get(ci, f"sub-cluster {ci + 1}"),
                     "n": len(mem), "frac": round(len(mem) / n_seen, 3),
                     "frac_lo": round(lo, 3), "frac_hi": round(hi, 3),
                     "exemplars": [_clip(text_arr[i], cfg.item_chars) for i in ex],
                     "member_row_indices": sorted(int(x) for x in mem)})
    subs.sort(key=lambda s: -s["n"])
    if leftover:
        subs.append({"sub_id": -1, "name": "unresolved / mixed",
                     "n": len(leftover), "frac": round(len(leftover) / n_seen, 3),
                     "frac_lo": None, "frac_hi": None,
                     "exemplars": [_clip(text_arr[i], cfg.item_chars) for i in leftover[:5]],
                     "member_row_indices": sorted(int(x) for x in leftover)})

    return {"cluster_id": str(cluster_id), "label": labels.get(str(cluster_id), {}).get("label", str(cluster_id)),
            "size": int(n_total), "pooled": int(len(pool)),
            "coverage": round(len(pool) / n_total, 3), "n_seen": int(len(seen)),
            "n_sub_clusters": len(majors), "had_leftover": bool(leftover),
            "sub_clusters": subs,
            "calibration": {"Sp": cal["Sp"], "Se": cal["Se"], "ok_corr": cal["ok_corr"],
                            "overall_pass": cal.get("overall_pass"), "source": "run" if results and (results.get("calibration") or {}).get("ok_corr") else "local"},
            "meta": {"same_when": cfg.same_when, "unit": cfg.unit,
                     "n_partition_draws": len(blocks), "redundancy": cfg.recluster_redundancy,
                     "k_partition": cfg.k_partition, "tau_edge": round(th["tau_edge"], 3),
                     "tau_mix": cfg.tau_mix, "n_llm_calls": client.n_calls,
                     "parse_fail": parse_fail, "judge_health": _judge_health(client, cfg),
                     "model": "mock" if client.mock else cfg.model}}


def render_recluster_report(r: dict) -> str:
    L: List[str] = []
    bar = "=" * 88
    L += [bar, f"RE-CLUSTERING — {r['cluster_id']}  «{_clip(r['label'], 60)}»", bar]
    L += [f"rule  two items are the same kind when they {r['meta']['same_when']}",
          f"size  {r['size']:,} items · pooled {r['pooled']:,} ({r['coverage'] * 100:.0f}% coverage) · "
          f"seen {r['n_seen']:,} · {r['meta']['n_partition_draws']} draws ×{r['meta']['redundancy']} · "
          f"judge calls {r['meta']['n_llm_calls']:,} · model {r['meta']['model']}"]
    c = r["calibration"]
    gate = "usable" if c.get("ok_corr") else "WEAK (Se+Sp-1<0.2 — sub-clusters are tentative)"
    L += [f"calib  Sp {c['Sp']:.2f} · Se {c['Se']:.2f} · correction {gate} · source {c['source']}"]
    hb = r["meta"].get("judge_health")
    if hb and not hb["ok"]:
        L += [f"  ⚠ judge health: {hb['n_empty']:,}/{hb['n_calls']:,} calls "
              f"({hb['empty_rate'] * 100:.0f}%) empty/unparseable — gateway unhealthy; "
              f"the sub-clusters below are built on partial data."]
    L += [f"found  {r['n_sub_clusters']} sub-cluster(s)" + (" + an unresolved remainder" if r['had_leftover'] else ""),
          ""]
    rows = [[str(s["sub_id"]) if s["sub_id"] >= 0 else "—", _clip(s["name"], 38),
             f"{s['n']:,}", f"{s['frac'] * 100:.0f}%",
             f"[{s['frac_lo'] * 100:.0f},{s['frac_hi'] * 100:.0f}]" if s["frac_lo"] is not None else "—"]
            for s in r["sub_clusters"]]
    L += ["  " + ln for ln in _tbl(["sub", "name", "n", "share", "CI%"], rows, rj={2, 3, 4})]
    L += ["", "EXEMPLARS"]
    for s in r["sub_clusters"]:
        L += [f"  [{s['sub_id'] if s['sub_id'] >= 0 else '—'}] {_clip(s['name'], 44)}  ({s['n']:,}, {s['frac'] * 100:.0f}%)"]
        for ex in s["exemplars"][:3]:
            L += [f"        · {_clip(ex, 76)}"]
    L += ["", "NOTES",
          "  member_row_indices in the JSON map each sub-cluster back to rows of your input.",
          "  shares are over SEEN (pooled-and-judged) items; CIs are Wilson on those counts.",
          "  an 'unresolved / mixed' bucket holds items the judge could not place confidently.",
          bar]
    return "\n".join(L)



_RATE_KEYS = ("homogeneous", "distinct", "fit_accurate", "split", "review")


def _assemble(df, states, judged, est, labels, cal, th, tau_conf, merge_groups, mg_of,
              fit_verdicts, choice_stats, comp_names, strategies, taxonomy,
              calls_by_kind, client, cfg) -> "Results":
    bands = _fit_bands([s.size for s in states.values()], cfg)
    clusters = []
    for st in sorted(states.values(), key=lambda s: -s.size):
        c: Dict[str, Any] = {"cluster_id": st.cid, "label": labels[st.cid]["label"],
                             "description": labels[st.cid]["description"], "size": st.size,
                             "band": _band_of(st.size, bands),
                             "coverage": round(len(st.pool) / st.size, 3), "judged": st.judged}
        if st.judged:
            e = est[st.cid]
            vs = [v.get("verdict", "accurate") for v in fit_verdicts.get(st.cid, []) if v]
            verdict = max(set(vs), key=vs.count) if vs else None
            prop = next((v for v in fit_verdicts.get(st.cid, []) if v.get("verdict") == verdict), {})
            comps = [{"name": ("mixed / other" if comp.get("residual")
                               else comp_names.get((st.cid, i), f"sub-theme {i + 1}")), **comp}
                     for i, comp in enumerate(e["components"])]
            reasons = []
            if e["split"]:                       # corrected mixing CI above tolerance
                reasons.append("split")
            elif not e["one_class"]:             # CI straddles tau_mix -> unresolved
                reasons.append("mixing_unclear")
            if e["distinctiveness"] and e["distinctiveness"]["score"] < th["det_bar_corr"]:
                reasons.append("indistinct")
            if mg_of.get(st.cid) is not None:
                reasons.append("redundant")
            if verdict and verdict != "accurate":
                reasons.append("label")
            if e["judge_uncertain"]:
                reasons.append("judge_uncertain")
            ch = choice_stats.get(st.cid)
            c.update({"n_draws": e["n_draws"], "homogeneity": e["homogeneity"],
                      "mixing": e["mixing"], "mixing_raw": e["mixing_raw"], "components": comps,
                      "n_classes": e["n_classes"], "minor_share": e["minor_share"],
                      "split": e["split"], "one_class": e["one_class"],
                      "distinctiveness": e["distinctiveness"], "far_detect": e["far_detect"],
                      "confusable_with": e["confusable_with"][:5],
                      "merge_group": mg_of.get(st.cid),
                      "fit": {"verdict": verdict,
                              "proposed_label": prop.get("proposed_label"),
                              "proposed_description": prop.get("proposed_description")},
                      "placement_accuracy": round(ch[0] / ch[1], 3) if ch and ch[1] else None,
                      "strategies": strategies.get(st.cid, {}).get("strategies", []),
                      "mineable": strategies.get(st.cid, {}).get("mineable"),
                      "judge_uncertain": e["judge_uncertain"],
                      "dup_agreement": e["dup_agreement"], "parse_fail": e["parse_fail"],
                      "review": bool(reasons), "review_reasons": reasons})
        else:
            c.update({"review": False, "review_reasons": [], "note": "too small to judge"})
        clusters.append(c)

    def rates(rows, weighted=False):
        out = {}
        for key in _RATE_KEYS:
            num = den = 0.0
            for r in rows:
                if not r["judged"]:
                    continue
                v = {"homogeneous": r["one_class"],
                     "distinct": (r["distinctiveness"]["score"] >= th["det_bar_corr"])
                     if r["distinctiveness"] else None,
                     "fit_accurate": (r["fit"]["verdict"] == "accurate") if r["fit"]["verdict"] else None,
                     "split": r["split"], "review": r["review"]}[key]
                if v is None:
                    continue
                w = r["size"] if weighted else 1.0
                num += w * float(v)
                den += w
            out[key] = round(num / den, 3) if den else None
        return out

    by_band = {}
    for b in bands:
        rows = [c for c in clusters if c["band"] == b["band"]]
        by_band[b["band"]] = {"n_clusters": len(rows), "n_texts": int(sum(r["size"] for r in rows)),
                              **rates(rows)}
    dup_all = [a for s in judged for a in s.dup_agree]
    health = _judge_health(client, cfg)
    cal_out = {k: v for k, v in cal.items() if k not in ("thresholds", "checks")}
    cal_out.update({"checks": cal["checks"], "judge_health": health,
                    "dup_disagreement": round(1 - float(np.mean(dup_all)), 3) if dup_all else None})
    if not health["ok"]:
        cal_out["overall_pass"] = False          # a failing gateway invalidates every downstream number
    meta = {"unit": cfg.unit, "same_when": cfg.same_when, "use": cfg.use_context,
            "model": "mock" if client.mock else cfg.model,
            "n_texts": int(len(df)), "n_clusters": len(states),
            "n_judged": len(judged), "n_llm_calls": client.n_calls,
            "n_empty": client.n_empty, "empty_rate": health["empty_rate"],
            "coverage_target": cfg.coverage_target, "k_partition": cfg.k_partition,
            "calls_by_kind": dict(calls_by_kind),
            "thresholds": {k: round(float(v), 3) for k, v in th.items()} | {"tau_conf": round(tau_conf, 3),
                                                                            "tau_mix": cfg.tau_mix},
            "seed": cfg.seed}
    for c in clusters:                                   # drop internals
        c.pop("_comp_items", None)
    return {"meta": meta, "calibration": cal_out, "kpis": rates(clusters),
            "kpis_weighted": rates(clusters, weighted=True), "by_band": by_band, "bands": bands,
            "clusters": clusters, "merge_groups": merge_groups, "taxonomy": taxonomy}


# ===========================================================================
# Text report
# ===========================================================================
def _p(x, dash="   —") -> str:
    return dash if x is None else f"{100 * x:4.0f}%"


def _tbl(headers: List[str], rows: List[List[str]], rj: Optional[set] = None) -> List[str]:
    rj = rj or set()
    w = [max(len(str(headers[i])), *(len(str(r[i])) for r in rows)) if rows else len(str(headers[i]))
         for i in range(len(headers))]
    fmt = lambda r: "  ".join(str(x).rjust(w[i]) if i in rj else str(x).ljust(w[i]) for i, x in enumerate(r))
    return [fmt(headers), "  ".join(("-" * w[i]) for i in range(len(headers)))] + [fmt(r) for r in rows]


def _rjust(headers: List[str], names: set) -> set:
    """Right-justify columns selected by name, so inserting a column can't shift the set."""
    return {i for i, h in enumerate(headers) if h in names}


def render_report(results: "Results", top_n: int = 25) -> str:
    m, cal, th = results["meta"], results["calibration"], results["meta"]["thresholds"]
    L: List[str] = []
    bar = "=" * 100
    L += [bar, "CLUSTER QUALITY REPORT — reference-free LLM-as-judge (cluster_judge v3)", bar]
    L += [f"rule  two items are the same kind when they {m['same_when']}"]
    if m.get("use"):
        L += [f"use   {m['use']}"]
    L += [f"unit  {m['unit']}",
          f"data  {m['n_texts']:,} texts · {m['n_clusters']} clusters ({m['n_judged']} judged, "
          f"{m['n_clusters'] - m['n_judged']} too small) · judge calls {m['n_llm_calls']:,} · model {m['model']}",
          ""]
    # gate
    gate = "PASS — the judge separates planted good from planted bad; numbers below are trustworthy." \
        if cal["overall_pass"] else "FAIL — a planted test failed; treat every number below with suspicion."
    L += [f"CALIBRATION GATE: {gate}"]
    if not cal.get("overall_pass"):
        L += ["  → See calibration_review.html for per-call details and human review form"]
    hb = cal.get("judge_health")
    if hb and not hb["ok"]:
        L += [f"  ⚠ JUDGE HEALTH: {hb['n_empty']:,}/{hb['n_calls']:,} calls "
              f"({hb['empty_rate'] * 100:.0f}%) returned empty/unparseable — above the "
              f"{hb['max_empty_rate'] * 100:.0f}% ceiling. The gateway looks unhealthy; this run is "
              f"built on partial data and is NOT trustworthy."]
    elif hb and hb["n_empty"]:
        L += [f"  judge health: {hb['n_empty']:,}/{hb['n_calls']:,} calls "
              f"({hb['empty_rate'] * 100:.0f}%) empty/unparseable (within tolerance)."]
    rows = []
    for k_, c in cal["checks"].items():
        sc = c["score"]
        s = f"{sc:.2f}" if k_ == "correction_usable" else (_p(sc) if isinstance(sc, (int, float)) else str(sc))
        rows.append([k_.replace("_", " "), s, c["want"], "pass" if c["pass"] else "FAIL", str(c["n"])])
    _checks_hdr = ["check", "score", "target", "verdict", "n"]
    L += ["  " + ln for ln in _tbl(_checks_hdr, rows, rj=_rjust(_checks_hdr, {"score", "n"}))]
    L += [f"  measured judge error: splits-true-pairs {(1 - cal['Sp']) * 100:.1f}% · "
          f"joins-different-pairs {(1 - cal['Se']) * 100:.1f}% · chance isolation {cal['gamma'] * 100:.1f}%"
          + (f" · duplicate disagreement {cal['dup_disagreement'] * 100:.1f}%" if cal.get("dup_disagreement") is not None else "")]
    L += [f"  derived thresholds: co-edge τ {th['tau_edge']:.2f} · mixing pass ≤{th['tau_mix']:.2f} "
          f"(raw ≤{th['tau_mix_raw']:.2f}) · detection pass ≥{th['det_bar_corr']:.2f} · merge τ {th['tau_conf']:.2f}",
          "  (all scores below are corrected for these error rates — Rogan-Gladen)", ""]
    # summary matrix
    band_names = [b["band"] for b in results["bands"]]
    headers = ["metric", "unwtd", "wtd"] + band_names
    rows = []
    for key, lab in [("homogeneous", "homogeneous"), ("split", "should split"),
                     ("distinct", "distinct"), ("fit_accurate", "label fits"), ("review", "needs review")]:
        row = [lab, _p(results["kpis"][key]), _p(results["kpis_weighted"][key])]
        row += [_p(results["by_band"][b].get(key)) for b in band_names]
        rows.append(row)
    L += ["SUMMARY — share of clusters (unwtd) vs share of texts (wtd), then by size band"]
    L += ["  " + ln for ln in _tbl(headers, rows, rj=set(range(1, len(headers))))]
    mass = ["  mass: " + " · ".join(f"{b}: {results['by_band'][b]['n_clusters']} cl / "
                                    f"{results['by_band'][b]['n_texts']:,} texts" for b in band_names)]
    L += mass + [""]
    # per-cluster table
    cls = [c for c in results["clusters"] if c["judged"]]
    flagged = [c for c in cls if c["review"]]
    show = sorted({c["cluster_id"]: c for c in (cls[:top_n] + flagged)}.values(), key=lambda c: -c["size"])
    rows = []
    for c in show:
        conf = c["confusable_with"][0] if c["confusable_with"] else None
        rows.append([c["cluster_id"], f"{c['size']:,}", c["band"],
                     _p(c["homogeneity"]["score"]), str(c["n_classes"]),
                     _p(c["distinctiveness"]["score"]) if c["distinctiveness"] else "   —",
                     (c["fit"]["verdict"] or "—"),
                     f"{conf['cluster_id']}:{conf['confusion']:.2f}" if conf else "—",
                     ",".join(c["review_reasons"]) or "—"])
    L += [f"PER-CLUSTER — top {top_n} by size plus every flagged cluster "
          f"({len(cls)} judged total; full detail in results.json)"]
    _pc_hdr = ["id", "size", "band", "homog", "cls", "distinct", "fit", "top-confusable", "review"]
    L += ["  " + ln for ln in _tbl(_pc_hdr, rows,
                                   rj=_rjust(_pc_hdr, {"size", "homog", "cls", "distinct"}))]
    L += [""]
    # flagged detail
    L += [f"FLAGGED FOR REVIEW — {len(flagged)} cluster(s)"]
    for c in flagged[:40]:
        L += [f"  {c['cluster_id']}  «{_clip(c['label'], 60)}»  size {c['size']:,}  — {', '.join(c['review_reasons'])}"]
        if c["split"]:
            comp = " · ".join(f"{k['name']} {k['frac'] * 100:.0f}%" for k in c["components"])
            L += [f"      sub-classes: {comp}" + (f" · minor {c['minor_share'] * 100:.0f}%" if c["minor_share"] >= 0.05 else "")]
        if "label" in c["review_reasons"] and c["fit"]["proposed_label"]:
            L += [f"      card rewrite: {c['fit']['proposed_label']} — {_clip(c['fit']['proposed_description'] or '', 80)}"]
        if c["confusable_with"]:
            L += ["      confusable with: " + " · ".join(f"{x['cluster_id']} ({x['confusion'] * 100:.0f}%)"
                                                         for x in c["confusable_with"][:3])]
    if len(flagged) > 40:
        L += [f"  … and {len(flagged) - 40} more (see results.json)"]
    L += [""]
    # merges + taxonomy
    L += [f"MERGE CANDIDATES — {len(results['merge_groups'])} group(s) "
          f"(item-level confusion ≥ {th['tau_conf']:.2f})"]
    for i, g in enumerate(results["merge_groups"][:20]):
        L += [f"  #{i}: " + " ".join(g[:12]) + (f" … +{len(g) - 12}" if len(g) > 12 else "")]
    if results.get("taxonomy"):
        L += ["", f"TAXONOMY — {len(results['taxonomy']['parents'])} parent groups (centroid hierarchy, judge-named)"]
        for p in results["taxonomy"]["parents"]:
            L += [f"  {_clip(p['name'], 40)}: " + " ".join(p["members"][:10])
                  + (f" … +{len(p['members']) - 10}" if len(p["members"]) > 10 else "")]
    L += ["", "NOTES",
          "  homogeneity = 1 - corrected mixing (share of within-cluster pairs the judge separates).",
          "  cls = sub-classes above the prevalence floor, from the co-grouping graph; split = cls >= 2.",
          "  distinct = corrected detection of near-neighbour intruders; low + high confusable = merge, not noise.",
          "  fit judges the card (label+description) against members; rewrites are judge proposals.",
          "  every estimate is sample-based and bias-corrected with this run's measured judge error rates.",
          bar]
    return "\n".join(L)


# ===========================================================================
# Public API
# ===========================================================================
def evaluate_clusters(data=None, embeddings=None, *, clusters=None, same_when=None, use=None,
                      unit=None, config: Optional[Config] = None, text_col="text",
                      cluster_col="cluster_id", label_col="label", description_col=None,
                      embedding_col=None, theme_col=None, workers=None, model=None,
                      coverage_target=None, progress=True) -> dict:
    """Evaluate a clustering. `data` = per-row table (DataFrame or path) with text + cluster_id
    (+ optional label/description columns); embeddings as an aligned array or `embedding_col`.
    (`clusters=` is accepted as a deprecated alias for `data`.) Returns the results dict."""
    if data is None:
        data = clusters
    cfg = config or Config()
    if same_when is not None:
        cfg.same_when = same_when
    if use is not None:
        cfg.use_context = use
    if unit is not None:
        cfg.unit = unit
    if workers is not None:
        cfg.workers = workers
    if model is not None:
        cfg.model = model
    if coverage_target is not None:
        cfg.coverage_target = coverage_target
    df, emb, labels = _coerce_inputs(data, embeddings, text_col=text_col, cluster_col=cluster_col,
                                     label_col=label_col, description_col=description_col,
                                     embedding_col=embedding_col, theme_col=theme_col)
    return evaluate(df, emb, labels, cfg, client=_make_client(cfg), progress=progress)


def generate_report(data=None, embeddings=None, out: Optional[str] = None, **kw) -> dict:
    """evaluate_clusters + optionally write results.json and report.txt to `out`."""
    results = evaluate_clusters(data, embeddings, **kw)
    if out:
        write_report(results, out)
    return results


def evaluate_from_files(data_path: str, embedding_path: Optional[str] = None, *,
                        embedding_col: Optional[str] = None, **kw) -> dict:
    emb = np.load(embedding_path) if embedding_path else None
    return evaluate_clusters(data_path, embeddings=emb, embedding_col=embedding_col, **kw)


def _render_calibration_html(results: dict) -> str:
    """Render a self-contained HTML review form for non-technical domain experts."""
    meta = results.get("meta", {})
    cal = results.get("calibration", {})
    calls = cal.get("calls", [])
    same_when = meta.get("same_when", "(not specified)")
    unit = meta.get("unit", "(not specified)")
    checks = cal.get("checks", {})
    overall = cal.get("overall_pass", None)
    clusters = results.get("clusters", []) or []
    label_of = {c.get("cluster_id"): c.get("label") or c.get("cluster_id") for c in clusters}

    by_type: Dict[str, List[dict]] = {}
    for r in calls:
        ct = r.get("ctrl_type", "?")
        by_type.setdefault(ct, []).append(r)
    for ct in by_type:
        by_type[ct].sort(key=lambda r: (1 if r.get("passed") else 0))

    gate_calls = [r for r in calls if r.get("ctrl_type") != "pure_cluster"]
    n_total = len(gate_calls)
    n_pass = sum(1 for r in gate_calls if r.get("passed"))

    def esc(s: str) -> str:
        return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                .replace('"', "&quot;").replace("'", "&#39;"))

    TYPE_ORDER = ["pure", "pure_cluster", "mixed", "junk", "farcal", "swap"]
    TYPE_TITLES = {
        "pure": "Same-kind grouping tests (cross-dataset — determines trust gate)",
        "pure_cluster": "Same-cluster grouping tests (informational — does NOT affect trust gate)",
        "mixed": "Cross-kind separation tests",
        "junk": "Obvious outlier tests",
        "farcal": "Far-away intruder tests",
        "swap": "Wrong label tests",
    }
    TYPE_DESC = {
        "pure": "We picked a random item from your data and showed the AI its nearest neighbours across the entire dataset. Because these items are genuinely similar by embedding distance — regardless of which cluster they belong to — a well-calibrated AI should keep them all in one group. This is the measure used to decide whether the trust gate passes.",
        "pure_cluster": "We showed the AI items drawn from within a single input cluster (nearest neighbours inside the cluster). If your clusters are themselves mixed, the AI may correctly detect sub-themes and split these — that is not a judge failure, it is a data quality signal. Low scores here mean your clusters contain multiple distinct kinds; they do NOT mean the AI is broken.",
        "mixed": "We showed the AI items from two genuinely different groups, mixed together. A good AI should separate them.",
        "junk": "We mixed in a randomly-constructed nonsense item with real items. A good AI should spot and isolate it.",
        "farcal": "We planted a single item from a completely different cluster. A good AI should identify it as the odd one out.",
        "swap": "We showed the AI items from one group but labelled them with the wrong category. A good AI should say the label doesn't fit.",
    }

    def card_html(r: dict, idx: int) -> str:
        ct = r.get("ctrl_type", "?")
        passed = r.get("passed", False)
        uid = r.get("uid", idx)
        border_color = "#c0392b" if not passed else "#27ae60"
        bg_color = "#fff5f5" if not passed else "#f0fff4"
        status_label = "AI FAILED" if not passed else "AI PASSED"
        status_color = "#c0392b" if not passed else "#27ae60"

        cid = r.get("cid", "")
        cid_label = label_of.get(cid, cid)
        cid_display = f"{esc(cid)} — {esc(cid_label)}" if cid and cid != cid_label else esc(cid)
        h = f'<div class="test-card" id="card-{uid}" style="border:2px solid {border_color};background:{bg_color};border-radius:8px;padding:20px;margin-bottom:20px;">'
        h += f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">'
        h += f'<span style="font-size:13px;color:#666;">Test #{uid} &middot; {esc(ct)}'
        if cid_display:
            h += f' &middot; group: {cid_display}'
        h += f'</span>'
        h += f'<span style="font-weight:bold;color:{status_color};font-size:14px;">{status_label}</span>'
        h += f'</div>'

        h += f'<p style="margin:0 0 8px 0;"><strong>Expected:</strong> {esc(r.get("expected", ""))}</p>'

        if ct == "pure":
            items = r.get("items", [])
            h += f'<div style="margin:10px 0;"><strong>Items shown:</strong><ol style="margin:6px 0 6px 20px;padding:0;">'
            for it in items:
                h += f'<li style="margin-bottom:4px;">{esc(it)}</li>'
            h += '</ol></div>'
            ng = r.get("n_groups_returned")
            if ng is not None:
                h += f'<p style="margin:4px 0;"><strong>AI returned:</strong> {esc(ng)} group(s)</p>'
            grps = r.get("groups_returned")
            if grps:
                h += '<div style="margin:8px 0;"><strong>Groups returned:</strong><ul style="margin:6px 0 6px 20px;padding:0;">'
                for gi, g in enumerate(grps):
                    h += f'<li>Group {gi+1}: ' + ", ".join(esc(x) for x in g) + '</li>'
                h += '</ul></div>'
            if passed:
                reviewer_q = "Do you agree these items are all the same kind?"
                opts = [
                    ("agree", "Yes — the AI correctly kept them together"),
                    ("disagree", "No — these items are actually different kinds"),
                    ("unsure", "I'm not sure"),
                ]
            else:
                reviewer_q = "The AI split these items into separate groups. Do you agree they are genuinely different kinds, or do you think they belong together?"
                opts = [
                    ("agree", "I agree with the AI — these items ARE different kinds"),
                    ("disagree", "I disagree — all these items are the same kind"),
                    ("unsure", "I'm not sure"),
                ]

        elif ct == "mixed":
            a_items = r.get("items_a", [])
            b_items = r.get("items_b", [])
            h += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:10px 0;">'
            h += '<div><strong>Group A items:</strong><ol style="margin:6px 0 6px 20px;padding:0;">'
            for it in a_items:
                h += f'<li style="margin-bottom:4px;">{esc(it["text"])}</li>'
            h += '</ol></div>'
            h += '<div><strong>Group B items:</strong><ol style="margin:6px 0 6px 20px;padding:0;">'
            for it in b_items:
                h += f'<li style="margin-bottom:4px;">{esc(it["text"])}</li>'
            h += '</ol></div></div>'
            ng = r.get("n_groups_returned")
            if ng is not None:
                h += f'<p style="margin:4px 0;"><strong>AI returned:</strong> {esc(ng)} group(s)</p>'
            reviewer_q = "Do you agree the AI correctly separated these two groups?"
            opts = [
                ("agree", "Yes — the AI correctly separated them"),
                ("disagree", "No — the AI should have kept them together"),
                ("unsure", "I'm not sure"),
            ]

        elif ct in ("junk", "farcal"):
            home_items = r.get("home_items", [])
            planted = r.get("planted_item", "")
            plant_src = r.get("plant_src", "")
            h += '<div style="margin:10px 0;"><strong>Regular items:</strong><ol style="margin:6px 0 6px 20px;padding:0;">'
            for it in home_items:
                h += f'<li style="margin-bottom:4px;">{esc(it)}</li>'
            h += '</ol></div>'
            if plant_src == "junk":
                src_display = "synthesized nonsense"
            else:
                src_label = label_of.get(plant_src, plant_src)
                src_display = f"{esc(plant_src)} — {esc(src_label)}" if src_label != plant_src else esc(plant_src)
            h += f'<div style="background:#fffbcc;border:1px solid #e6d600;border-radius:4px;padding:8px 12px;margin:8px 0;">'
            h += f'<strong>Planted item</strong> (from group: {src_display}): {esc(planted)}'
            h += '</div>'
            ng = r.get("n_groups_returned")
            if ng is not None:
                h += f'<p style="margin:4px 0;"><strong>AI returned:</strong> {esc(ng)} group(s)</p>'
            if passed:
                h += '<p style="margin:4px 0;color:#27ae60;"><strong>AI correctly isolated the planted item.</strong></p>'
            else:
                h += '<p style="margin:4px 0;color:#c0392b;"><strong>AI did NOT isolate the planted item.</strong></p>'
            reviewer_q = "Do you agree the highlighted item is the odd one out?"
            opts = [
                ("agree", "Yes — the highlighted item doesn't belong here"),
                ("disagree", "No — it fits in with the others"),
                ("unsure", "I'm not sure"),
            ]

        elif ct == "swap":
            label_shown = r.get("label_shown", "")
            desc_shown = r.get("description_shown", "")
            items = r.get("items", [])
            verdict = r.get("verdict_returned", "")
            proposed = r.get("proposed_label", "")
            h += f'<div style="background:#f0f4ff;border:1px solid #b0c0e0;border-radius:4px;padding:8px 12px;margin:8px 0;">'
            h += f'<strong>Label shown to AI:</strong> {esc(label_shown)}'
            if desc_shown:
                h += f'<br><em>{esc(desc_shown)}</em>'
            h += '</div>'
            h += '<div style="margin:10px 0;"><strong>Items (actually from a different cluster):</strong><ol style="margin:6px 0 6px 20px;padding:0;">'
            for it in items[:10]:
                h += f'<li style="margin-bottom:4px;">{esc(it)}</li>'
            if len(items) > 10:
                h += f'<li><em>…and {len(items)-10} more</em></li>'
            h += '</ol></div>'
            h += f'<p style="margin:4px 0;"><strong>AI verdict:</strong> {esc(verdict or "(none)")}'
            if proposed:
                h += f' &middot; proposed label: {esc(proposed)}'
            h += '</p>'
            reviewer_q = "Do you agree the label shown doesn't fit these items?"
            opts = [
                ("agree", "Yes — the label is wrong for these items"),
                ("disagree", "No — the label actually fits"),
                ("unsure", "I'm not sure"),
            ]

        else:
            reviewer_q = "What do you think of this AI decision?"
            opts = [
                ("agree", "I agree with the AI"),
                ("disagree", "I disagree with the AI"),
                ("unsure", "I'm not sure"),
            ]

        h += f'<div style="margin-top:14px;padding-top:12px;border-top:1px solid #ccc;">'
        h += f'<p style="font-weight:bold;margin:0 0 8px 0;">{esc(reviewer_q)}</p>'
        for val, label in opts:
            radio_id = f"r-{uid}-{val}"
            h += f'<label style="display:block;margin-bottom:6px;cursor:pointer;">'
            h += f'<input type="radio" name="vote-{uid}" id="{radio_id}" value="{val}" style="margin-right:6px;">'
            h += f'{esc(label)}</label>'
        h += f'<div style="margin-top:10px;">'
        h += f'<label for="notes-{uid}" style="display:block;margin-bottom:4px;font-size:13px;color:#555;">Notes (optional):</label>'
        h += f'<textarea id="notes-{uid}" rows="2" style="width:100%;box-sizing:border-box;font-size:13px;border:1px solid #ccc;border-radius:4px;padding:6px;resize:vertical;" placeholder="Any comments about this test case..."></textarea>'
        h += '</div></div>'
        h += '</div>'
        return h

    sections_html = ""
    for ct in TYPE_ORDER:
        recs = by_type.get(ct, [])
        if not recs:
            continue
        title = TYPE_TITLES.get(ct, ct)
        desc = TYPE_DESC.get(ct, "")
        n_s = len(recs)
        n_p = sum(1 for r in recs if r.get("passed"))
        sections_html += f'<details open class="cal-section">'
        sections_html += (f'<summary><h2 style="font-size:20px;">{esc(title)} '
                          f'<span class="sec-count">({n_p}/{n_s} passed)</span></h2></summary>')
        sections_html += f'<p style="color:#555;margin:10px 0 6px 0;">{esc(desc)}</p>'
        sections_html += f'<p style="color:#666;font-size:13px;margin:0 0 16px 0;">Showing {n_s} tests — {n_p} passed, {n_s - n_p} failed. Failing tests shown first.</p>'
        for idx, r in enumerate(recs):
            sections_html += card_html(r, idx)
        sections_html += '</details>'

    gate_color = "#27ae60" if overall else "#c0392b"
    gate_text = "OVERALL CALIBRATION: PASS" if overall else "OVERALL CALIBRATION: FAIL"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Calibration Review</title>
<style>
  body {{ font-family: Georgia, 'Times New Roman', serif; max-width: 900px; margin: 0 auto; padding: 20px 30px; color: #222; line-height: 1.6; }}
  h1 {{ font-size: 26px; border-bottom: 3px solid #222; padding-bottom: 10px; }}
  h2 {{ font-size: 20px; }}
  .intro {{ background: #f7f9fc; border-left: 4px solid #3498db; padding: 16px 20px; margin-bottom: 30px; border-radius: 0 6px 6px 0; }}
  .meta-box {{ background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 16px 20px; margin-bottom: 24px; }}
  .summary-box {{ background: #f0f7ff; border: 2px solid #3498db; border-radius: 6px; padding: 14px 20px; margin-bottom: 30px; font-size: 16px; }}
  .download-btn {{ display: inline-block; background: #2980b9; color: white; padding: 12px 28px; border: none; border-radius: 6px; font-size: 16px; cursor: pointer; margin-top: 20px; font-family: inherit; }}
  .download-btn:hover {{ background: #1a6a9a; }}
  .footer {{ margin-top: 50px; padding-top: 20px; border-top: 2px solid #333; text-align: center; color: #666; font-size: 14px; }}
  details.cal-section {{ margin-bottom: 32px; }}
  details.cal-section > summary {{ list-style: none; cursor: pointer; outline: none; border-bottom: 2px solid #333; padding-bottom: 6px; }}
  details.cal-section > summary::-webkit-details-marker {{ display: none; }}
  details.cal-section > summary::before {{ content: "\\25B8"; color: #999; font-size: 14px; margin-right: 8px; }}
  details.cal-section[open] > summary::before {{ content: "\\25BE"; }}
  details.cal-section > summary h2 {{ display: inline; border: none; }}
  details.cal-section > summary:hover h2 {{ color: #2980b9; }}
  details.cal-section .sec-count {{ color: #666; font-size: 13px; font-weight: normal; }}
</style>
</head>
<body>
<h1>Calibration Review Form</h1>

<div class="intro">
  <p><strong>What is this?</strong> Before running a full evaluation, our AI judge was tested on a set of carefully designed control cases where we already know the correct answer. This lets us check whether the AI is making sensible decisions.</p>
  <p><strong>Your role:</strong> You are a domain expert. We need you to look at each test case below and tell us whether <em>you</em> agree or disagree with what the AI did. Your feedback helps us decide whether to trust the AI's judgements on the real data.</p>
  <p><strong>How to use this form:</strong> For each test case, read the items shown, see what the AI decided, and select whether you agree or disagree. Add notes if you have additional thoughts. When done, click the "Download my review" button at the bottom.</p>
</div>

<div class="meta-box">
  <p style="margin:0 0 6px 0;"><strong>The grouping rule:</strong> Two items are the same kind when they {esc(same_when)}</p>
  <p style="margin:0;"><strong>What each item is:</strong> {esc(unit)}</p>
</div>

<div class="summary-box">
  <strong>{n_pass} of {n_total} tests passed</strong>
  &nbsp;&middot;&nbsp;
  <span style="color:{gate_color};font-weight:bold;">{gate_text}</span>
</div>

{sections_html}

<div class="footer">
  <button class="download-btn" onclick="downloadReview()">Download my review (JSON)</button>
  <p>Your annotations are only stored in your browser until you click the button above.</p>
</div>

<script>
function downloadReview() {{
  var cards = document.querySelectorAll('.test-card');
  var results = [];
  cards.forEach(function(card) {{
    var id = card.id.replace('card-', '');
    var voted = card.querySelector('input[name="vote-' + id + '"]:checked');
    var notes = card.querySelector('#notes-' + id);
    results.push({{
      test_id: id,
      ctrl_type: card.querySelector('span[style*="color:#666"]') ? card.querySelector('span[style*="color:#666"]').textContent.split('·')[1].trim() : '',
      ai_passed: card.querySelector('span[style*="font-weight:bold"]') ? card.querySelector('span[style*="font-weight:bold"]').textContent.trim() === 'AI PASSED' : null,
      reviewer_vote: voted ? voted.value : null,
      notes: notes ? notes.value.trim() : ''
    }});
  }});
  var blob = new Blob([JSON.stringify(results, null, 2)], {{type: 'application/json'}});
  var a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'calibration_review.json';
  a.click();
}}
</script>
</body>
</html>"""
    return html


def _render_results_html(results: dict) -> str:
    """Full results report as a self-contained HTML page for a non-technical audience.

    Built entirely from the results dict (the same content as results.json). Every metric is
    explained in plain English; sections move from "can we trust this?" through the headline
    numbers, the clusters that need attention, every cluster at a glance, possible duplicates,
    the topic hierarchy, and a glossary. No external dependencies.
    """
    import html as _h

    def esc(s):
        return _h.escape(str("" if s is None else s))

    def pct(x):
        return "—" if x is None else f"{x * 100:.0f}%"

    meta = results.get("meta", {})
    cal = results.get("calibration", {})
    kpis = results.get("kpis", {}) or {}
    kpisw = results.get("kpis_weighted", {}) or {}
    by_band = results.get("by_band", {}) or {}
    bands = results.get("bands", []) or []
    clusters = results.get("clusters", []) or []
    merge_groups = results.get("merge_groups", []) or []
    taxonomy = results.get("taxonomy")
    th = meta.get("thresholds", {}) or {}
    label_of = {c.get("cluster_id"): c.get("label") or c.get("cluster_id") for c in clusters}

    same_when = esc(meta.get("same_when", "(not specified)"))
    unit_str = esc(meta.get("unit", "(not specified)"))
    use_str = esc(meta.get("use", "")) if meta.get("use") else ""
    model = esc(meta.get("model", "?"))
    n_texts = meta.get("n_texts", 0)
    n_clusters = meta.get("n_clusters", 0)
    n_judged = meta.get("n_judged", 0)
    n_too_small = n_clusters - n_judged
    n_calls = meta.get("n_llm_calls", 0)

    overall = cal.get("overall_pass", None)
    health = cal.get("judge_health", {}) or {}
    checks = cal.get("checks", {}) or {}

    judged = [c for c in clusters if c.get("judged")]
    flagged = [c for c in judged if c.get("review")]

    # ------------------------------------------------------------------ CSS
    css = """
body{font-family:Georgia,'Times New Roman',serif;max-width:980px;margin:0 auto;color:#222;line-height:1.6;padding:0 20px 80px}
h1{font-size:1.7em;margin:40px 0 .2em}
h2{font-size:1.3em;border-bottom:2px solid #ddd;padding-bottom:.3em;margin-top:2.4em}
h3{font-size:1.08em;margin:1.4em 0 .4em}
.lead{color:#444;font-size:1.05em}
.rule-box{background:#f0f7ff;border-left:4px solid #4a9;padding:14px 18px;margin:20px 0;border-radius:0 8px 8px 0}
.rule-box p{margin:.4em 0}
.callout{border-radius:8px;padding:16px 20px;margin:18px 0}
.callout.good{background:#f1faf1;border:1px solid #8c8}
.callout.bad{background:#fdf1f1;border:1px solid #d99}
.callout.warn{background:#fffae8;border:1px solid #dca}
.callout.info{background:#f4f7fb;border:1px solid #bcd}
.callout h3{margin-top:0}
.statgrid{display:flex;flex-wrap:wrap;gap:14px;margin:18px 0}
.stat{flex:1 1 150px;border:1px solid #ddd;border-radius:8px;padding:14px 16px;background:#fafafa;text-align:center}
.stat .num{font-size:1.9em;font-weight:bold;display:block;line-height:1.1}
.stat .lab{font-size:.86em;color:#555;margin-top:.3em;display:block}
.stat.g .num{color:#2a7}.stat.r .num{color:#c33}.stat.n .num{color:#a60}
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:.93em}
th,td{border:1px solid #ddd;padding:7px 10px;text-align:left}
th{background:#f2f2f2}
td.r,th.r{text-align:right}
tr.flag{background:#fff6f6}
.scroll{max-height:560px;overflow:auto;border:1px solid #e2e2e2;border-radius:6px}
.scroll table{margin:0}
.scroll th{position:sticky;top:0}
.card{border:1px solid #ccc;border-radius:8px;padding:16px 20px;margin:14px 0;background:#fff}
.card.attn{border-left:5px solid #d44}
.card h3{margin-top:0}
.tag{display:inline-block;padding:2px 9px;border-radius:11px;font-size:.78em;font-weight:bold;margin:2px 4px 2px 0}
.tag.split{background:#fde2e2;color:#a22}
.tag.mixing_unclear{background:#fff0d6;color:#a60}
.tag.indistinct{background:#e7e0fb;color:#63c}
.tag.redundant{background:#dceefe;color:#06a}
.tag.label{background:#ffe6f0;color:#a25}
.tag.judge_uncertain{background:#eee;color:#555}
.tag.clean{background:#e2f5e2;color:#272}
.bar{height:14px;border-radius:7px;background:#eee;overflow:hidden;display:inline-block;width:140px;vertical-align:middle}
.bar > span{display:block;height:100%}
.muted{color:#777;font-size:.9em}
.kv{margin:.3em 0}
.kv b{display:inline-block;min-width:160px;color:#444}
details{margin:.5em 0}
summary{cursor:pointer;font-weight:bold;color:#358}
.glossary dt{font-weight:bold;margin-top:.7em}
.glossary dd{margin:.1em 0 .1em 1.2em;color:#444}
.toc{background:#fafafa;border:1px solid #e0e0e0;border-radius:8px;padding:14px 20px;margin:20px 0}
.toc a{display:block;margin:.25em 0;color:#358;text-decoration:none}
.toc a:hover{text-decoration:underline}
details.section{margin-top:1.2em}
details.section > summary{list-style:none;cursor:pointer;outline:none}
details.section > summary::-webkit-details-marker{display:none}
details.section > summary h2{display:inline-block;width:calc(100% - 1.4em)}
details.section > summary::before{content:"\\25B8";color:#999;font-size:.8em;margin-right:.5em;vertical-align:.15em}
details.section[open] > summary::before{content:"\\25BE"}
details.section > summary:hover h2{color:#358}
.method dt{font-weight:bold;margin-top:.9em;color:#235}
.method dd{margin:.2em 0 .2em 1.2em;color:#333}
.method .sci{color:#555;font-size:.93em;margin-top:.25em}
"""

    def bar(frac, color):
        f = 0 if frac is None else max(0, min(1, frac))
        return f'<span class="bar"><span style="width:{f*100:.0f}%;background:{color}"></span></span>'

    P = []
    P.append(f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Cluster Quality Report</title><style>{css}</style></head><body>""")

    # ---------------------------------------------------------------- header
    P.append("<h1>Cluster Quality Report</h1>")
    P.append("""<p class="lead">An AI judge examined your groups of items and assessed how
well-formed each group is — whether each group holds together as a single kind of thing,
whether any groups overlap or duplicate each other, and whether each group's name fits its
contents. This report explains what it found, in plain language.</p>""")

    P.append(f"""<div class="rule-box">
<p><strong>The rule used to decide if two items belong together:</strong><br>
Two items are the same kind when they <em>{same_when}</em></p>
<p><strong>What each item is:</strong> {unit_str}</p>
{('<p><strong>What the groups are used for:</strong> ' + use_str + '</p>') if use_str else ''}
<p class="muted">{n_texts:,} items · {n_clusters} groups ({n_judged} examined in detail,
{n_too_small} too small to examine) · {n_calls:,} AI judgements · judge model: {model}</p>
</div>""")

    # brief "how it works" up front
    P.append("""<div class="callout info"><h3>How the AI judged your groups (in brief)</h3>
<p>We never asked the AI &ldquo;is this group good?&rdquo; — a yes/no question an AI tends to
rubber-stamp. Instead we asked it to <strong>do tasks</strong> and watched what it did:</p>
<ul>
<li><strong>To measure cleanness:</strong> we repeatedly took a handful of items from a group,
shuffled them, and asked the AI to sort them into same-kind piles. If a group is truly one kind,
the AI keeps them in one pile; if it splits them, the group is mixed.</li>
<li><strong>To measure distinctness:</strong> we slipped an item from a neighbouring group into
the mix and checked whether the AI spotted the intruder. A group whose intruders go unnoticed
overlaps with its neighbours.</li>
<li><strong>To check the name:</strong> we showed the AI a group&rsquo;s items alongside its
name and asked whether the name fits, is too broad, or too narrow.</li>
</ul>
<p>Because the AI makes occasional mistakes, we also ran it on <strong>trick questions with known
answers</strong> (Section 1) to measure its error rate, then mathematically subtracted that error
from every score. The detailed, scientific version of all of this is in the glossary (Section 8).</p>
</div>""")

    # section helper: each <h2> becomes a collapsible (open by default) panel
    _sec_open = [False]

    def H2(anchor, title):
        pre = "</details>" if _sec_open[0] else ""
        _sec_open[0] = True
        return f'{pre}<details open class="section"><summary><h2 id="{anchor}">{esc(title)}</h2></summary>'

    # table of contents
    P.append("""<div class="toc"><strong>What's in this report</strong>
<a href="#trust">1. Can we trust these results?</a>
<a href="#headline">2. The headline numbers</a>
<a href="#bysize">3. Results by group size</a>
<a href="#attention">4. Groups that need your attention</a>
<a href="#all">5. Every group at a glance</a>
<a href="#dupes">6. Possible duplicate groups</a>
<a href="#hierarchy">7. Topic hierarchy</a>
<a href="#glossary">8. Glossary — how to read this</a>
</div>""")

    # ---------------------------------------------------- 1. trust / gate
    P.append(H2("trust", "1. Can we trust these results?"))
    P.append("""<p>Before judging your groups, we ran the AI through a set of trick questions
with known right answers — groups that are genuinely clean, groups deliberately mixed, obvious
odd-ones-out, and wrong labels. If the AI gets these right, we can trust its judgement on your
real data. This is the single most important thing to check first.</p>""")

    if overall is True:
        P.append("""<div class="callout good"><h3>✓ PASSED — results are trustworthy</h3>
<p>The AI reliably told planted-good from planted-bad on the trick questions, so the numbers
in the rest of this report can be relied upon.</p></div>""")
    elif overall is False:
        P.append("""<div class="callout bad"><h3>✗ FAILED — treat every number below with caution</h3>
<p>The AI did <strong>not</strong> reliably pass the trick questions. That means it is not
applying the grouping rule the way we need, so the scores in the rest of this report may be
wrong. The most common cause is that the grouping rule needs to be sharpened, or that the
groups themselves are already mixed. A companion file, <em>calibration_review.html</em>, shows
each trick question and lets a domain expert check whether they agree with the AI.</p></div>""")

    if health and not health.get("ok", True):
        P.append(f"""<div class="callout bad"><h3>⚠ The AI service looked unhealthy</h3>
<p>{health.get('n_empty',0):,} of {health.get('n_calls',0):,} requests
({pct(health.get('empty_rate'))}) came back empty or unreadable — above the
{pct(health.get('max_empty_rate'))} limit. This run was built on incomplete answers and should
not be relied upon; it is usually a temporary problem with the AI service.</p></div>""")
    elif health and health.get("n_empty"):
        P.append(f"""<p class="muted">AI service health: {health.get('n_empty',0):,} of
{health.get('n_calls',0):,} requests ({pct(health.get('empty_rate'))}) came back empty — within
the acceptable limit.</p>""")

    # the checks
    CHECK_LABELS = {
        "pure_kept_whole": ("Keeps genuinely similar items together ✓ gate",
                            "Shown items that are nearest neighbours across the whole dataset — truly similar regardless of cluster. Did the AI keep them in one group? This check determines whether the trust gate passes."),
        "cluster_purity": ("Keeps same-cluster items together (informational only)",
                           "Shown items drawn from within a single input cluster. Low scores here mean the input clusters themselves contain multiple sub-kinds — the AI may be correctly detecting those sub-themes. This does NOT affect the trust gate."),
        "mixture_split": ("Separates mixed groups ✓ gate",
                          "When shown two genuinely different kinds mixed together, did the AI split them apart?"),
        "junk_isolated": ("Spots nonsense items ✓ gate",
                          "When a nonsense item was slipped in, did the AI single it out?"),
        "far_intruder_found": ("Spots obvious intruders ✓ gate",
                               "When an item from a totally different group was planted, did the AI catch it?"),
        "swapped_label_flagged": ("Rejects wrong labels ✓ gate",
                                  "When shown a group with the wrong name attached, did the AI flag the mismatch?"),
        "correction_usable": ("Judge is reliable enough to correct ✓ gate",
                              "Is the AI accurate enough that we can mathematically correct for its small mistakes?"),
    }
    rows = []
    for k_, c in checks.items():
        lab, desc = CHECK_LABELS.get(k_, (k_.replace("_", " "), ""))
        sc = c.get("score")
        if k_ == "correction_usable":
            score_str = f"{sc:.2f}" if isinstance(sc, (int, float)) else esc(sc)
        else:
            score_str = pct(sc) if isinstance(sc, (int, float)) else esc(sc)
        is_gate = c.get("gate", True)
        if not is_gate:
            verdict = "<span style='color:#888;font-style:italic'>informational</span>"
        else:
            verdict = ("<span style='color:#2a7;font-weight:bold'>✓ pass</span>" if c.get("pass")
                       else "<span style='color:#c33;font-weight:bold'>✗ fail</span>")
        row_style = " style='background:#f8f8f8;'" if not is_gate else ""
        rows.append(f"<tr{row_style}><td>{esc(lab)}<br><span class='muted'>{esc(desc)}</span></td>"
                    f"<td class='r'>{score_str}</td><td>{esc(c.get('want',''))}</td>"
                    f"<td>{verdict}</td><td class='r'>{c.get('n','')}</td></tr>")
    if rows:
        P.append("<h3>The trick-question scorecard</h3>")
        P.append("<table><tr><th>What we tested</th><th class='r'>AI score</th>"
                 "<th>Needs to be</th><th>Result</th><th class='r'>Tests</th></tr>"
                 + "".join(rows) + "</table>")

    Sp, Se = cal.get("Sp"), cal.get("Se")
    if Sp is not None and Se is not None:
        P.append(f"""<p class="muted">Measured AI error: it wrongly separated same-kind items
{(1-Sp)*100:.1f}% of the time, and wrongly joined different-kind items {(1-Se)*100:.1f}% of the
time. All scores in this report are mathematically corrected for these measured error rates, so
they reflect the truth about your data rather than the AI's imperfections.</p>""")

    # ------------------------------------------------- 2. headline numbers
    P.append(H2("headline", "2. The headline numbers"))
    P.append("""<p>These figures are weighted by how many items each group contains, so a large
group counts for more than a tiny one — they tell you about the share of your
<em>items</em> that sit in good groups. Each measure is explained in the glossary at the end.</p>""")

    def stat(num, lab, cls=""):
        return f'<div class="stat {cls}"><span class="num">{num}</span><span class="lab">{lab}</span></div>'

    P.append('<div class="statgrid">')
    P.append(stat(pct(kpisw.get("homogeneous")), "of items are in a single clean kind", "g"))
    P.append(stat(pct(kpisw.get("split")), "of items are in groups that should be split", "r"))
    P.append(stat(pct(kpisw.get("distinct")), "of items are in clearly distinct groups", "g"))
    P.append(stat(pct(kpisw.get("fit_accurate")), "of items have an accurately named group", "g"))
    P.append(stat(pct(kpisw.get("review")), "of items need a human to review their group", "n"))
    P.append('</div>')

    P.append("""<div class="callout info"><p><strong>Two ways to read these.</strong> The big
numbers above are weighted by group size — a group with 1,000 items counts ten times as much as
one with 100. The table below also shows the unweighted figures, where every group counts equally
regardless of size. If the two columns differ a lot, your largest groups behave differently from
your smallest ones.</p></div>""")

    KPI_LABELS = [("homogeneous", "A single clean kind"), ("split", "Should be split up"),
                  ("distinct", "Clearly distinct from neighbours"),
                  ("fit_accurate", "Name accurately fits"), ("review", "Needs human review")]
    rows = []
    for key, lab in KPI_LABELS:
        rows.append(f"<tr><td>{lab}</td><td class='r'>{pct(kpisw.get(key))}</td>"
                    f"<td class='r'>{pct(kpis.get(key))}</td></tr>")
    P.append("<table><tr><th>Quality test</th><th class='r'>Share of items (weighted)</th>"
             "<th class='r'>Share of groups (unweighted)</th></tr>" + "".join(rows) + "</table>")

    # ------------------------------------------------- 3. by size band
    if bands:
        P.append(H2("bysize", "3. Results by group size"))
        P.append("""<p>Quality often depends on size. Here the same tests are broken out by how
large the groups are, so you can see whether (say) your biggest groups are the messy ones.</p>""")
        band_names = [b["band"] for b in bands]
        head = "<tr><th>Size band</th><th class='r'>Groups</th><th class='r'>Items</th>"
        head += "".join(f"<th class='r'>{esc(KPI_LABELS[i][1])}</th>" for i in range(len(KPI_LABELS)))
        head += "</tr>"
        rows = []
        for b in band_names:
            bb = by_band.get(b, {})
            cells = "".join(f"<td class='r'>{pct(bb.get(key))}</td>" for key, _ in KPI_LABELS)
            rows.append(f"<tr><td>{esc(b)}</td><td class='r'>{bb.get('n_clusters',0)}</td>"
                        f"<td class='r'>{bb.get('n_texts',0):,}</td>{cells}</tr>")
        P.append("<table>" + head + "".join(rows) + "</table>")
        P.append('<p class="muted">Size bands group your clusters by item count (smallest to '
                 'largest). "Groups" counts clusters in the band; "Items" sums their contents.</p>')

    # ------------------------------------------------- 4. needs attention
    P.append(H2("attention", "4. Groups that need your attention"))
    REASON_EXPLAIN = {
        "split": ("Should be split up",
                  "This group contains more than one distinct kind of item and would be clearer if broken into separate groups."),
        "mixing_unclear": ("Borderline — possibly mixed",
                           "We could not confidently tell whether this is one kind or several; it sits right on the borderline and is worth a human look."),
        "indistinct": ("Overlaps with neighbours",
                       "The AI had trouble telling this group's items apart from those in similar groups — its boundary is fuzzy."),
        "redundant": ("Possible duplicate",
                      "This group looks like a near-duplicate of one or more other groups and could potentially be merged."),
        "label": ("Name doesn't fit",
                  "The group's current name or description does not accurately describe what is actually in it."),
        "judge_uncertain": ("AI was inconsistent",
                            "When shown the same items twice, the AI gave different answers — treat this group's numbers with extra caution."),
    }
    if not flagged:
        P.append('<div class="callout good"><p>No groups were flagged for review — every '
                 'examined group passed all quality tests.</p></div>')
    else:
        P.append(f"""<p>{len(flagged)} of {n_judged} examined groups were flagged. Each card
below explains <em>why</em>, in plain language. The coloured tags summarise the issues; hover
intuition is given in the glossary.</p>""")
        for c in sorted(flagged, key=lambda x: -x.get("size", 0))[:80]:
            tags = "".join(f'<span class="tag {r}">{esc(REASON_EXPLAIN.get(r,(r,))[0])}</span>'
                           for r in c.get("review_reasons", []))
            P.append('<div class="card attn">')
            P.append(f'<h3>{esc(c.get("cluster_id"))} — {esc(c.get("label") or "(no name)")} '
                     f'<span class="muted">({c.get("size",0):,} items)</span></h3>')
            P.append(f'<p>{tags}</p>')
            # plain-English reasons
            P.append("<ul>")
            for r in c.get("review_reasons", []):
                _, desc = REASON_EXPLAIN.get(r, (r, ""))
                P.append(f"<li>{esc(desc)}</li>")
            P.append("</ul>")
            # split sub-classes
            if c.get("split") and c.get("components"):
                comp = " · ".join(f"{esc(k.get('name'))} ({k.get('frac',0)*100:.0f}%)"
                                  for k in c["components"])
                extra = (f" Plus {c.get('minor_share',0)*100:.0f}% scattered across smaller pieces."
                         if c.get("minor_share", 0) >= 0.05 else "")
                P.append(f'<p class="kv"><b>Sub-kinds found:</b> {comp}.{esc(extra)}</p>')
            # label rewrite proposal
            fit = c.get("fit", {}) or {}
            if "label" in c.get("review_reasons", []) and fit.get("proposed_label"):
                P.append(f'<p class="kv"><b>AI&rsquo;s suggested name:</b> '
                         f'&ldquo;{esc(fit.get("proposed_label"))}&rdquo;'
                         + (f' — {esc(fit.get("proposed_description"))}'
                            if fit.get("proposed_description") else "") + '</p>')
            # confusable with
            if c.get("confusable_with"):
                conf = " · ".join(
                    f"{esc(x.get('cluster_id'))} — {esc(label_of.get(x.get('cluster_id'), x.get('cluster_id')))} ({x.get('confusion',0)*100:.0f}%)"
                    for x in c["confusable_with"][:3])
                P.append(f'<p class="kv"><b>Most easily confused with:</b> {conf}</p>')
            P.append('</div>')
        if len(flagged) > 80:
            P.append(f'<p class="muted">… and {len(flagged)-80} more flagged groups '
                     f'(full detail in results.json).</p>')

    # ------------------------------------------------- 5. every cluster
    P.append(H2("all", "5. Every group at a glance"))
    P.append("""<p>A compact scorecard for every examined group, largest first. Green bars are
good (clean / distinct); the verdict column gives the one-line summary.</p>""")
    show = sorted(judged, key=lambda x: -x.get("size", 0))[:400]
    rows = []
    for c in show:
        homog = (c.get("homogeneity") or {}).get("score")
        dist = (c.get("distinctiveness") or {}).get("score") if c.get("distinctiveness") else None
        fitv = (c.get("fit") or {}).get("verdict") or "—"
        if c.get("review"):
            verdict = "<span style='color:#c33'>needs review</span>"
        else:
            verdict = "<span style='color:#2a7'>clean</span>"
        flagcls = " class='flag'" if c.get("review") else ""
        homog_cell = f"{bar(homog,'#3a8')} {pct(homog)}" if homog is not None else "—"
        dist_cell = f"{bar(dist,'#39c')} {pct(dist)}" if dist is not None else "—"
        rows.append(f"<tr{flagcls}><td>{esc(c.get('cluster_id'))}</td>"
                    f"<td>{esc(c.get('label') or '')}</td>"
                    f"<td class='r'>{c.get('size',0):,}</td><td>{esc(c.get('band'))}</td>"
                    f"<td>{homog_cell}</td><td class='r'>{c.get('n_classes','—')}</td>"
                    f"<td>{dist_cell}</td><td>{esc(fitv)}</td><td>{verdict}</td></tr>")
    P.append('<div class="scroll"><table>'
             "<tr><th>ID</th><th>Name</th><th class='r'>Items</th><th>Size band</th>"
             "<th>Cleanness</th><th class='r'>Sub-kinds</th><th>Distinctness</th>"
             "<th>Name fit</th><th>Verdict</th></tr>" + "".join(rows) + "</table></div>")
    note = f"Showing the largest {len(show)} of {n_judged} examined groups." if n_judged > len(show) \
        else f"All {n_judged} examined groups shown."
    if n_too_small:
        note += f" A further {n_too_small} groups were too small to examine."
    P.append(f'<p class="muted">{note} Complete machine-readable detail is in results.json.</p>')

    # ------------------------------------------------- 6. duplicates
    P.append(H2("dupes", "6. Possible duplicate groups"))
    if not merge_groups:
        P.append('<div class="callout good"><p>No groups looked like duplicates of each other — '
                 'each group appears to cover its own distinct territory.</p></div>')
    else:
        P.append(f"""<p>The AI found {len(merge_groups)} set(s) of groups whose items it
repeatedly confused with each other. These are candidates to merge into one group — but a human
should confirm, since sometimes two genuinely different groups simply sit close together.</p>""")
        for i, g in enumerate(merge_groups[:30]):
            members = " · ".join(f"{esc(x)} — {esc(label_of.get(x, x))}" for x in g[:15])
            more = f" … +{len(g)-15} more" if len(g) > 15 else ""
            P.append(f'<p class="kv"><b>Set {i+1}:</b> {members}{more}</p>')
        if len(merge_groups) > 30:
            P.append(f'<p class="muted">… and {len(merge_groups)-30} more sets (see results.json).</p>')

    # ------------------------------------------------- 7. taxonomy
    P.append(H2("hierarchy", "7. Topic hierarchy"))
    parents = (taxonomy or {}).get("parents") if taxonomy else None
    if not parents:
        P.append('<p class="muted">No topic hierarchy was generated for this run.</p>')
    else:
        P.append("""<p>The AI grouped your groups into broader themes, giving a bird&rsquo;s-eye
map of your data. Each heading is a broad theme; the entries beneath are the groups within it.</p>""")
        for p in parents:
            members = " · ".join(f"{esc(x)} — {esc(label_of.get(x, x))}" for x in p.get("members", [])[:15])
            more = f" … +{len(p.get('members',[]))-15} more" if len(p.get("members", [])) > 15 else ""
            P.append(f'<details><summary>{esc(p.get("name"))} '
                     f'<span class="muted">({len(p.get("members",[]))} groups)</span></summary>'
                     f'<p>{members}{more}</p></details>')

    # ------------------------------------------------- 8. glossary
    P.append(H2("glossary", "8. Glossary — how to read this"))
    P.append("""<p>Each entry gives a one-line plain-English meaning followed by the detailed,
scientific explanation of how the number is actually produced. Expand &ldquo;The method&rdquo;
under any term for the full picture.</p>""")

    GLOSSARY = [
        ("A single clean kind (cleanness / homogeneity)",
         "The group holds together as one kind of thing. 100% means every item belongs; lower means "
         "the group mixes several kinds.",
         "Cleanness = 1 − corrected mixing rate. <b>Mixing</b> is estimated with the PARTITION task: "
         "we repeatedly draw a small sample (k items) from the group, shuffle them, and ask the judge "
         "to sort them into same-kind sub-groups under your rule. Every pair of items that the judge "
         "places in <i>different</i> sub-groups is evidence of mixing; the raw mixing rate is the share "
         "of within-group pairs that the judge separates, pooled over many draws. We then apply the "
         "<b>Rogan-Gladen correction</b> using the judge&rsquo;s measured error: "
         "corrected = (raw − (1 − Sp)) / (Se + Sp − 1), clamped to [0,1], where Sp (specificity) and "
         "Se (sensitivity) come from the trick questions. A 95% Wilson confidence interval is carried "
         "through the whole calculation, so the score has an upper and lower bound, not just a point."),
        ("Should be split up",
         "The group confidently contains more than one distinct kind and would be clearer broken apart.",
         "The split verdict keys off the <i>lower bound</i> of the corrected mixing interval, not a raw "
         "count: a group is flagged <b>split</b> only when we are confident (lower Wilson bound) that "
         "true mixing exceeds the tolerance τ (default 0.15); it is declared <b>one clean kind</b> when "
         "the <i>upper</i> bound is below τ; in between it is &ldquo;borderline&rdquo;. We use the "
         "low-variance pair statistic rather than the sub-group count because, at modest sampling, even "
         "a pure group fragments into several pieces by chance."),
        ("Sub-kinds",
         "When a group should be split, how many distinct kinds were found inside it (and roughly what "
         "share each makes up).",
         "Across all PARTITION draws we build a <b>co-grouping graph</b>: nodes are items, edge weight is "
         "how often two items were placed together. Connected components of this graph (found by "
         "likelihood-ratio-gated agglomeration, with draw-replication gating to defeat single-draw false "
         "joins) are the candidate sub-kinds. Because reconstruction is high-variance at low coverage, we "
         "report the honest, decision-useful resolution: the largest component as the named major kind, "
         "everything else pooled as one &ldquo;mixed / other&rdquo; remainder, each shown with a Wilson "
         "interval on its share."),
        ("Clearly distinct (distinctness)",
         "How well-separated the group is from its nearest neighbours. High = easy to tell apart from "
         "similar groups; low = fuzzy boundary that overlaps others.",
         "Measured by the planted-intruder task: into a PARTITION draw from the group we insert one item "
         "from a near-neighbour group (chosen by embedding distance) and check whether the judge isolates "
         "it. Distinctness = corrected detection rate of these near intruders, again Rogan-Gladen-corrected "
         "for chance isolation γ: corrected = (raw − γ)/(1 − γ). Per-neighbour <b>confusion</b> = 1 − "
         "detection against that specific neighbour; a low distinctness combined with a high confusion "
         "against one neighbour points to a merge, whereas diffuse low detection points to genuine noise."),
        ("Name accurately fits (name fit)",
         "Whether the group&rsquo;s name and description match its actual contents.",
         "The FIT task shows the judge a sample of members alongside the group&rsquo;s label card and asks "
         "for a verdict — <b>accurate</b> (right breadth), <b>too_broad</b> (label vaguer/wider than the "
         "members), or <b>too_narrow</b> (members include kinds the label omits). When the first verdict "
         "is not &ldquo;accurate&rdquo; we take additional replicates and use the majority. The judge also "
         "proposes a replacement label and description, shown in Section 4. FIT is calibrated separately by "
         "the wrong-label trick questions and never influences the cleanness or split scores."),
        ("Possible duplicate (redundant)",
         "The group is repeatedly confused with other groups and might be the same thing split across "
         "several groups.",
         "Built from item-level cross-group confusion measured during the intruder tests. Two groups are "
         "linked when the judge co-groups their items at a rate whose conservative (lower-bound) estimate "
         "exceeds the merge threshold τ_conf, derived from this run&rsquo;s own statistics. Linked groups "
         "are assembled into merge sets. Because the bound is conservative, a reported merge is a strong "
         "signal — but two genuinely adjacent kinds can still sit close, so a human confirms."),
        ("Needs human review",
         "At least one quality test flagged this group — see Section 4 for the specific reason.",
         "A group is flagged if any of these hold: it is <b>split</b> or borderline (mixing unclear); its "
         "distinctness is below the detection bar (<b>indistinct</b>); it belongs to a merge set "
         "(<b>redundant</b>); its name verdict is not accurate (<b>label</b>); or the judge gave "
         "inconsistent answers when shown identical duplicated draws (<b>judge_uncertain</b>, from the "
         "silent-duplicate agreement rate)."),
        ("Share of groups vs share of items",
         "&ldquo;Share of groups&rdquo; counts every group equally; &ldquo;share of items&rdquo; weights "
         "by group size.",
         "The unweighted KPI is a simple mean over judged groups; the weighted KPI weights each group by "
         "its item count. Comparing the two reveals whether quality is concentrated in your large or small "
         "groups — e.g. a high unweighted but low weighted cleanness means a few big groups are messy."),
        ("Corrected for AI error (Rogan-Gladen)",
         "Every score is adjusted using the AI&rsquo;s measured mistake rates from the trick questions, so "
         "it reflects the truth about your data, not the AI&rsquo;s imperfections.",
         "The planted controls yield the judge&rsquo;s pair-level <b>specificity</b> Sp (rate of correctly "
         "keeping same-kind pairs together) and <b>sensitivity</b> Se (rate of correctly separating "
         "different-kind pairs). The Rogan-Gladen estimator inverts the measurement: it maps an observed "
         "rate back to the true prevalence given Sp and Se. The correction is only applied when the judge "
         "is informative enough (Se + Sp − 1 ≥ 0.2); otherwise raw rates pass through and the gate fails, "
         "warning you the numbers are unreliable."),
        ("The trick questions (calibration controls)",
         "Known-answer tasks we run before judging your real data, to measure the AI&rsquo;s error rate.",
         "Five families are planted, all drawn from your own data: <b>pure</b> (a tight same-group sample, "
         "should stay one pile → estimates Sp), <b>mixed</b> (half from each of two groups, should split → "
         "estimates Se), <b>junk</b> (a nonsense item, should be isolated), <b>far</b> (an obvious intruder "
         "from a distant group, should be isolated), and <b>label-swap</b> (a group shown with the wrong "
         "name, FIT should reject it). The five pass/fail checks plus the correction-usable check form the "
         "calibration gate; every check must pass for the run to be trustworthy."),
    ]
    P.append('<dl class="method">')
    for term, plain, sci in GLOSSARY:
        P.append(f"<dt>{esc(term)}</dt><dd>{plain}"
                 f"<details><summary>The method</summary>"
                 f"<div class='sci'>{sci}</div></details></dd>")
    P.append("</dl>")

    P.append("</details>")  # close the final collapsible section
    P.append("<hr><p class='muted'>Generated by cluster_judge. Trustworthiness of every number "
             "above depends on Section 1 passing. The companion file calibration_review.html lets a "
             "domain expert audit the AI&rsquo;s trick-question answers directly.</p>")
    P.append("</body></html>")
    return "\n".join(P)


def write_report(results: dict, outdir: str = "cj_out") -> str:
    import os
    os.makedirs(outdir, exist_ok=True)
    written = ["results.json", "report.txt"]
    with open(os.path.join(outdir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    with open(os.path.join(outdir, "report.txt"), "w") as f:
        f.write(render_report(results) + "\n")
    with open(os.path.join(outdir, "results_review.html"), "w", encoding="utf-8") as f:
        f.write(_render_results_html(results))
    written.append("results_review.html")
    if results.get("calibration", {}).get("calls"):
        with open(os.path.join(outdir, "calibration_review.html"), "w", encoding="utf-8") as f:
            f.write(_render_calibration_html(results))
        written.append("calibration_review.html")
    log.info("wrote %s to %s/", ", ".join(written), outdir)
    return outdir


def make_demo_data(seed: int = 7, dim: int = 64):
    """Synthetic skewed dataset: ~103k texts, 227 clusters incl. a 20k 70/30-mixed giant.

    Note: the demo *text* deliberately embeds the cluster id and theme ("C_GIANT :: theme0
    item 5"). This is only safe because the demo runs the offline mock, which scores off the
    `_theme` ground-truth column and never reads the text. Do NOT point a real gateway judge
    at this data — it would read the leaked labels and report flatteringly perfect numbers."""
    rng = np.random.default_rng(seed)
    n_themes = 24
    centers = rng.normal(0, 1, (n_themes, dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    rows = []

    def emit(cid, mix, n):
        ths = rng.choice([t for t, _ in mix], size=n, p=[f for _, f in mix])
        for j in range(n):
            rows.append((f"{cid} :: theme{ths[j]} item {j}", cid, int(ths[j])))

    plan = [("C_GIANT", [(0, 0.70), (1, 0.30)], 20000), ("C_BIG", [(2, 1.0)], 10000)]
    plan += [(f"L{i}", [(3 + i, 1.0)], 5000) for i in range(5)]
    plan += [(f"M{i}", [(8 + (i % 12), 1.0)], 1000) for i in range(13)]
    plan += [(f"S{i}", [(int(rng.integers(0, n_themes)), 1.0)], int(rng.integers(40, 320)))
             for i in range(227 - len(plan) - 13)]
    plan += [(f"X{i}", [(int(rng.integers(0, n_themes)), 1.0)], 1) for i in range(13)]
    for cid, mix, n in plan:
        emit(cid, mix, n)
    df = pd.DataFrame(rows, columns=["text", "cluster_id", "_theme"])
    emb = (centers[df["_theme"].to_numpy()] + rng.normal(0, 0.05, (len(df), dim))).astype(np.float32)
    labels = {}
    for cid, g in df.groupby("cluster_id"):
        dom = int(g["_theme"].value_counts().idxmax())
        labels[str(cid)] = {"label": f"theme {dom}", "_theme": dom,
                            "description": f"Customer objections of theme {dom} raised on outbound calls."}
    return df, emb, labels


def _demo_config(**overrides) -> Config:
    """The objection-domain Config used by the offline `--demo` flows (noisy mock)."""
    base = dict(unit="each text is a customer objection raised on an outbound sales call",
                same_when="are the same kind of objection (the underlying concern), regardless of how it is answered",
                use_context="each cluster is mined for the range of response strategies to that objection type",
                model="mock", mock_eps_split=0.08, mock_eps_join=0.05)
    base.update(overrides)
    return Config(**base)


# ===========================================================================
# CLI
# ===========================================================================
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="cluster_judge v3 — LLM-as-judge clustering evaluator (text report)")
    ap.add_argument("--demo", action="store_true", help="run the synthetic skewed dataset offline (noisy mock)")
    ap.add_argument("--data", "--clusters", dest="data", help="per-row table (csv/tsv/parquet/jsonl)")
    ap.add_argument("--embeddings", help=".npy embeddings aligned to rows")
    ap.add_argument("--embedding-col", help="column holding per-row vectors")
    ap.add_argument("--same-when", help="equivalence rule: two items are the same kind when they ...")
    ap.add_argument("--use", help="what each cluster is used for")
    ap.add_argument("--unit", help="what one text is")
    ap.add_argument("--out", default="cj_out")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--coverage", type=float, default=0.20)
    ap.add_argument("--model", default=None, help="judge id; 'mock' forces offline")
    ap.add_argument("--recluster", metavar="CLUSTER_ID",
                    help="instead of evaluating, deeply re-cluster ONE flagged cluster into named sub-clusters")
    a = ap.parse_args(argv)
    configure_logging()
    import os
    if a.recluster:
        if a.demo:
            df, emb, labels = make_demo_data()
            rc = recluster_cluster(a.recluster, data=df, embeddings=emb,
                                   config=_demo_config(workers=a.workers))
        else:
            if not a.data:
                ap.error("provide --data (and --embeddings or --embedding-col), or use --demo")
            rc = recluster_cluster(a.recluster, data=a.data,
                                   embeddings=(np.load(a.embeddings) if a.embeddings else None),
                                   embedding_col=a.embedding_col, same_when=a.same_when,
                                   use=a.use, unit=a.unit, config=Config(workers=a.workers,
                                                                         model=(a.model or "gateway")))
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, f"recluster_{a.recluster}.json"), "w") as f:
            json.dump(rc, f, indent=2, default=str)
        with open(os.path.join(a.out, f"recluster_{a.recluster}.txt"), "w") as f:
            f.write(render_recluster_report(rc) + "\n")
        print(render_recluster_report(rc))
        return 0
    if a.demo:
        df, emb, labels = make_demo_data()
        R = evaluate(df, emb, labels, _demo_config(coverage_target=a.coverage, workers=a.workers))
    else:
        if not a.data:
            ap.error("provide --data (and --embeddings or --embedding-col), or use --demo")
        R = evaluate_clusters(a.data, embeddings=(np.load(a.embeddings) if a.embeddings else None),
                              embedding_col=a.embedding_col, same_when=a.same_when, use=a.use,
                              unit=a.unit, workers=a.workers, coverage_target=a.coverage,
                              model=(a.model or "gateway"))
    write_report(R, a.out)
    print(render_report(R))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
