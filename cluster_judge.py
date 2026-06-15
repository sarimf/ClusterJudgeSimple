"""cluster_judge — weighted distinctiveness evaluator for text clusterings.

Plants one item from a neighbouring cluster among k−1 items from the
target cluster and asks the LLM to sort them by kind. A cluster is
*distinct* when the LLM reliably isolates the intruder. The headline
number is the size-weighted fraction of clusters that pass this test,
corrected for judge error via a calibration pre-pass.

Output: results dict  +  HTML report  (no dashboard, no build step).
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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

log = logging.getLogger("cluster_judge")

__version__ = "1.0.0"
__all__ = ["Config", "use_genai", "evaluate", "write_html", "render_html",
           "make_demo_data", "main"]


# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    # Required: the equivalence rule that defines "same kind"
    same_when: str = ""
    unit: str = "each text is a short customer message"
    # Sampling
    coverage_target: float = 0.20
    coverage_ceiling: float = 0.40
    min_pool: int = 6
    item_chars: int = 1024
    # PARTITION / intruder design
    k_partition: int = 10
    neighbor_m: int = 3          # near-neighbour clusters used as intruder sources
    intruder_per_wave: int = 4   # intruder draws per cluster per wave
    intruder_waves_max: int = 3
    min_judgeable: int = 5       # clusters smaller than this are skipped
    # Calibration controls
    n_pure: int = 24
    n_mixed: int = 24
    n_junk: int = 12
    n_far: int = 24
    # Budget / concurrency
    max_llm_calls: int = 20000
    workers: int = 64
    max_retries: int = 4
    backoff_base: float = 0.5
    max_empty_rate: float = 0.25
    z: float = 1.96
    # Judge
    model: str = "mock"
    seed: int = 7
    # Mock-judge noise (to exercise the correction layer)
    mock_eps_split: float = 0.0
    mock_eps_join: float = 0.0

    def __post_init__(self):
        if not self.same_when:
            raise ValueError("Config.same_when is required")
        if not 0 < self.coverage_target <= 1:
            raise ValueError("coverage_target must be in (0, 1]")
        if not 0 < self.coverage_ceiling <= 1:
            raise ValueError("coverage_ceiling must be in (0, 1]")
        if self.coverage_target > self.coverage_ceiling:
            raise ValueError("coverage_target must be <= coverage_ceiling")
        if self.k_partition < 3:
            raise ValueError("k_partition must be >= 3")


# ══════════════════════════════════════════════════════════════════════════════
# Gateway wiring
# ══════════════════════════════════════════════════════════════════════════════

_GATEWAY: Dict[str, Optional[Callable]] = {"fn": None}


def use_genai(fn: Callable) -> Callable:
    """Register gateway: fn(messages: list[dict], json_mode=True) -> str."""
    _GATEWAY["fn"] = fn
    return fn


def _discover_gateway() -> Optional[Callable]:
    if _GATEWAY["fn"]:
        return _GATEWAY["fn"]
    import builtins, sys
    for ns in (getattr(sys.modules.get("__main__"), "__dict__", {}), vars(builtins)):
        fn = ns.get("run_model_messages")
        if callable(fn):
            return fn
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Unit / LLM client
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Unit:
    uid: int
    cid: str
    payload: dict
    truth: dict = field(default_factory=dict)


def _clip(t: Any, n: int) -> str:
    s = str(t).replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _numbered(items: Sequence[str]) -> str:
    return "\n".join(f"{i + 1}. {t}" for i, t in enumerate(items))


def _partition_prompt(u: Unit, cfg: Config) -> str:
    head = (f"# unit: {cfg.unit}\n"
            f"# rule: two items are the SAME KIND when they {cfg.same_when}\n")
    return (head +
            "Sort the items into groups of the same kind under the rule. "
            "Singletons are allowed. Group by the rule, not by wording or shared topic words. "
            "Use every index exactly once.\n"
            f"ITEMS:\n{_numbered(u.payload['items'])}\n"
            'Return STRICT JSON: {"groups": [[1,4],[2],[3,5]]}')


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
                pass
    return None


class LLMClient:
    def __init__(self, cfg: Config, fn: Optional[Callable] = None):
        self.cfg = cfg
        self.fn = fn
        self.mock = fn is None
        self.n_calls = 0
        self.n_empty = 0
        self._lock = threading.Lock()
        self._rng = random.Random(cfg.seed * 7919 + 13)

    def complete(self, u: Unit) -> dict:
        with self._lock:
            self.n_calls += 1
        if self.mock:
            with self._lock:
                return _mock(u, self.cfg, self._rng)
        msgs = [{"role": "system", "content": "You are a careful evaluator. Reply with STRICT JSON only."},
                {"role": "user", "content": _partition_prompt(u, self.cfg)}]
        delay = self.cfg.backoff_base
        for att in range(self.cfg.max_retries + 1):
            try:
                v = _parse_json(self.fn(msgs, json_mode=True))
                if v:
                    return v
            except Exception as e:
                if att == self.cfg.max_retries:
                    log.warning("unit %d failed after retries: %s", u.uid, e)
            time.sleep(delay)
            delay *= 2
        with self._lock:
            self.n_empty += 1
        return {}


def _run_units(units: List[Unit], client: LLMClient, cfg: Config,
               desc: str = "judging", progress: bool = True,
               executor: Optional[ThreadPoolExecutor] = None) -> Dict[int, dict]:
    out: Dict[int, dict] = {}
    if not units:
        return out
    bar = tqdm(total=len(units), desc=desc, unit="call") if (progress and tqdm) else None
    own = executor is None
    ex = executor or ThreadPoolExecutor(max_workers=max(1, min(cfg.workers, len(units))))
    try:
        futs = {ex.submit(lambda u=u: (u.uid, client.complete(u))): None for u in units}
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


# ══════════════════════════════════════════════════════════════════════════════
# Mock judge (for offline testing / demo)
# ══════════════════════════════════════════════════════════════════════════════

def _mock(u: Unit, cfg: Config, rng: random.Random) -> dict:
    themes = u.truth.get("themes")
    k = len(u.payload["items"])
    if themes is None:
        return {"groups": [[i + 1] for i in range(k)]}
    groups: Dict[str, List[int]] = {}
    for i, th in enumerate(themes, start=1):
        groups.setdefault(str(th), []).append(i)
    gs = list(groups.values())
    if cfg.mock_eps_split > 0:
        out: List[List[int]] = []
        for g in gs:
            if len(g) >= 2 and rng.random() < cfg.mock_eps_split:
                cut = rng.randrange(1, len(g))
                out += [g[:cut], g[cut:]]
            else:
                out.append(g)
        gs = out
    if cfg.mock_eps_join > 0 and len(gs) >= 2 and rng.random() < cfg.mock_eps_join:
        i, j = rng.sample(range(len(gs)), 2)
        merged = gs[i] + gs[j]
        gs = [g for ki, g in enumerate(gs) if ki not in (i, j)] + [merged]
    return {"groups": gs}


# ══════════════════════════════════════════════════════════════════════════════
# Data loading + prep
# ══════════════════════════════════════════════════════════════════════════════

def _load_df(path: str) -> pd.DataFrame:
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


def _coerce(data, embeddings=None, *, text_col="text", cluster_col="cluster_id",
            label_col="label", embedding_col=None):
    df = data if isinstance(data, pd.DataFrame) else _load_df(data)
    ren = {}
    if text_col != "text" and text_col in df.columns:
        ren[text_col] = "text"
    if cluster_col != "cluster_id" and cluster_col in df.columns:
        ren[cluster_col] = "cluster_id"
    df = df.rename(columns=ren).copy()
    if "text" not in df.columns or "cluster_id" not in df.columns:
        raise ValueError("need 'text' and 'cluster_id' columns (map via *_col args)")
    labels: Dict[str, str] = {}
    for cid, g in df.groupby("cluster_id"):
        cid = str(cid)
        labels[cid] = str(g[label_col].iloc[0]) if label_col in df.columns else cid
    if embeddings is not None:
        emb = np.asarray(embeddings, dtype=np.float32)
    elif embedding_col and embedding_col in df.columns:
        emb = np.vstack(df[embedding_col].to_list()).astype(np.float32)
    else:
        raise ValueError("embeddings required: pass embeddings=<array> or embedding_col=<col>")
    if len(emb) != len(df):
        raise ValueError(f"embeddings length {len(emb)} != rows {len(df)}")
    return df, emb, labels


def _normalize(emb: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(emb, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (emb / n).astype(np.float32)


def _pool(emb_n: np.ndarray, idxs: np.ndarray, cfg: Config,
          rng: np.random.Generator) -> np.ndarray:
    n = len(idxs)
    target = int(min(
        max(cfg.min_pool, round(n * cfg.coverage_target)),
        round(n * cfg.coverage_ceiling),
        n,
    ))
    if n <= target:
        return np.asarray(idxs)
    return rng.choice(np.asarray(idxs), size=target, replace=False)


# ══════════════════════════════════════════════════════════════════════════════
# Math helpers
# ══════════════════════════════════════════════════════════════════════════════

def _wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def _rg_det(d: float, gamma: float) -> float:
    """Rogan-Gladen: correct raw detection rate for chance isolation (gamma)."""
    return min(1.0, max(0.0, (d - gamma) / (1 - gamma))) if gamma < 1 else d


def _groups_valid(v: dict, k: int) -> Optional[List[List[int]]]:
    try:
        gs = [[int(x) for x in g] for g in v.get("groups", [])]
    except Exception:
        return None
    flat = sorted(x for g in gs for x in g)
    return gs if flat == list(range(1, k + 1)) else None


def _co_pairs(gs: List[List[int]]) -> set:
    return {(min(a, b), max(a, b)) for g in gs for a in g for b in g if a != b}


# ══════════════════════════════════════════════════════════════════════════════
# Calibration
# ══════════════════════════════════════════════════════════════════════════════

def _calibrate(units: List[Unit], res: Dict[int, dict]) -> dict:
    """Pure/mixed/junk/far planted controls → Sp, Se, gamma, gate checks."""
    same_co = [0, 0]   # co, tot on truly-same pairs  → Sp
    diff_sp = [0, 0]   # split, tot on truly-diff pairs → Se
    iso = [0, 0]       # singletons in pure controls   → gamma
    pure_one = [0, 0]
    mixed_det = [0, 0]
    junk_det = [0, 0]
    far_det = [0, 0]
    pfail = 0

    for u in units:
        v = res.get(u.uid, {})
        k = len(u.payload["items"])
        gs = _groups_valid(v, k)
        if gs is None:
            pfail += 1
            continue
        co = _co_pairs(gs)
        ctx = u.truth["ctx"]

        if ctx == "pure":
            pure_one[1] += 1
            pure_one[0] += int(len(gs) == 1)
            tot = k * (k - 1) // 2
            same_co[0] += len(co)
            same_co[1] += tot
            iso[0] += sum(1 for g in gs if len(g) == 1)
            iso[1] += k

        elif ctx == "mixed":
            side = u.truth["side"]
            cj = ct = 0
            for a, b in combinations(range(1, k + 1), 2):
                if side[a - 1] == side[b - 1]:
                    same_co[1] += 1
                    same_co[0] += int((min(a, b), max(a, b)) in co)
                else:
                    ct += 1
                    diff_sp[1] += 1
                    joined = (min(a, b), max(a, b)) in co
                    diff_sp[0] += int(not joined)
                    cj += int(joined)
            mixed_det[1] += 1
            mixed_det[0] += int(ct > 0 and cj / ct < 0.5)

        elif ctx == "junk":
            pp = u.truth["plant_pos"]
            junk_det[1] += 1
            junk_det[0] += int(any(g == [pp] for g in gs))

        elif ctx == "far":
            pp = u.truth["plant_pos"]
            far_det[1] += 1
            far_det[0] += int(any(g == [pp] for g in gs))

    def _r(c): return round(c[0] / c[1], 3) if c[1] else None

    Sp = same_co[0] / same_co[1] if same_co[1] else 0.9
    Se = diff_sp[0] / diff_sp[1] if diff_sp[1] else 0.9
    gamma = iso[0] / iso[1] if iso[1] else 0.02
    denom = Se + Sp - 1
    ok_corr = denom >= 0.2

    checks = {
        "pure_kept_whole":    {"score": _r(pure_one),  "want": "≥ 0.70",
                               "pass": (_r(pure_one)  or 0) >= 0.70, "n": pure_one[1]},
        "mixture_split":      {"score": _r(mixed_det), "want": "≥ 0.70",
                               "pass": (_r(mixed_det) or 0) >= 0.70, "n": mixed_det[1]},
        "junk_isolated":      {"score": _r(junk_det),  "want": "≥ 0.80",
                               "pass": (_r(junk_det)  or 0) >= 0.80, "n": junk_det[1]},
        "far_intruder_found": {"score": _r(far_det),   "want": "≥ 0.80",
                               "pass": (_r(far_det)   or 0) >= 0.80, "n": far_det[1]},
        "correction_usable":  {"score": round(denom, 3), "want": "Se+Sp−1 ≥ 0.20",
                               "pass": ok_corr, "n": same_co[1] + diff_sp[1]},
    }
    overall_pass = all(c["pass"] for c in checks.values())
    return {"Sp": round(Sp, 3), "Se": round(Se, 3), "gamma": round(gamma, 3),
            "denom": round(denom, 3), "ok_corr": ok_corr,
            "checks": checks, "overall_pass": overall_pass, "parse_fail": pfail}


def _build_cal_units(emb_n, codes, cid_of, K, text_arr, theme_arr, states, cfg,
                     prng, rng, uid_counter) -> List[Unit]:
    """Build planted-control units: pure/mixed/junk/far."""
    units: List[Unit] = []
    judged = [s for s in states.values() if s["judged"] and len(s["pool"]) >= cfg.k_partition]
    if not judged or K < 2:
        return units

    k = cfg.k_partition
    n_items = len(text_arr)

    def items_of(ids):
        return [_clip(text_arr[i], cfg.item_chars) for i in ids]

    def themes_of(ids):
        return [str(theme_arr[i]) for i in ids] if theme_arr is not None else None

    def U(payload, truth):
        uid_counter[0] += 1
        return Unit(uid_counter[0], "_cal_", payload, truth)

    # Far pool: each cluster's far-side neighbours (excluding self)
    cent = np.zeros((K, emb_n.shape[1]), dtype=np.float32)
    for c in range(K):
        idxs = np.where(codes == c)[0]
        if len(idxs):
            cent[c] = emb_n[idxs].mean(axis=0)
    sims = cent @ cent.T
    np.fill_diagonal(sims, -2.0)
    nb_order = np.argsort(-sims, axis=1)
    far_pool_by_code = nb_order[:, K // 2: -1] if K > 3 else nb_order[:, :-1]

    # Pure: within-cluster kNN — tests that same-cluster items stay together (→ Sp)
    pool_pure = [s for s in judged if len(s["pool"]) >= k]
    for _ in range(cfg.n_pure):
        if not pool_pure:
            break
        st = prng.choice(pool_pure)
        center = int(prng.choice(list(st["pool"])))
        s = emb_n[st["pool"]] @ emb_n[center]
        ids = [int(x) for x in st["pool"][np.argsort(-s)[:k]]]
        prng.shuffle(ids)
        units.append(U({"items": items_of(ids)},
                       {"ctx": "pure", "themes": themes_of(ids)}))

    # Mixed: half from A + half from far B — tests different-cluster items are split (→ Se)
    for _ in range(cfg.n_mixed):
        if not pool_pure:
            break
        a = prng.choice(pool_pure)
        b_code = int(prng.choice(far_pool_by_code[a["code"]]))
        b = states[cid_of[b_code]]
        if not len(b["pool"]):
            continue
        ka = k // 2
        ia = [int(x) for x in rng.choice(a["pool"], size=min(ka, len(a["pool"])), replace=False)]
        ib = [int(x) for x in rng.choice(b["pool"], size=min(k - len(ia), len(b["pool"])), replace=False)]
        ids = ia + ib
        side = [0] * len(ia) + [1] * len(ib)
        order = list(range(len(ids)))
        prng.shuffle(order)
        units.append(U({"items": items_of([ids[i] for i in order])},
                       {"ctx": "mixed", "side": [side[i] for i in order],
                        "themes": themes_of([ids[i] for i in order])}))

    # Junk: random word soup planted among cluster items (sanity / easy detection floor)
    words = " ".join(_clip(text_arr[i], 80) for i in rng.choice(n_items, size=20)).split()
    for _ in range(cfg.n_junk):
        if not pool_pure:
            break
        st = prng.choice(pool_pure)
        home = [int(x) for x in rng.choice(st["pool"], size=k - 1, replace=False)]
        junk_text = " ".join(prng.choice(words) for _ in range(12))
        pos = prng.randrange(k)
        items = items_of(home[:pos]) + [junk_text] + items_of(home[pos:])
        th_list = ((themes_of(home[:pos]) or []) + ["JUNK"] +
                   (themes_of(home[pos:]) or [])) if theme_arr is not None else None
        units.append(U({"items": items},
                       {"ctx": "junk", "plant_pos": pos + 1, "themes": th_list}))

    # Far intruder: item from far cluster planted among cluster items (strong detection)
    for _ in range(cfg.n_far):
        if not pool_pure:
            break
        st = prng.choice(pool_pure)
        b = states[cid_of[int(prng.choice(far_pool_by_code[st["code"]]))]]
        if not len(b["pool"]):
            continue
        home = [int(x) for x in rng.choice(st["pool"], size=k - 1, replace=False)]
        plant = int(prng.choice(list(b["pool"])))
        pos = prng.randrange(k)
        ids = home[:pos] + [plant] + home[pos:]
        units.append(U({"items": items_of(ids)},
                       {"ctx": "far", "plant_pos": pos + 1,
                        "themes": themes_of(ids)}))

    return units


# ══════════════════════════════════════════════════════════════════════════════
# Intruder measurement
# ══════════════════════════════════════════════════════════════════════════════

def _build_intruder_units(judged_states, nb_order, cid_of, K, states, text_arr, theme_arr,
                          cfg, prng, rng, uid_counter) -> List[Unit]:
    """One wave of intruder PARTITION draws across all judged clusters."""
    units: List[Unit] = []
    far_pool = nb_order[:, K // 2: -1] if K > 3 else nb_order[:, :-1]

    def items_of(ids):
        return [_clip(text_arr[i], cfg.item_chars) for i in ids]

    def themes_of(ids):
        return [str(theme_arr[i]) for i in ids] if theme_arr is not None else None

    for st in judged_states:
        if st["intr_done"] or len(st["pool"]) < cfg.k_partition - 1:
            continue
        near = [int(x) for x in nb_order[st["code"]][:-1]][:cfg.neighbor_m]
        sources = near + ["far"]
        for j in range(cfg.intruder_per_wave):
            src = sources[(uid_counter[0] // max(1, len(judged_states)) + j) % len(sources)]
            if src == "far":
                b_code = int(prng.choice(far_pool[st["code"]]))
                b = states[cid_of[b_code]]
                tag = "far"
            else:
                b = states[cid_of[src]]
                tag = b["cid"]
            if not len(b["pool"]):
                continue
            home = [int(x) for x in rng.choice(st["pool"], size=cfg.k_partition - 1, replace=False)]
            plant = int(prng.choice(list(b["pool"])))
            pos = prng.randrange(cfg.k_partition)
            ids = home[:pos] + [plant] + home[pos:]
            uid_counter[0] += 1
            units.append(Unit(uid_counter[0], st["cid"],
                              {"items": items_of(ids)},
                              {"ctx": "intr", "plant_pos": pos + 1, "plant_src": tag,
                               "themes": themes_of(ids)}))
    return units


def _ingest_intruder(wave: List[Unit], res: Dict[int, dict], states: Dict[str, dict]) -> None:
    for u in wave:
        if u.truth.get("ctx") != "intr":
            continue
        v = res.get(u.uid, {})
        k = len(u.payload["items"])
        gs = _groups_valid(v, k)
        if gs is None:
            states[u.cid]["parse_fail"] += 1
            continue
        pp = u.truth["plant_pos"]
        detected = any(g == [pp] for g in gs)
        src = u.truth["plant_src"]
        rec = states[u.cid]["det"].setdefault(src, [0, 0])
        rec[0] += int(detected)
        rec[1] += 1


def _near_counts(det: dict) -> Tuple[int, int]:
    d = n = 0
    for s, (di, ti) in det.items():
        if s != "far":
            d += di
            n += ti
    return d, n


def _check_intr_done(judged_states, nb_order, cfg, cal) -> None:
    """Mark intruder pass done when near-detection CI has settled."""
    far_rate = cal.get("checks", {}).get("far_intruder_found", {}).get("score") or 0.5
    gamma = cal.get("gamma", 0.02)
    det_bar_raw = 0.5 * (1 - gamma) + gamma
    for st in judged_states:
        det, n = _near_counts(st["det"])
        if n >= 6:
            _, dlo, dhi = _wilson(det, n, cfg.z)
            if dhi < det_bar_raw or dlo > det_bar_raw:
                st["intr_done"] = True


# ══════════════════════════════════════════════════════════════════════════════
# Assembly
# ══════════════════════════════════════════════════════════════════════════════

def _score_cluster(st: dict, cal: dict, cfg: Config) -> dict:
    det, n = _near_counts(st["det"])
    gamma = cal.get("gamma", 0.02)
    if n:
        d, dl, dh = _wilson(det, n, cfg.z)
        score = _rg_det(d, gamma)
        lo = _rg_det(dl, gamma)
        hi = _rg_det(dh, gamma)
        return {"score": round(score, 3), "lo": round(lo, 3), "hi": round(hi, 3), "n": n}
    return None


def _assemble(states: Dict[str, dict], cal: dict, cfg: Config,
              labels: Dict[str, str], df: pd.DataFrame, client: LLMClient) -> dict:
    det_threshold = 0.5  # corrected detection rate above which a cluster is considered distinct

    judged = [s for s in states.values() if s["judged"]]
    clusters = []
    for st in sorted(states.values(), key=lambda s: -s["size"]):
        c: Dict[str, Any] = {
            "cluster_id": st["cid"],
            "label": labels.get(st["cid"], st["cid"]),
            "size": st["size"],
            "judged": st["judged"],
        }
        if st["judged"]:
            dist = _score_cluster(st, cal, cfg)
            c["distinctiveness"] = dist
            c["distinct"] = bool(dist and dist["score"] >= det_threshold)
            c["parse_fail"] = st["parse_fail"]
        clusters.append(c)

    # Size-weighted distinctiveness fraction (the headline KPI)
    num = den = 0.0
    w_score_num = w_score_den = 0.0
    for c in clusters:
        if not c["judged"] or c["distinctiveness"] is None:
            continue
        w = float(c["size"])
        num += w * float(c["distinct"])
        den += w
        w_score_num += w * c["distinctiveness"]["score"]
        w_score_den += w
    weighted_distinct_rate = round(num / den, 3) if den else None
    weighted_distinct_score = round(w_score_num / w_score_den, 3) if w_score_den else None

    n_judged = len(judged)
    n_distinct = sum(1 for c in clusters if c.get("distinct"))
    health = {
        "n_calls": client.n_calls,
        "n_empty": client.n_empty,
        "empty_rate": round(client.n_empty / client.n_calls, 3) if client.n_calls else 0.0,
        "ok": (client.n_empty / client.n_calls if client.n_calls else 0.0) <= cfg.max_empty_rate,
    }
    overall_pass = cal["overall_pass"] and health["ok"]

    return {
        "meta": {
            "unit": cfg.unit,
            "same_when": cfg.same_when,
            "model": "mock" if client.mock else cfg.model,
            "n_texts": int(len(df)),
            "n_clusters": len(states),
            "n_judged": n_judged,
            "n_llm_calls": client.n_calls,
        },
        "kpi": {
            "weighted_distinct_rate": weighted_distinct_rate,
            "weighted_distinct_score": weighted_distinct_score,
            "n_distinct": n_distinct,
            "n_judged": n_judged,
            "threshold": det_threshold,
        },
        "calibration": {**cal, "judge_health": health, "overall_pass": overall_pass},
        "clusters": clusters,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main evaluate()
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(data, embeddings=None, *, same_when: str = "", unit: str = "",
             model: str = "mock", config: Optional[Config] = None,
             text_col="text", cluster_col="cluster_id", label_col="label",
             embedding_col=None, progress: bool = True) -> dict:
    """Run the distinctiveness evaluation. Returns a results dict."""
    cfg = config or Config(same_when=same_when or "are the same kind",
                           unit=unit or "each text is a short customer message",
                           model=model)
    if same_when:
        cfg.same_when = same_when
    if unit:
        cfg.unit = unit
    if not cfg.same_when:
        raise ValueError("same_when is required")

    df, emb, labels = _coerce(data, embeddings, text_col=text_col, cluster_col=cluster_col,
                               label_col=label_col, embedding_col=embedding_col)
    df = df.reset_index(drop=True)
    rng = np.random.default_rng(cfg.seed)
    prng = random.Random(cfg.seed)
    uid_counter = [0]

    # Build cluster index structures
    cats = pd.Categorical(df["cluster_id"].astype(str))
    codes = np.asarray(cats.codes)
    cid_of = {i: str(c) for i, c in enumerate(cats.categories)}
    K = len(cats.categories)
    text_arr = df["text"].astype(str).to_numpy()
    theme_arr = df["_theme"].to_numpy() if "_theme" in df.columns else None
    emb_n = _normalize(emb)

    # Centroids + neighbour order (for intruder source selection)
    cent = np.zeros((K, emb_n.shape[1]), dtype=np.float32)
    for c in range(K):
        idxs = np.where(codes == c)[0]
        if len(idxs):
            cent[c] = emb_n[idxs].mean(axis=0)
    sims = cent @ cent.T
    np.fill_diagonal(sims, -2.0)
    nb_order = np.argsort(-sims, axis=1)

    # Per-cluster state
    states: Dict[str, dict] = {}
    for c in range(K):
        idxs = np.where(codes == c)[0]
        pool = _pool(emb_n, idxs, cfg, rng)
        states[cid_of[c]] = {
            "cid": cid_of[c], "code": c, "size": int(len(idxs)), "pool": pool,
            "judged": len(pool) >= cfg.min_judgeable,
            "det": {},          # plant_src -> [detected, total]
            "intr_done": False,
            "parse_fail": 0,
        }
    judged_states = [s for s in states.values() if s["judged"]]
    for st in judged_states:
        if K < 2 or len(st["pool"]) < cfg.k_partition - 1:
            st["intr_done"] = True

    gw = _discover_gateway() if cfg.model != "mock" else None
    client = LLMClient(cfg, fn=gw)

    with ThreadPoolExecutor(max_workers=max(1, cfg.workers)) as executor:
        # Calibration
        log.info("calibrating (%d pure + %d mixed + %d junk + %d far controls)…",
                 cfg.n_pure, cfg.n_mixed, cfg.n_junk, cfg.n_far)
        cal_units = _build_cal_units(emb_n, codes, cid_of, K, text_arr, theme_arr,
                                     states, cfg, prng, rng, uid_counter)
        room = max(0, cfg.max_llm_calls - client.n_calls)
        cal_units = cal_units[:room]
        cal_res = _run_units(cal_units, client, cfg, desc="calibrating",
                             progress=progress, executor=executor)
        cal = _calibrate(cal_units, cal_res)
        log.info("calibration: gate=%s  Sp=%.2f  Se=%.2f  gamma=%.3f  ok_corr=%s",
                 "PASS" if cal["overall_pass"] else "FAIL",
                 cal["Sp"], cal["Se"], cal["gamma"], cal["ok_corr"])

        # Intruder detection waves
        log.info("measuring intruder detection (%d judged clusters, %d waves)…",
                 len(judged_states), cfg.intruder_waves_max)
        for w in range(cfg.intruder_waves_max):
            wave = _build_intruder_units(judged_states, nb_order, cid_of, K, states,
                                         text_arr, theme_arr, cfg, prng, rng, uid_counter)
            room = max(0, cfg.max_llm_calls - client.n_calls)
            wave = wave[:room]
            if not wave:
                break
            res = _run_units(wave, client, cfg, desc=f"intruder wave {w + 1}",
                             progress=progress, executor=executor)
            _ingest_intruder(wave, res, states)
            _check_intr_done(judged_states, nb_order, cfg, cal)
            if all(s["intr_done"] for s in judged_states):
                break

    return _assemble(states, cal, cfg, labels, df, client)


# ══════════════════════════════════════════════════════════════════════════════
# HTML report
# ══════════════════════════════════════════════════════════════════════════════

def render_html(results: dict) -> str:
    kpi = results.get("kpi", {})
    cal = results.get("calibration", {})
    meta = results.get("meta", {})
    clusters = results.get("clusters", [])
    checks = cal.get("checks", {})

    score = kpi.get("weighted_distinct_rate")
    score_raw = kpi.get("weighted_distinct_score")
    gate = cal.get("overall_pass", False)
    n_distinct = kpi.get("n_distinct", 0)
    n_judged = kpi.get("n_judged", 0)
    threshold = kpi.get("threshold", 0.5)

    pct = f"{score * 100:.1f}%" if score is not None else "—"
    color = ("#2a7a3b" if score is not None and score >= 0.75
             else "#b85c00" if score is not None and score >= 0.50
             else "#c0392b")
    gate_color = "#2a7a3b" if gate else "#c0392b"
    gate_label = "PASS" if gate else "FAIL"

    def _pct(x): return f"{x * 100:.0f}%" if x is not None else "—"
    def _chk(ok): return "✓" if ok else "✗"

    check_rows = ""
    check_labels = {
        "pure_kept_whole":    "Same-kind items kept together",
        "mixture_split":      "Different-kind items separated",
        "junk_isolated":      "Nonsense text isolated",
        "far_intruder_found": "Far-cluster intruder detected",
        "correction_usable":  "Error correction usable (Se+Sp−1 ≥ 0.20)",
    }
    for key, info in checks.items():
        lbl = check_labels.get(key, key)
        ok = info.get("pass", False)
        chk_color = "#2a7a3b" if ok else "#c0392b"
        check_rows += (
            f"<tr><td>{lbl}</td>"
            f"<td style='text-align:center;color:{chk_color};font-weight:bold'>{_chk(ok)}</td>"
            f"<td style='text-align:right'>{_pct(info.get('score'))}</td>"
            f"<td style='text-align:right;color:#666'>{info.get('want','')}</td>"
            f"<td style='text-align:right;color:#888'>{info.get('n','')}</td></tr>\n"
        )

    cluster_rows = ""
    for c in clusters:
        if not c.get("judged"):
            continue
        dist = c.get("distinctiveness") or {}
        sc = dist.get("score")
        lo = dist.get("lo")
        hi = dist.get("hi")
        n = dist.get("n", 0)
        is_dist = c.get("distinct", False)
        row_color = "#f0faf2" if is_dist else "#fff8f8"
        badge = (f"<span style='color:#2a7a3b;font-weight:bold'>✓ distinct</span>"
                 if is_dist else
                 f"<span style='color:#c0392b;font-weight:bold'>✗ indistinct</span>")
        ci = f"{lo*100:.0f}–{hi*100:.0f}%" if lo is not None and hi is not None else "—"
        cluster_rows += (
            f"<tr style='background:{row_color}'>"
            f"<td>{c['cluster_id']}</td>"
            f"<td>{c.get('label', c['cluster_id'])}</td>"
            f"<td style='text-align:right'>{c['size']:,}</td>"
            f"<td style='text-align:right'>{_pct(sc)}</td>"
            f"<td style='text-align:right;color:#666'>{ci}</td>"
            f"<td style='text-align:right;color:#888'>{n}</td>"
            f"<td>{badge}</td></tr>\n"
        )

    not_judged = [c for c in clusters if not c.get("judged")]
    small_note = (f"<p style='color:#888;font-size:0.85em'>{len(not_judged)} cluster(s) "
                  f"had fewer than {meta.get('min_judgeable', 5)} items and were not judged.</p>"
                  if not_judged else "")

    gate_warn = ""
    if not gate:
        gate_warn = """
        <div style='background:#fff3cd;border:1px solid #ffc107;padding:1em;border-radius:6px;margin-bottom:1.5em'>
          <strong>⚠ Calibration gate FAILED.</strong> The judge model could not reliably separate
          same-kind from different-kind items in planted controls. The distinctiveness numbers below
          may not be trustworthy. Check your <code>same_when</code> rule and judge gateway.
        </div>"""

    health = cal.get("judge_health", {})
    health_note = ""
    if not health.get("ok", True):
        rate = health.get("empty_rate", 0)
        health_note = (f"<p style='color:#c0392b'><strong>⚠ Judge health:</strong> "
                       f"{rate*100:.0f}% of calls returned empty/unparseable responses "
                       f"(ceiling: {health.get('max_empty_rate', 0.25)*100:.0f}%). "
                       f"Results are not reliable.</p>")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cluster Distinctiveness Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          max-width: 960px; margin: 2em auto; padding: 0 1.5em; color: #1a1a1a; line-height: 1.6; }}
  h1 {{ font-size: 1.6em; font-weight: 700; margin-bottom: 0.2em; }}
  h2 {{ font-size: 1.15em; font-weight: 600; margin-top: 2em; border-bottom: 1px solid #e0e0e0;
        padding-bottom: 0.3em; }}
  .headline {{ display: flex; align-items: center; gap: 2em; margin: 1.5em 0; flex-wrap: wrap; }}
  .big-number {{ font-size: 4em; font-weight: 800; color: {color}; line-height: 1; }}
  .subtext {{ color: #555; font-size: 0.95em; }}
  .gate-badge {{ display: inline-block; padding: 0.3em 0.9em; border-radius: 4px;
                  font-weight: 700; font-size: 0.95em;
                  background: {'#e8f5eb' if gate else '#fdecea'};
                  color: {gate_color}; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.9em; margin-top: 0.8em; }}
  th {{ text-align: left; padding: 0.5em 0.75em; background: #f5f5f5;
        border-bottom: 2px solid #ddd; font-weight: 600; }}
  td {{ padding: 0.45em 0.75em; border-bottom: 1px solid #eee; }}
  tr:last-child td {{ border-bottom: none; }}
  .meta {{ color: #666; font-size: 0.85em; margin-bottom: 1em; }}
  .explain {{ background: #f9f9f9; border-left: 4px solid #ccc; padding: 1em 1.25em;
              border-radius: 0 6px 6px 0; margin: 1em 0; }}
  .explain p {{ margin: 0.4em 0; }}
  code {{ background: #eef; padding: 0.1em 0.3em; border-radius: 3px; font-size: 0.9em; }}
</style>
</head>
<body>

<h1>Cluster Distinctiveness Report</h1>
<p class="meta">
  {meta.get('n_clusters', '?')} clusters · {meta.get('n_texts', '?'):,} texts ·
  {n_judged} judged · model: <code>{meta.get('model', '?')}</code>
</p>

{gate_warn}

<div class="headline">
  <div>
    <div class="big-number">{pct}</div>
    <div class="subtext" style="margin-top:0.3em">
      Weighted distinctiveness<br>
      <span style="font-size:0.9em;color:#888">
        {n_distinct} of {n_judged} judged clusters pass the {threshold*100:.0f}% threshold
      </span>
    </div>
  </div>
  <div>
    <div class="subtext">Calibration gate</div>
    <div class="gate-badge">{gate_label}</div>
    <div class="subtext" style="margin-top:0.5em">
      Sp = {_pct(cal.get('Sp'))} &nbsp;·&nbsp;
      Se = {_pct(cal.get('Se'))} &nbsp;·&nbsp;
      γ = {_pct(cal.get('gamma'))}
    </div>
  </div>
</div>

{health_note}

<h2>What this number means</h2>
<div class="explain">
  <p><strong>Weighted distinctiveness ({pct})</strong> is the share of your text corpus
  that lives in clusters well-separated from their nearest neighbours, as judged by
  the LLM under your equivalence rule.</p>
  <p>A cluster scores high when items from a neighbouring cluster, secretly planted
  among its members, are reliably detected and isolated. A low score means the
  cluster bleeds into its neighbours — the LLM cannot tell them apart under your rule.</p>
  <p>The weighting is by cluster size, so a large cluster that fails pulls the score
  down more than a small one that fails.</p>
</div>

<h2>How it is computed</h2>
<div class="explain">
  <p><strong>Step 1 — Calibration.</strong> Before measuring anything, the LLM is run on
  planted controls of known composition: groups of truly-same items (→ measures specificity Sp),
  mixed groups of items from two different clusters (→ sensitivity Se), nonsense text (→ junk
  detection), and items from far-away clusters (→ far detection). The gate passes only if the
  judge is reliable enough on all four checks.</p>
  <p><strong>Step 2 — Intruder detection.</strong> For each cluster, one item from a
  neighbouring cluster is planted among <code>k={meta.get("k_partition","?")}</code> home items
  and the LLM is asked to sort them by kind. If the planted item ends up as its own singleton
  group it is <em>detected</em>. This is repeated across several waves and neighbour sources.</p>
  <p><strong>Step 3 — Correction.</strong> Raw detection rates are corrected for the chance
  isolation rate γ (how often the LLM isolates a truly-same item by accident) using the
  Rogan–Gladen formula: <code>corrected = (raw − γ) / (1 − γ)</code>.</p>
  <p><strong>Step 4 — Aggregation.</strong> Each cluster's corrected detection rate is thresholded
  at {threshold*100:.0f}%. The headline number is the size-weighted fraction of clusters above
  that threshold (raw weighted average score: {_pct(score_raw)}).</p>
</div>

<h2>How much to trust it</h2>
<div class="explain">
  <p>Trust rests on two things:</p>
  <p><strong>Calibration gate: <span style="color:{gate_color}">{gate_label}</span>.</strong>
  A PASS means the judge separated known-good from known-bad controls reliably enough for
  the error correction to be meaningful. A FAIL means the numbers below are unreliable —
  the judge or the rule needs fixing before you read the scores.</p>
  <p><strong>Per-cluster confidence intervals</strong> (lo–hi columns in the table below)
  are 95% Wilson intervals on the corrected detection rate. Clusters with few intruder draws
  (<code>n</code>) have wide intervals. Raise <code>intruder_waves_max</code> or
  <code>coverage_target</code> to narrow them.</p>
</div>

<h2>Calibration checks</h2>
<table>
  <tr><th>Check</th><th style="text-align:center">Pass</th>
      <th style="text-align:right">Score</th><th style="text-align:right">Target</th>
      <th style="text-align:right">n</th></tr>
  {check_rows}
</table>
<p class="meta" style="margin-top:0.5em">
  Sp={_pct(cal.get('Sp'))} (specificity: same items co-grouped) ·
  Se={_pct(cal.get('Se'))} (sensitivity: diff items separated) ·
  γ={_pct(cal.get('gamma'))} (chance isolation rate) ·
  Se+Sp−1={_pct(cal.get('denom'))} (correction denominator)
</p>

<h2>Per-cluster distinctiveness</h2>
{small_note}
<table>
  <tr><th>ID</th><th>Label</th><th style="text-align:right">Size</th>
      <th style="text-align:right">Score</th><th style="text-align:right">95% CI</th>
      <th style="text-align:right">n draws</th><th>Verdict</th></tr>
  {cluster_rows}
</table>

</body>
</html>"""
    return html


def write_html(results: dict, path: str) -> None:
    import os
    if os.path.isdir(path):
        path = os.path.join(path, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_html(results))
    log.info("report written to %s", path)


# ══════════════════════════════════════════════════════════════════════════════
# Demo data
# ══════════════════════════════════════════════════════════════════════════════

def make_demo_data(seed: int = 7, dim: int = 32):
    """Small synthetic dataset: 5 distinct clusters + 3 confusable clusters (G, H, I).

    G, H, and I all contain items of the same theme (6) and are placed close together
    in embedding space, so each one's nearest neighbours are the other two. The mock
    judge cannot detect an item from H planted among G's items (same theme), so all
    three should score low on distinctiveness. Clusters A–E are cleanly separated and
    score near 1.0.
    """
    rng = np.random.default_rng(seed)
    n_themes = 6
    centers = rng.normal(0, 1, (n_themes, dim)).astype(np.float32)
    norms = np.linalg.norm(centers, axis=1, keepdims=True)
    centers = centers / norms

    # G, H, I share theme 5 and sit in a tight cluster in embedding space
    shared_center = centers[5].copy()
    offsets = rng.normal(0, 0.02, (3, dim)).astype(np.float32)

    rows = []
    plan = [
        ("A", 0, 200, centers[0]),
        ("B", 1, 160, centers[1]),
        ("C", 2, 140, centers[2]),
        ("D", 3, 120, centers[3]),
        ("E", 4, 100, centers[4]),
        ("G", 5,  90, shared_center + offsets[0]),   # same theme as H, I
        ("H", 5,  90, shared_center + offsets[1]),   # same theme as G, I
        ("I", 5,  80, shared_center + offsets[2]),   # same theme as G, H
    ]

    for cid, th, n, base in plan:
        for j in range(n):
            noise = rng.normal(0, 0.08, dim).astype(np.float32)
            rows.append((f"cluster {cid} item {j}: text about topic {th}", cid, th, base + noise))

    df = pd.DataFrame(rows, columns=["text", "cluster_id", "_theme", "_emb"])
    emb = np.vstack(df["_emb"].to_numpy()).astype(np.float32)
    df = df.drop(columns=["_emb"])
    return df, emb


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def configure_logging(level=logging.INFO):
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    log.handlers[:] = [h]
    log.setLevel(level)
    log.propagate = False


def main():
    configure_logging()
    p = argparse.ArgumentParser(description="Weighted cluster distinctiveness evaluator")
    p.add_argument("--demo", action="store_true", help="Run offline demo (no gateway needed)")
    p.add_argument("--data", help="Path to data file (csv/tsv/parquet/jsonl/json)")
    p.add_argument("--embeddings", help="Path to .npy embeddings file")
    p.add_argument("--embedding-col", help="Column in data file holding embeddings")
    p.add_argument("--same-when", help="Equivalence rule (required unless --demo)")
    p.add_argument("--unit", default="each text is a short customer message")
    p.add_argument("--model", default="mock")
    p.add_argument("--out", default="report.html", help="Output HTML path")
    p.add_argument("--workers", type=int, default=64)
    p.add_argument("--coverage", type=float, default=0.20)
    args = p.parse_args()

    if args.demo:
        print("Running offline demo…")
        df, emb = make_demo_data()
        cfg = Config(same_when="are about the same topic", unit="each text is a short customer message",
                     model="mock", workers=4, coverage_target=0.35)
        results = evaluate(df, emb, config=cfg, progress=True)
    elif args.data:
        if not args.same_when:
            p.error("--same-when is required")
        emb = np.load(args.embeddings) if args.embeddings else None
        cfg = Config(same_when=args.same_when, unit=args.unit, model=args.model,
                     workers=args.workers, coverage_target=args.coverage)
        results = evaluate(args.data, emb, config=cfg,
                           embedding_col=args.embedding_col, progress=True)
    else:
        p.print_help()
        return

    kpi = results["kpi"]
    cal = results["calibration"]
    print(f"\nWeighted distinctiveness: {kpi['weighted_distinct_rate'] * 100:.1f}%"
          if kpi.get("weighted_distinct_rate") is not None else "\nWeighted distinctiveness: —")
    print(f"  ({kpi['n_distinct']} of {kpi['n_judged']} judged clusters pass "
          f"the {kpi['threshold']*100:.0f}% threshold)")
    print(f"Calibration gate: {'PASS' if cal['overall_pass'] else 'FAIL'}  "
          f"Sp={cal['Sp']:.2f}  Se={cal['Se']:.2f}  γ={cal['gamma']:.3f}")

    out = args.out if not args.demo else "demo_report.html"
    write_html(results, out)
    print(f"HTML report: {out}")


if __name__ == "__main__":
    main()
