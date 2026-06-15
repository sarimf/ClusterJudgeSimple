"""Regression tests for cluster_judge.

The engine exists to produce correct numbers from an *imperfect* judge, so the headline
tests run the offline mock with injected error (Config.mock_eps_*) and assert that the
Rogan-Gladen correction recovers ground truth. Pure helpers are unit-tested directly.
"""
import json
import os

import numpy as np
import pytest

import cluster_judge as cj


# --------------------------------------------------------------------------- #
# Small synthetic dataset: distinct-theme clusters + one 70/30-mixed "GIANT".
# Every non-giant cluster owns a unique theme, so there are no genuine merges
# and pure clusters are unambiguously single-class.
# --------------------------------------------------------------------------- #
def _small_dataset(seed=7, dim=32):
    rng = np.random.default_rng(seed)
    n_themes = 8
    centers = rng.normal(0, 1, (n_themes, dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    rows = []

    def emit(cid, mix, n):
        ths = rng.choice([t for t, _ in mix], size=n, p=[f for _, f in mix])
        for j in range(n):
            rows.append((f"{cid} item {j}", cid, int(ths[j])))

    plan = [("GIANT", [(0, 0.70), (1, 0.30)], 400),
            ("A", [(2, 1.0)], 200), ("B", [(3, 1.0)], 160),
            ("C", [(4, 1.0)], 140), ("D", [(5, 1.0)], 120),
            ("E", [(6, 1.0)], 100), ("F", [(7, 1.0)], 80)]
    for cid, mix, n in plan:
        emit(cid, mix, n)
    df = cj.pd.DataFrame(rows, columns=["text", "cluster_id", "_theme"])
    emb = (centers[df["_theme"].to_numpy()] + rng.normal(0, 0.05, (len(df), dim))).astype(np.float32)
    labels = {}
    for cid, g in df.groupby("cluster_id"):
        dom = int(g["_theme"].value_counts().idxmax())
        labels[str(cid)] = {"label": f"theme {dom}", "_theme": dom,
                            "description": f"objections of theme {dom}"}
    return df, emb, labels


def _clean_cfg(**kw):
    base = dict(same_when="are the same kind of objection", model="mock",
                coverage_target=0.35, workers=4, seed=7)
    base.update(kw)
    return cj.Config(**base)


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def test_wilson():
    p, lo, hi = cj._wilson(5, 10)
    assert p == 0.5 and 0 < lo < 0.5 < hi < 1
    assert cj._wilson(0, 0) == (0.0, 0.0, 1.0)
    p, lo, hi = cj._wilson(10, 10)
    assert p == 1.0 and hi == 1.0 and lo < 1.0


def test_rogan_gladen():
    cal = {"ok_corr": True, "Sp": 0.9, "denom": 0.8, "gamma": 0.02}
    assert cj._rg_mix(0.5, cal) == pytest.approx((0.5 - 0.1) / 0.8)
    assert cj._rg_mix(0.0, cal) == 0.0                         # clamped to [0,1]
    assert cj._rg_det(0.5, cal) == pytest.approx((0.5 - 0.02) / 0.98)
    # no usable correction -> passthrough
    assert cj._rg_mix(0.42, {"ok_corr": False, "Sp": 0.9, "denom": 0.0}) == 0.42


def test_pair_pattern_and_groups_valid():
    assert cj._pair_pattern([[1, 2], [3]], 3) == {(1, 2)}
    assert cj._pair_pattern([[1, 2, 3]], 3) == {(1, 2), (1, 3), (2, 3)}
    assert cj._groups_valid({"groups": [[1, 2], [3]]}, 3) == [[1, 2], [3]]
    assert cj._groups_valid({"groups": [[1, 2]]}, 3) is None         # 3 missing
    assert cj._groups_valid({"groups": [[1, 1, 2, 3]]}, 3) is None   # duplicate index
    assert cj._groups_valid({}, 3) is None


def test_components_from_pairs():
    # two strongly-co-grouped pairs with no cross edges -> two components
    pair = {(1, 2): [4, 4, {10, 11}], (3, 4): [4, 4, {12, 13}]}
    comps = cj._components_from_pairs({1, 2, 3, 4}, pair, 0.5, 0.9, 0.9)
    assert {frozenset(c) for c in comps} == {frozenset({1, 2}), frozenset({3, 4})}

    # fully co-grouped triangle -> one component
    pair = {(1, 2): [3, 3, {10, 11}], (2, 3): [3, 3, {11, 12}], (1, 3): [3, 3, {10, 12}]}
    comps = cj._components_from_pairs({1, 2, 3}, pair, 0.5, 0.9, 0.9)
    assert {frozenset(c) for c in comps} == {frozenset({1, 2, 3})}


def test_fair_trim():
    us = [cj.Unit(i, "partition", cid, {}, {})
          for i, cid in enumerate(["a", "a", "a", "b", "b", "c"])]
    assert cj._fair_trim(us, 0) == []
    assert len(cj._fair_trim(us, 100)) == 6
    kept = cj._fair_trim(us, 3)
    assert len(kept) == 3
    assert {u.cid for u in kept} == {"a", "b", "c"}      # one from each cluster, not 3x"a"


def test_fit_bands_isolates_singletons():
    bands = cj._fit_bands([1, 1, 1, 50, 60, 200, 5000], cj.Config())
    assert bands[0]["band"] == "1" and bands[0]["lo"] == 1 and bands[0]["hi"] == 1
    assert any(b["hi"] >= 5000 for b in bands)


# --------------------------------------------------------------------------- #
# Config validation
# --------------------------------------------------------------------------- #
def test_config_validation():
    cj.Config()                                              # defaults are valid
    with pytest.raises(ValueError):
        cj.Config(coverage_target=0.5, coverage_ceiling=0.4)
    with pytest.raises(ValueError):
        cj.Config(k_partition=2)
    with pytest.raises(ValueError):
        cj.Config(max_empty_rate=1.5)
    with pytest.raises(ValueError):
        cj.Config(workers=0)


# --------------------------------------------------------------------------- #
# End-to-end on the mock judge
# --------------------------------------------------------------------------- #
def test_clean_run_gate_and_verdicts():
    df, emb, labels = _small_dataset()
    R = cj.evaluate(df, emb, dict(labels), _clean_cfg(), progress=False)

    assert R["calibration"]["overall_pass"] is True
    assert R["calibration"]["judge_health"]["ok"] is True
    assert R["calibration"]["judge_health"]["n_empty"] == 0

    by_id = {c["cluster_id"]: c for c in R["clusters"]}
    assert by_id["GIANT"]["split"] is True                   # 70/30 mix must be caught
    pure = [c for cid, c in by_id.items() if cid != "GIANT" and c["judged"]]
    assert all(c["one_class"] for c in pure)                 # distinct-theme clusters are single-class
    assert R["merge_groups"] == []                           # no shared themes -> no merges


def test_correction_recovers_truth_under_noise():
    df, emb, labels = _small_dataset()
    # workers=1 keeps the shared mock RNG deterministic under injected noise
    cfg = _clean_cfg(workers=1, mock_eps_split=0.08, mock_eps_join=0.05)
    R = cj.evaluate(df, emb, dict(labels), cfg, progress=False)

    by_id = {c["cluster_id"]: c for c in R["clusters"]}
    assert by_id["GIANT"]["split"] is True                   # still caught despite noise
    pure = [c for cid, c in by_id.items() if cid != "GIANT" and c["judged"]]
    # corrected mixing must separate the real mix from noisy-but-pure clusters
    assert by_id["GIANT"]["mixing"] > max(c["mixing"] for c in pure)
    assert sum(c["one_class"] for c in pure) >= len(pure) - 1


def test_determinism_clean_run():
    df, emb, labels = _small_dataset()
    a = cj.evaluate(df, emb, dict(labels), _clean_cfg(), progress=False)
    b = cj.evaluate(df, emb, dict(labels), _clean_cfg(), progress=False)
    dump = lambda r: json.dumps(r, sort_keys=True, default=str)
    assert dump(a) == dump(b)


def test_determinism_across_processes():
    # Separate processes get different PYTHONHASHSEED; the mock's tie-breaks must not depend
    # on set iteration order, so the full results JSON must match across hash seeds.
    import subprocess
    import sys

    prog = (
        "import json, numpy as np, cluster_judge as cj\n"
        "import tests.test_cluster_judge as t\n"
        "df, emb, labels = t._small_dataset()\n"
        "R = cj.evaluate(df, emb, dict(labels), t._clean_cfg(), progress=False)\n"
        "print(json.dumps(R, sort_keys=True, default=str))\n"
    )
    outs = []
    for seed in ("0", "1"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        r = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stderr
        outs.append(r.stdout)
    assert outs[0] == outs[1]


def test_recluster_giant_finds_subthemes():
    df, emb, labels = _small_dataset()
    rc = cj.recluster_cluster("GIANT", data=df, embeddings=emb, config=_clean_cfg(), progress=False)
    assert rc["n_sub_clusters"] >= 2                         # themes 0 and 1 resolved
    assert rc["meta"]["judge_health"]["ok"] is True


def test_judge_health_flips_gate_on_failing_gateway():
    df, emb, labels = _small_dataset()
    import random as _r
    rng = _r.Random(0)

    def flaky(messages, json_mode=True):
        if rng.random() < 0.9:
            raise RuntimeError("gateway 503")
        return '{"groups": [[1]]}'

    cj.use_genai(flaky)
    try:
        cfg = _clean_cfg(model="gateway", workers=4, max_retries=0, backoff_base=0.0)
        R = cj.evaluate(df, emb, dict(labels), cfg, progress=False)
    finally:
        cj._GATEWAY["messages_fn"] = None                   # don't leak into other tests

    health = R["calibration"]["judge_health"]
    assert health["ok"] is False
    assert health["empty_rate"] > cfg.max_empty_rate
    assert R["calibration"]["overall_pass"] is False
    assert "JUDGE HEALTH" in cj.render_report(R)
