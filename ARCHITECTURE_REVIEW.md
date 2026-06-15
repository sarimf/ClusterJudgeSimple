# ClusterJudge — Architecture Review & Suggestions

A deep review of `cluster_judge.py` (v3, ~1.9k lines, single module). The engine is
genuinely well-designed: this document is mostly about hardening, testability, and
maintainability, not a redesign. Findings are ordered by impact.

> **Status — all review items implemented.** Done: the test suite (§2.1), fair budget
> trimming (§2.3), the judge-health guard (§3.2), the `recluster_cluster` dead code (§3.1),
> the redundant `_mix_ci` recompute (§3.4), `Config` validation (§4.1), the CLI demo-config
> dedup + a `main()` entry point (§4.3), packaging via `pyproject.toml` (§4.4), the
> demo-leak note (§5), **the `evaluate()` decomposition into phase functions over a shared
> `_Eval` context (§2.2), typed results (`Results`/`ClusterResult`/`Band`, §4.2), single
> shared `ThreadPoolExecutor` across waves (§3.5), and name-based `render_report` columns
> (§5)**. The decomposition was verified byte-identical to the prior `evaluate()` under a
> fixed `PYTHONHASHSEED`, and a cross-process determinism test now guards the mock's
> tie-breaks. The only intentionally-unimplemented note is optional run checkpointing/
> resumability (§5), which would change the on-disk contract and is better scoped on its own.

---

## 1. What the architecture gets right

These are deliberate strengths worth preserving through any refactor.

- **Two-primitive measurement core.** Every metric reduces to `PARTITION` (sort *k*
  shuffled items into same-kind groups) and `FIT` (judge a card against members).
  Collapsing seven ad-hoc tasks into two designs (`prompt_for`, lines 191–231) is the
  central good idea — it keeps the judge contract small and auditable.
- **Generate/detect, never confirm.** Coherence is measured by whether the judge
  *separates* a planted intruder or *splits* a mixed draw, not by asking "is this
  coherent?". This sidesteps LLM rubber-stamping and is the project's main correctness
  claim.
- **Measured-error correction layer.** Planted controls estimate pair-level
  sensitivity/specificity (`_calibrate`, 984–1067), then Rogan–Gladen back-corrects every
  rate (`_rg_mix`, `_rg_det`, 1070–1080). Data-driven thresholds (`tau_edge`,
  `tau_mix_raw`, `det_bar_raw`) live in the same statistic space as the edges they gate.
- **Separation of measurement from presentation.** Naming, strategies, and taxonomy run
  *after* scoring and "can never move a score" (module docstring, lines 16–17). This is
  enforced structurally — presentation calls read `est` but never write it.
- **Flat dispatch.** One work queue, one thread pool per wave (`run_units`, 290–305), so
  a 20k-item cluster never serializes the run.
- **Statistical care throughout.** Wilson intervals, LR-gated agglomeration with
  draw-replication gating to defeat single-draw false-join cliques
  (`_components_from_pairs`, 1083–1137), and conservative confusion lower bounds for merge
  edges.

The rest of this document assumes these are non-negotiable and should survive.

---

## 2. High-impact gaps

### 2.1 There is no test suite — and the code is built to be tested

This is the single biggest gap. The whole correction layer exists to produce correct
numbers from an *imperfect* judge, and the codebase already ships the exact machinery to
prove it works:

- `Config.mock_eps_split` / `mock_eps_join` (lines 123–124) inject known judge error.
- The mock consumes `_theme` ground truth (`_mock`, 311–358).
- `_theme` is plumbed end-to-end so detection/mixing can be checked against truth.

Yet nothing asserts on any of it. The demo runs the pipeline but checks nothing.

**Suggestion.** Add a `tests/` suite (pytest) that runs `make_demo_data()` through
`evaluate` with `model="mock"` and asserts:

- The calibration gate passes at `mock_eps_* = 0` and the giant (70/30 mix) is flagged
  `split` while pure clusters are `one_class`.
- **Correction recovers truth under noise.** With `mock_eps_split=0.08, mock_eps_join=0.05`,
  the *corrected* mixing for pure clusters stays ≤ `tau_mix` and the giant still splits —
  i.e. Rogan–Gladen removes the injected bias. This is the headline claim; it should be a
  regression test.
- `C_GIANT` resolves into ~theme-0 / theme-1 sub-clusters via `recluster_cluster`.
- Determinism: same seed → identical `results` for the mock path.

Also add unit tests for the pure functions, which are easy to test in isolation:
`_wilson`, `_rg_mix`/`_rg_det`, `_pair_pattern`, `_groups_valid`, `_components_from_pairs`
(feed it a hand-built co-graph), and `_fit_bands`.

### 2.2 `evaluate` is a ~340-line monolith

`evaluate` (562–903) runs calibration → measurement waves → resolution waves → merge
groups → FIT → presentation → assemble, all inline with nested closures (`U`, `items_of`,
`part_unit`, `homog_unit`, `guard`). You cannot test or reuse a single phase, and the
closures capture mutable state (`uid`, `calls_by_kind`) that makes the data flow hard to
follow.

**Suggestion.** Extract each phase into a top-level function with an explicit signature,
e.g. `run_calibration(states, cfg, client) -> cal`, `run_measurement_waves(...) -> None`
(mutates states), `run_fit(...) -> (fit_verdicts, choice_stats)`,
`run_presentation(...) -> (...)`. Promote the closures (`part_unit`, `homog_unit`) to
functions taking an explicit `UnitBuilder`/counter object instead of closing over `uid`.
This makes the resolution-wave logic (the trickiest part, 758–784) independently testable
and lets `recluster_cluster` reuse `run_calibration` rather than maintaining the parallel
`_mini_calibrate` (1293–1338).

### 2.3 Budget trimming can starve whole clusters

`guard` (646–651) trims an over-budget wave with `units[:room]` — a prefix cut. Because
waves are built by iterating `judged` in order, the clusters appended last get dropped
*entirely* when the budget bites, rather than every cluster losing a proportional share.
A run that hits `max_llm_calls` therefore silently produces zero data for an arbitrary
tail of clusters, which then report degenerate estimates.

**Suggestion.** Make trimming fair: interleave units round-robin by `cid` before slicing,
or allocate the remaining budget per-cluster. At minimum, record which clusters were
under-sampled due to the ceiling and surface it in `meta` so the report can mark them.

---

## 3. Correctness & robustness

### 3.1 Dead code in `recluster_cluster`

Line 1373–1374:

```python
cid_str = df[cluster_col if cluster_col in df.columns else "cluster_id"].astype(str) \
    if False else df["cluster_id"].astype(str)
```

The `... if False else ...` is a leftover; the first branch never runs. Replace with
`cid_str = df["cluster_id"].astype(str)`.

### 3.2 No global judge-health check

`LLMClient.complete` (268–287) retries then returns `{}`, which becomes a `parse_fail`.
That is correct *local* degradation, but there is no *global* guard: if the gateway is
broken and 90% of calls return `{}`, the run still completes and emits confident-looking
(garbage) numbers. The calibration gate catches *judge quality* but not *infrastructure
failure*.

**Suggestion.** Track the empty-result / parse-fail rate across the run and (a) emit a
loud `log.error` and (b) set an `meta["aborted_reason"]` / force the calibration gate to
FAIL when it exceeds a threshold (say 25%). Surface `calls_by_kind` failures in the
report header.

### 3.3 `load_inputs` JSON handling is fragile

`.json` → `pd.read_json(path)` (372–373) assumes a specific orientation and will raise or
silently mis-parse other shapes. Either document the expected schema in the error path or
sniff orientation.

### 3.4 Redundant recomputation in `_estimate_cluster`

Line 1178 calls `_mix_ci(st, cfg)` twice (for `[1]` and `[2]`), and line 1141 already
computed `m, lo, hi` from the same call. Reuse `lo`/`hi` instead of recomputing — three
calls collapse to one. Trivial, but it's on the hot path (called per cluster, per
resolution wave).

### 3.5 New `ThreadPoolExecutor` per wave

`run_units` (297) builds and tears down a pool on every call, and there are up to
`waves_max + intruder_waves_max + resolution_waves + fit + presentation` waves. Pool
churn is cheap relative to LLM latency, so this is minor — but a single long-lived
executor owned by the client (or passed in) would be cleaner and slightly faster for the
mock/CI path.

---

## 4. Maintainability

### 4.1 `Config` has ~50 fields with no validation

Many are interdependent: `waves_max` vs `intruder_waves_max`, `k_partition` vs
`min_judgeable`/`min_pool`, the five `n_*` calibration counts, `coverage_target` ≤
`coverage_ceiling`. Nothing checks consistency, so a typo (e.g. `coverage_ceiling <
coverage_target`) fails silently or oddly downstream.

**Suggestion.** Add `__post_init__` validation raising on incoherent combinations, and
group the dataclass into nested config objects (`SamplingConfig`, `CalibrationConfig`,
`BudgetConfig`) or at least keep the existing comment-section grouping as the documented
contract.

### 4.2 Untyped result dict

`evaluate` returns a free-form `dict`; consumers (`render_report`, the now-removed
dashboard, and any downstream code) rely on string keys with no schema. The README
documents the schema in prose, which will drift.

**Suggestion.** Define `TypedDict`s (`Results`, `ClusterResult`, `Calibration`, `Meta`)
in one place and annotate `evaluate -> Results`. This documents the contract, gives
editors/`mypy` traction, and makes the report renderer's field access checkable.

### 4.3 CLI duplicates the demo Config

The objection-domain demo `Config` is constructed twice with identical long strings
(1886–1889 and 1909–1913). Factor into a `demo_config(**overrides)` helper.

### 4.4 Single-file packaging

The single-file design is a *stated feature* (easy to drop behind a corporate gateway),
so I would **not** split the module by default. But it should still be installable: add a
`pyproject.toml` declaring deps (`numpy`, `pandas`, `scikit-learn`) and an optional
`[progress]` extra for `tqdm`, with `cluster-judge` as a console entry point. This keeps
the one-file ergonomics while making `pip install .` and CI setup trivial. If the file
keeps growing, the natural seam is `engine` / `io` / `report` / `cli` as a small package
that still re-exports the flat API.

---

## 5. Smaller notes

- **Demo texts leak labels.** `make_demo_data` emits `"C_GIANT :: theme0 item 5"`
  (1843) — cluster id and theme are literally in the text. Harmless because the demo only
  runs the mock (which scores off `_theme`, not text), but a one-line comment saying so
  would prevent someone wiring a real judge to the demo and getting flattering numbers.
- **`render_report` magic indices.** The `_tbl` right-justify sets like `rj={1, 3, 4, 5}`
  (1738) are positional and break silently if a column is inserted. Minor, but a small
  column-spec structure would be more robust.
- **Resolution-wave rationale is excellent — keep it.** The comment at 760–771 explaining
  why gating on the split flag would be circular is exactly the kind of load-bearing
  documentation to preserve; consider promoting it to a docstring on the extracted
  `run_resolution_waves`.
- **Checkpointing for long runs.** A 20k-call run that dies at call 19k loses everything.
  Optional: persist per-wave `res` to disk (keyed by unit hash) so a re-run resumes and
  identical prompts are cached — this also caps spend on retries.

---

## 6. Suggested priority order

1. **Add the mock-based regression test suite** (§2.1) — proves the correction layer and
   guards every future change. Highest value, lowest risk.
2. **Fix budget starvation** (§2.3) and **dead code** (§3.1) — small, correctness-affecting.
3. **Add global judge-health guard** (§3.2).
4. **Decompose `evaluate` into phase functions** (§2.2) — enables everything above to be
   tested in isolation; do it *after* the test net exists.
5. **Typed results + Config validation + `pyproject.toml`** (§4) — maintainability.
6. The smaller notes (§5) as you touch the surrounding code.
