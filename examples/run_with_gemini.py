"""Run cluster_judge against a real LLM judge (Google Gemini).

This is a smoke test on a small, *realistic* dataset — real customer-objection
sentences a model can genuinely cluster by meaning (unlike the synthetic demo,
whose text is contentless and label-leaking). Embeddings are synthetic but
category-correlated, which is all the sampler needs; the judge works off text.

Usage:
    GEMINI_API_KEY=... python examples/run_with_gemini.py

Requires: numpy, pandas, scikit-learn (cluster_judge deps). No Google SDK — the
gateway is a ~30-line urllib call so there's nothing extra to install.
"""
import json
import os
import threading
import time
import urllib.error
import urllib.request

import numpy as np

import cluster_judge as cj

# gemini-flash-latest points at gemini-3.5-flash, whose free tier is only 20 req/day.
# Default to a lite model with a far larger free quota; override with GEMINI_MODEL.
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")
ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
API_KEY = os.environ.get("GEMINI_API_KEY")

# Client-side throttle so we stay under the free-tier per-minute cap (~15 RPM).
_MIN_INTERVAL = 4.5          # seconds between request starts (~13 RPM)
_rate_lock = threading.Lock()
_last_call = [0.0]


def _throttle():
    with _rate_lock:
        wait = _MIN_INTERVAL - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()


def run_model_messages(messages, json_mode: bool = True) -> str:
    """The gateway cluster_judge calls. Translates its [{system},{user}] messages into a
    Gemini request and returns the model's text (a JSON string). Thinking is disabled for
    latency; JSON mode is requested so the reply is a bare object."""
    system = next((m["content"] for m in messages if m["role"] == "system"), None)
    user = next(m["content"] for m in messages if m["role"] == "user")
    body = {
        "contents": [{"parts": [{"text": user}]}],
        "generationConfig": {"thinkingConfig": {"thinkingBudget": 0}},
    }
    if system:
        body["system_instruction"] = {"parts": [{"text": system}]}
    if json_mode:
        body["generationConfig"]["responseMimeType"] = "application/json"

    data = json.dumps(body).encode()
    delay = 5.0
    for attempt in range(5):
        _throttle()
        req = urllib.request.Request(
            ENDPOINT, data=data,
            headers={"Content-Type": "application/json", "X-goog-api-key": API_KEY},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                out = json.loads(resp.read())
            cands = out.get("candidates", [])
            if not cands:
                return "{}"
            parts = cands[0].get("content", {}).get("parts", [])
            return "".join(p["text"] for p in parts if "text" in p) or "{}"
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 503) and attempt < 4:   # rate limit / transient
                time.sleep(delay)
                delay *= 2
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            if attempt < 4:
                time.sleep(delay)
                delay *= 2
                continue
            raise
    return "{}"


# --------------------------------------------------------------------------- #
# Small realistic dataset: 5 clusters of sales-call objections. Four are pure
# (one concern each); "PRICE_OR_TIMING" deliberately mixes two concerns so we can
# see whether a real judge flags it as should-split.
# --------------------------------------------------------------------------- #
CLUSTERS = {
    "PRICE": [
        "It's just too expensive for us right now.",
        "The price is way over our budget.",
        "We can't justify spending that much.",
        "Honestly your pricing is higher than competitors.",
        "There's no room in the budget for this.",
        "The cost is hard to swallow.",
        "We'd need a serious discount to consider it.",
    ],
    "AUTHORITY": [
        "I'll have to run this by my manager first.",
        "This isn't really my decision to make.",
        "I need sign-off from leadership before moving.",
        "Let me check with the rest of the team.",
        "My boss handles purchases like this.",
        "I can't approve something this size on my own.",
        "We'd need the committee to weigh in.",
    ],
    "NEED": [
        "I'm not sure we actually need this.",
        "We already have a solution that works fine.",
        "I don't really see the value for us.",
        "This doesn't solve a problem we have.",
        "Our current process is good enough.",
        "Not convinced this would change anything for us.",
        "We get by without it today.",
    ],
    "TRUST": [
        "I've honestly never heard of your company.",
        "How do I know this actually works?",
        "I'd want to see references or case studies first.",
        "Worried about whether you'll be reliable.",
        "Can you prove the results you're claiming?",
        "Seems risky to bet on a vendor I don't know.",
        "I'd need proof before I trust this.",
    ],
    "PRICE_OR_TIMING": [
        "It's too pricey for what we have budgeted.",      # price
        "We just can't afford it this year.",              # price
        "The cost is more than we planned to spend.",      # price
        "Now really isn't a good time for us.",            # timing
        "Maybe revisit this next quarter.",                # timing
        "We're not ready to make a decision yet.",         # timing
        "Circle back with me in a few months.",            # timing
    ],
}

# Themes for synthetic embeddings (PRICE_OR_TIMING items carry their true sub-theme).
THEME_OF = {
    "PRICE": ["price"] * 7,
    "AUTHORITY": ["authority"] * 7,
    "NEED": ["need"] * 7,
    "TRUST": ["trust"] * 7,
    "PRICE_OR_TIMING": ["price", "price", "price", "timing", "timing", "timing", "timing"],
}


def build_dataset(dim=48, seed=7):
    rng = np.random.default_rng(seed)
    themes = ["price", "timing", "authority", "need", "trust"]
    centers = {t: rng.normal(0, 1, dim) for t in themes}
    for t in centers:
        centers[t] /= np.linalg.norm(centers[t])
    rows, embs = [], []
    for cid, texts in CLUSTERS.items():
        for text, theme in zip(texts, THEME_OF[cid]):
            rows.append({"text": text, "cluster_id": cid, "label": cid.replace("_", " ").title(),
                         "_theme": theme})
            embs.append(centers[theme] + rng.normal(0, 0.05, dim))
    df = cj.pd.DataFrame(rows)
    return df, np.asarray(embs, dtype=np.float32)


def main():
    if not API_KEY:
        raise SystemExit("set GEMINI_API_KEY in the environment")
    cj.configure_logging()
    cj.use_genai(run_model_messages)

    df, emb = build_dataset()
    labels = {cid: {"label": cid.replace("_", " ").title(), "description": ""}
              for cid in df["cluster_id"].unique()}

    # Small config — each judge call is a real network round-trip, so keep counts low.
    cfg = cj.Config(
        same_when="are the same kind of sales objection (the underlying concern), "
                  "regardless of wording",
        use_context="each cluster is mined for rebuttal strategies to that objection type",
        unit="each text is a customer objection raised on a sales call",
        model="gemini", workers=2,
        k_partition=6, coverage_target=0.9, coverage_ceiling=1.0, min_pool=5, min_judgeable=4,
        n_pure=2, n_mixed=2, n_junk=1, n_far=2, n_labelswap=1,
        waves_max=1, intruder_waves_max=1, draws_per_wave=1, intruder_per_wave=1, neighbor_m=2,
        dup_rate=0.0,
        resolution_waves=0, fit_items=6, fit_replicates=1, do_fit_choice=False,
        do_strategy=False, strategy_n=6, do_taxonomy=False,
    )

    t0 = time.time()
    results = cj.evaluate(df, emb, labels, cfg, progress=True)
    print(f"\n[ran {results['meta']['n_llm_calls']} real Gemini calls in {time.time() - t0:.0f}s]\n")
    print(cj.render_report(results))
    cj.write_report(results, "cj_gemini_out")


if __name__ == "__main__":
    main()
