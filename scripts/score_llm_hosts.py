"""Score the LLM hosts logged by run_llm_hosts.py against the equilibrium host.

    python scripts/score_llm_hosts.py                    # reads results/llm/raw/*.jsonl
    python scripts/score_llm_hosts.py --out results/llm

Each host distribution p is scored with the exact best-response guesser:
    V_LLM = min over guesser trees of E_p[guesses]
    gain  = V_LLM - V_unif                       (extra guesses forced, vs the uniform host)
    score = (V_LLM - V_unif) / (V* - V_unif)     0 = uniform host, 1 = equilibrium host
    tv    = total variation distance between p and p*  (0 = identical, 1 = disjoint)
    cluster_mass = share of p on the look-alike cluster (cluster lists only)

explicit (the headline metric): every valid reply is a distribution the model commits to, so each
    reply is scored on its own and the scores are averaged. `score_pooled` scores the average of
    the replies instead (a host that randomises over its own answers).
sample: p = frequencies of the words the model actually picked. Use it for cluster_mass and
    top_words (unbiased at any sample size), NOT for score: with few samples p is lumpy and the
    guesser exploits that, so even samples drawn from p* score far below 1. The "ref: ... sampled"
    rows show this: they draw the same number of samples from p* and from the uniform distribution.

Writes <out>/llm_scores.csv (one row per model, list, condition, method) and
<out>/llm_summary.csv (averaged over lists), and prints the summary.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from make_cluster_list import CLUSTERS  # noqa: E402
from run_llm_hosts import list_name, read_jsonl  # noqa: E402
from wordle import Game, best_response, load_words  # noqa: E402

REF_SEEDS = 5


def load_list_info(cfg: dict) -> dict[str, dict]:
    allowed = load_words("allowed")
    info = {}
    for f in cfg["word_lists"]:
        r = json.loads((ROOT / f).read_text())
        name = list_name(f)
        words = r["words"]
        cl = CLUSTERS.get(name.split("_", 1)[1]) if name.startswith("cluster/") else None
        info[name] = dict(
            words=words, game=Game(words, allowed), v_star=r["lower"], v_unif=r["v_uniform"],
            p_star=np.array([r["p"].get(w, 0.0) for w in words]),
            cluster=set(cl) if cl else None,
        )
    return info


def evaluate(li: dict, p: np.ndarray) -> dict:
    p = p / p.sum()
    v = best_response(li["game"], p).value
    out = dict(v_llm=v, gain=v - li["v_unif"], score=(v - li["v_unif"]) / (li["v_star"] - li["v_unif"]),
               tv=0.5 * float(np.abs(p - li["p_star"]).sum()), support=int((p > 0).sum()))
    if li["cluster"]:
        mask = np.array([w in li["cluster"] for w in li["words"]])
        out.update(cluster_mass=float(p[mask].sum()), p_star_cluster_mass=float(li["p_star"][mask].sum()),
                   uniform_share=float(mask.mean()))
    return out


def to_vec(words: list[str], d: dict[str, float]) -> np.ndarray:
    return np.array([d.get(w, 0.0) for w in words], dtype=float)


def score_records(model: str, recs: list[dict], info: dict) -> list[dict]:
    rows = []
    groups = defaultdict(list)
    for r in recs:
        if r.get("status") != "api_error" and r["list"] in info:
            groups[(r["list"], r["condition"], r["method"])].append(r)
    for (lname, cond, method), rs in sorted(groups.items()):
        li = info[lname]
        ok = [r for r in rs if r.get("status") == "ok"]
        row = dict(model=model, list=lname, condition=cond, method=method, n_calls=len(rs),
                   n_ok=len(ok), cost=round(sum(r.get("cost", 0.0) for r in rs), 5),
                   v_unif=li["v_unif"], v_star=li["v_star"])
        if not ok:
            rows.append(row)
            continue
        if method == "sample":
            counts = Counter(r["word"] for r in ok)
            row.update(evaluate(li, to_vec(li["words"], counts)))
            row["top_words"] = " ".join(w for w, _ in counts.most_common(5))
        else:
            dists = [to_vec(li["words"], r["p"]) for r in ok]
            evals = [evaluate(li, d) for d in dists]
            for k in evals[0]:
                row[k] = float(np.mean([e[k] for e in evals]))
            pooled = np.mean(dists, axis=0)
            row["score_pooled"] = evaluate(li, pooled)["score"]
            row["off_list_mass_mean"] = float(np.mean([r.get("off_list_mass", 0.0) for r in ok]))
            top = np.argsort(-pooled)[:5]
            row["top_words"] = " ".join(li["words"][i] for i in top if pooled[i] > 0)
        rows.append(row)
    return rows


def reference_rows(info: dict, n_samples: int) -> list[dict]:
    rows = []
    for lname, li in info.items():
        n = len(li["words"])
        for ref, base in (("ref: p* sampled", li["p_star"]), ("ref: uniform sampled", np.full(n, 1 / n))):
            evals = []
            for s in range(REF_SEEDS):
                draws = np.random.default_rng(s).choice(n, size=n_samples, p=base / base.sum())
                evals.append(evaluate(li, np.bincount(draws, minlength=n).astype(float)))
            row = dict(model=ref, list=lname, condition="-", method="sample", n_calls=n_samples,
                       n_ok=n_samples, cost=0.0, v_unif=li["v_unif"], v_star=li["v_star"])
            for k in evals[0]:
                row[k] = float(np.mean([e[k] for e in evals]))
            rows.append(row)
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        if "score" in r:
            groups[(r["model"], r["condition"], r["method"])].append(r)
    out = []
    for (model, cond, method), rs in groups.items():
        cl = [r for r in rs if "cluster_mass" in r]
        out.append(dict(
            model=model, condition=cond, method=method, lists=len(rs),
            gain_mean=round(float(np.mean([r["gain"] for r in rs])), 4),
            score_mean=round(float(np.mean([r["score"] for r in rs])), 3),
            score_random_lists=round(float(np.mean([r["score"] for r in rs if r["list"].startswith("main/")] or [np.nan])), 3),
            score_cluster_lists=round(float(np.mean([r["score"] for r in cl] or [np.nan])), 3),
            tv_mean=round(float(np.mean([r["tv"] for r in rs])), 3),
            cluster_mass=round(float(np.mean([r["cluster_mass"] for r in cl] or [np.nan])), 3),
            p_star_cluster_mass=round(float(np.mean([r["p_star_cluster_mass"] for r in cl] or [np.nan])), 3),
            uniform_share=round(float(np.mean([r["uniform_share"] for r in cl] or [np.nan])), 3),
            support_mean=round(float(np.mean([r["support"] for r in rs])), 1),
            parse_ok_rate=round(sum(r["n_ok"] for r in rs) / max(sum(r["n_calls"] for r in rs), 1), 3),
            cost=round(sum(r["cost"] for r in rs), 4),
        ))
    out.sort(key=lambda d: (d["model"].startswith("ref"), d["model"], d["condition"], d["method"]))
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    cols = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "llm_hosts.json"))
    ap.add_argument("--out", default=str(ROOT / "results" / "llm"))
    args = ap.parse_args(argv)

    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.out)
    files = sorted((out / "raw").glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no raw results in {out / 'raw'}")
    info = load_list_info(cfg)

    rows = []
    for f in files:
        recs = read_jsonl(f)
        if recs:
            rows += score_records(recs[0]["model"], recs, info)
    rows += reference_rows(info, cfg["samples"])
    write_csv(out / "llm_scores.csv", rows)
    summary = summarize(rows)
    write_csv(out / "llm_summary.csv", summary)

    show = ["model", "condition", "method", "lists", "gain_mean", "score_mean", "score_random_lists",
            "score_cluster_lists", "tv_mean", "cluster_mass", "p_star_cluster_mass",
            "uniform_share", "support_mean", "parse_ok_rate", "cost"]
    widths = {c: max(len(c), *(len(str(r[c])) for r in summary)) for c in show}
    print("  ".join(c.ljust(widths[c]) for c in show))
    for r in summary:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in show))
    print(f"\nwrote {out / 'llm_scores.csv'} and {out / 'llm_summary.csv'}")


if __name__ == "__main__":
    main()
