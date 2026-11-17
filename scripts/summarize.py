"""Collect results/*.json into one table (printed, and written as results/summary.csv).

    python scripts/summarize.py            # reads results/*.json
    python scripts/summarize.py results/main
"""
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
rows = []
for f in sorted(root.rglob("*.json")):
    r = json.loads(f.read_text())
    if "lower" not in r:
        continue
    rows.append(dict(
        file=str(f.relative_to(root)), n=r["n"], seed=r["args"]["seed"],
        guesses=r["args"]["guesses"], host_oracle=r["args"]["host_oracle"],
        v_uniform=round(r["v_uniform"], 6), v_star_lower=round(r["lower"], 6),
        v_star_upper=round(r["upper"], 6), gain=round(r["lower"] - r["v_uniform"], 6),
        converged=r["converged"], iterations=r["iterations"], time_s=round(r["time_s"], 1),
        supp_p=len(r["p"]), supp_q=len(r["q"]),
        max_depth_eq=r["max_depth_equilibrium"], max_depth_any=r["max_depth_any"],
        top_words=" ".join(list(r["p"])[:5]),
    ))
rows.sort(key=lambda d: (d["guesses"], d["n"], d["seed"], d["host_oracle"]))
if not rows:
    sys.exit(f"no result files in {root}")
cols = list(rows[0])
with open(root / "summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)
show = ["file", "n", "seed", "host_oracle", "v_uniform", "v_star_lower", "v_star_upper", "gain",
        "converged", "iterations", "time_s", "supp_p", "max_depth_any"]
widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in show}
print("  ".join(c.ljust(widths[c]) for c in show))
for r in rows:
    print("  ".join(str(r[c]).ljust(widths[c]) for c in show))
print(f"\nwrote {root / 'summary.csv'}")
