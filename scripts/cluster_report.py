"""Summarise cluster-experiment results: how much weight p* puts on the look-alike cluster.

    python scripts/cluster_report.py                # reads results/cluster/*.json
    python scripts/cluster_report.py results/cluster

Writes results/cluster/cluster_summary.csv and prints the same table.
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_cluster_list import CLUSTERS  # noqa: E402

root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/cluster")
rows = []
for f in sorted(root.glob("n50_*.json")):
    name = f.stem.split("_", 1)[1]
    if name not in CLUSTERS:
        continue
    r = json.loads(f.read_text())
    cl = CLUSTERS[name]
    mass = sum(r["p"].get(w, 0.0) for w in cl)
    rows.append(dict(
        cluster=name, n=r["n"], cluster_size=len(cl),
        uniform_share=round(len(cl) / r["n"], 4), p_star_mass=round(mass, 4),
        v_uniform=round(r["v_uniform"], 6), v_star=round(r["lower"], 6),
        gain=round(r["lower"] - r["v_uniform"], 6), converged=r["converged"],
        max_depth_eq=r["max_depth_equilibrium"], time_s=round(r["time_s"], 1),
        top_words=" ".join(list(r["p"])[:5]),
    ))
if not rows:
    sys.exit(f"no cluster result files in {root}")
with open(root / "cluster_summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
cols = list(rows[0])
widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
print("  ".join(c.ljust(widths[c]) for c in cols))
for r in rows:
    print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))
print(f"\nwrote {root / 'cluster_summary.csv'}")
