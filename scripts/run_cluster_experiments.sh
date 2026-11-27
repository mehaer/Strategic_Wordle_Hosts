#!/usr/bin/env bash
# Run the double oracle on each 50-word cluster list, then summarise.
#   bash scripts/run_cluster_experiments.sh            # all four clusters
#   bash scripts/run_cluster_experiments.sh ight ound  # just these
# Finished clusters (result file already present) are skipped, so it is safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
if [ $# -gt 0 ]; then CLUSTERS=("$@"); else CLUSTERS=(ight ound atch ower); fi
mkdir -p results/cluster
for c in "${CLUSTERS[@]}"; do
  out=results/cluster/n50_$c.json
  if [ -f "$out" ]; then echo "== $c: $out exists, skipping"; continue; fi
  echo "== $c: started $(date +%H:%M)"
  W=$($PY scripts/make_cluster_list.py --cluster "$c")
  $PY scripts/run_double_oracle.py --words "$W" --quiet --out "$out" | tee "results/cluster/n50_$c.log"
done
$PY scripts/cluster_report.py results/cluster
