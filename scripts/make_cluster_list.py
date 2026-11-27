"""Build a word list: a 50-word base list with some words swapped out for a look-alike cluster.

    python scripts/make_cluster_list.py --cluster ight
"""
import argparse
import json
import random

CLUSTERS = {
    "ight": ["eight", "fight", "light", "might", "night", "right", "sight", "tight"],
    "atch": ["batch", "catch", "hatch", "latch", "match", "patch", "watch"],
    "ound": ["bound", "found", "hound", "mound", "pound", "round", "sound", "wound"],
    "ower": ["cower", "lower", "mower", "power", "rower", "sower", "tower"],
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster", choices=CLUSTERS, required=True)
    ap.add_argument("--base", default="results/main/n50_s0.json", help="result file whose word list is the base")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cluster = CLUSTERS[args.cluster]
    base = [w for w in json.load(open(args.base))["words"] if w not in cluster]
    keep = random.Random(args.seed).sample(base, 50 - len(cluster))   # drop random base words to stay at 50
    print(",".join(sorted(keep + cluster)))


if __name__ == "__main__":
    main()
