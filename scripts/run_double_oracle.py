"""Compute bounds on the equilibrium host p* and value V* for a reduced Wordle instance.

Examples:
    python scripts/run_double_oracle.py --n 50 --seed 0
    python scripts/run_double_oracle.py --n 100 --out results/n100_s0.json
    python scripts/run_double_oracle.py --words light,might,sight,night,right,crane,slate
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wordle import Game, double_oracle  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=50, help="random subset size of the answer list")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--words", type=str, default=None, help="comma-separated explicit word list")
ap.add_argument("--guesses", choices=["all", "answers"], default="all",
                help="all = any of the 12,972 allowed words (default, as in Selby / Bertsimas); "
                     "answers = only subset words (fast, for debugging)")
ap.add_argument("--no-host-oracle", dest="host_oracle", action="store_false",
                help="put every word in the LP from the start instead of growing the host's "
                     "word set with a best-response oracle (slower at n=100: 64 vs 29 min)")
ap.add_argument("--init-words", type=int, default=10)
ap.add_argument("--eps", type=float, default=1e-6)
ap.add_argument("--max-iters", type=int, default=2000)
ap.add_argument("--out", type=str, default=None, help="write results JSON here")
ap.add_argument("--quiet", action="store_true")
args = ap.parse_args()

words = args.words.split(",") if args.words else None
game = Game.subset(n=args.n, seed=args.seed, words=words, guesses=args.guesses)
res = double_oracle(game, eps=args.eps, max_iters=args.max_iters, host_oracle=args.host_oracle,
                    init_words=args.init_words, verbose=not args.quiet)
print()
print(res.summary(game))

if args.out:
    h = res.history[-1]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(dict(
        args=vars(args), n=game.n, n_guesses=len(game.guesses), words=game.answers,
        v_uniform=res.v_uniform, lower=res.lower, upper=res.upper, gap=res.gap,
        converged=bool(res.gap <= args.eps), iterations=len(res.history),
        time_s=h["time"], n_trees=len(res.trees), words_in_lp=h["words"],
        max_depth_equilibrium=res.max_depth_equilibrium, max_depth_any=res.max_depth_any,
        p={w: float(x) for w, x in sorted(zip(game.answers, res.p), key=lambda t: -t[1])
           if x > 1e-12},
        q=[dict(weight=float(x), opener=game.guesses[t.guess], cost=c.astype(int).tolist())
           for x, t, c in zip(res.q, res.trees, res.costs) if x > 1e-12],
        history=res.history), indent=1))
    print(f"\nwrote {args.out}")
