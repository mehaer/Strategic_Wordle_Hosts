"""Equilibrium host via double oracle.

Restricted game: host words W' (subset of all words) vs guesser trees T'.
Each iteration:

    1. Solve the restricted LP
           max v  s.t.  sum_{w in W'} p_w c(T, w) >= v  for every T in T',  p in simplex(W')
       -> host p (zero outside W'), guesser mix q over T' (LP duals), restricted value v.
    2. Guesser oracle: exact best-response tree T_new to p.
           L = E_p[c(T_new)] = V(p) <= V*      (p is a real host strategy)
    3. Host oracle: hardest word against q.
           U = max_w sum_T q_T c(T, w) >= V*   (q is a real guesser strategy)
    4. Add T_new if it beats v, add the hardest word if it beats v.
       Stop when neither oracle improves, i.e. U - L <= eps.

So L <= V* <= U at every iteration, which gives valid bounds even when stopped early.

host_oracle=False puts every word in W' from the start (the host oracle is then
trivial and only trees are generated); host_oracle=True starts from a few words and
grows W', which keeps the LP small on large word lists. Both converge to the same V*.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog

from .solver import Game, Node, best_response


def solve_host_lp(C: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """C: (k trees, n words) cost matrix. Returns host p, value v, guesser mix q over trees."""
    k, n = C.shape
    c = np.zeros(n + 1)
    c[-1] = -1.0                                         # maximise v
    A_ub = np.hstack([-C, np.ones((k, 1))])              # v - C p <= 0
    A_eq = np.hstack([np.ones((1, n)), np.zeros((1, 1))])
    bounds = [(0, None)] * n + [(None, None)]
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(k), A_eq=A_eq, b_eq=[1.0],
                  bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(f"LP failed: {res.message}")
    p = np.clip(res.x[:n], 0, None)
    p /= p.sum()
    q = np.clip(-res.ineqlin.marginals, 0, None)
    q = q / q.sum() if q.sum() > 0 else np.full(k, 1 / k)
    return p, float(res.x[-1]), q


@dataclass
class DOResult:
    p: np.ndarray              # host strategy with the best guaranteed value (achieves `lower`)
    q: np.ndarray              # guesser mix over `trees` with the best guarantee (achieves `upper`)
    lower: float               # V(p) <= V*
    upper: float               # V* <= upper
    trees: list[Node]
    costs: np.ndarray          # (k, n) cost matrix c(T, w) over all words
    v_uniform: float           # optimal guesser vs uniform host
    host_oracle: bool
    history: list[dict] = field(default_factory=list)

    @property
    def gap(self) -> float:
        return self.upper - self.lower

    @property
    def max_depth_equilibrium(self) -> int:
        """Deepest guess count used by any tree the equilibrium guesser plays."""
        return int(self.costs[self.q > 1e-9].max())

    @property
    def max_depth_any(self) -> int:
        """Deepest guess count in any tree generated during the run."""
        return int(self.costs.max())

    def summary(self, game: Game, top: int = 15) -> str:
        h = self.history[-1]
        lines = [
            f"n = {game.n} words, {len(game.guesses)} allowed guesses, "
            f"host oracle {'on' if self.host_oracle else 'off'}",
            f"V(uniform)          = {self.v_uniform:.6f}",
            f"V* in [{self.lower:.6f}, {self.upper:.6f}]  (gap {self.gap:.2e})",
            f"V* - V(uniform)     = {self.lower - self.v_uniform:.6f}",
            f"iterations          = {len(self.history)}, trees = {len(self.trees)}, "
            f"words in LP = {h['words']}, time = {h['time']:.1f}s",
            f"support of p*       = {(self.p > 1e-9).sum()} words",
            f"support of q*       = {(self.q > 1e-9).sum()} trees",
            f"max depth           = {self.max_depth_equilibrium} (equilibrium trees), "
            f"{self.max_depth_any} (any generated tree)"
            + ("" if self.max_depth_any <= 6 else "  ** exceeds the 6-guess limit **"),
            "",
            "top host words (p*):",
        ]
        for w in np.argsort(-self.p)[:top]:
            if self.p[w] <= 1e-9:
                break
            lines.append(f"  {game.answers[w].upper()}  {self.p[w]:.4f}")
        lines.append("")
        lines.append("guesser mix (q*), top trees by weight:")
        for t in np.argsort(-self.q)[:top]:
            if self.q[t] <= 1e-9:
                break
            lines.append(f"  tree {t:3d}  opens {game.guesses[self.trees[t].guess].upper()}"
                         f"  q = {self.q[t]:.4f}")
        return "\n".join(lines)


def double_oracle(game: Game, eps: float = 1e-6, max_iters: int = 2000,
                  host_oracle: bool = False, init_words: int = 10,
                  verbose: bool = True) -> DOResult:
    """Compute (bounds on) the equilibrium host distribution p* and value V*.

    The tree set starts with the best response to the uniform host. With
    host_oracle=True the word set starts with the `init_words` words that tree
    finds hardest, and the host's best-response word is added each iteration.
    """
    n = game.n
    t0 = time.time()
    tol = 1e-9

    uni = best_response(game, np.full(n, 1 / n))
    trees, costs = [uni.tree], [uni.cost]
    if host_oracle:
        words = [int(w) for w in np.argsort(-uni.cost, kind="stable")[: min(init_words, n)]]
    else:
        words = list(range(n))

    best_lower, best_p = uni.value, np.full(n, 1 / n)
    best_upper, best_q = np.inf, None
    history: list[dict] = []
    for it in range(1, max_iters + 1):
        C = np.array(costs, dtype=float)                  # (k, n), all words
        p_r, v, q = solve_host_lp(C[:, words])
        p = np.zeros(n)
        p[words] = p_r

        # guesser oracle: best tree against p -> lower bound
        br = best_response(game, p)
        if br.value > best_lower:
            best_lower, best_p = br.value, p
        # host oracle: hardest word against q -> upper bound
        word_vals = q @ C
        hard = int(np.argmax(word_vals))
        upper = float(word_vals[hard])
        if upper < best_upper:
            best_upper, best_q = upper, q

        history.append(dict(iter=it, lp_value=v, upper=upper, lower=br.value,
                            best_lower=best_lower, best_upper=best_upper,
                            support=int((p > tol).sum()), trees=len(trees), words=len(words),
                            br_nodes=br.nodes_expanded, time=time.time() - t0))
        if verbose:
            print(f"[{it:4d}] U={best_upper:.6f}  L={best_lower:.6f}  "
                  f"gap={best_upper - best_lower:.2e}  |supp p|={history[-1]['support']}  "
                  f"trees={len(trees)}  words={len(words)}  ({history[-1]['time']:.1f}s)",
                  flush=True)
        if best_upper - best_lower <= eps:
            break
        added = False
        if br.value < v - tol:                            # guesser can improve
            trees.append(br.tree)
            costs.append(br.cost)
            added = True
        if upper > v + tol and hard not in words:         # host can improve
            words.append(hard)
            added = True
        if not added:                                     # both oracles agree: converged
            break

    k = len(best_q)
    q_full = np.zeros(len(trees))
    q_full[:k] = best_q
    return DOResult(p=best_p, q=q_full, lower=best_lower, upper=best_upper, trees=trees,
                    costs=np.array(costs, dtype=float), v_uniform=uni.value,
                    host_oracle=host_oracle, history=history)
