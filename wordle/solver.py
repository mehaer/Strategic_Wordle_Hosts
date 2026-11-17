"""Exact guesser best response against a host distribution p.

Payoff: c(T, w) = number of guesses decision tree T needs to find word w
(the guess that turns all-green counts). The guesser minimises sum_w p(w) c(T, w).

DP over candidate sets S, using unnormalised mass W(S) = sum_{w in S} p(w):

    E(S) = W(S) + min_g  sum_{patterns pi != ALL_GREEN}  E(S_pi)

Every remaining word pays for the current guess (the W(S) term); the word equal
to g (if any) lands in the all-green bucket and costs nothing further.

Lower bound used for pruning: for any bucket B, E(B) >= 2 W(B) - max_{w in B} p(w)
(the next guess can finish at most one word; everyone else needs >= 2 more).
The bound is admissible, so the result is an exact best response over the
given guess set (no 6-guess cap is imposed).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .feedback import ALL_GREEN, decode, pattern_matrix, load_words


# --------------------------------------------------------------------------- game

class Game:
    """A (possibly reduced) Wordle instance: a host word list and a guess list.

    P[g, a] is the feedback when guess g is played against answer a.
    Every answer is always included in the guess list.
    """

    def __init__(self, answers: list[str], guesses: list[str] | None = None):
        self.answers = list(answers)
        guesses = list(self.answers if guesses is None else guesses)
        seen = set(guesses)
        guesses += [w for w in self.answers if w not in seen]
        self.guesses = guesses
        self.P = pattern_matrix(self.guesses, self.answers)
        row = {w: i for i, w in enumerate(self.guesses)}
        self.ans_row = np.array([row[w] for w in self.answers])

    @property
    def n(self) -> int:
        return len(self.answers)

    @classmethod
    def subset(cls, n: int | None = None, seed: int = 0, words: list[str] | None = None,
               guesses: str = "all") -> "Game":
        """Random n-word subset of the official answers (or an explicit word list).

        guesses = "answers": guesser may only guess words from the subset (fast).
        guesses = "all":     guesser may use all 12,972 allowed words (stronger guesser).
        """
        if words is None:
            pool = load_words("answers")
            rng = np.random.default_rng(seed)
            words = sorted(rng.choice(pool, size=n, replace=False).tolist()) if n else pool
        g = load_words("allowed") if guesses == "all" else None
        return cls(words, g)


# --------------------------------------------------------------------------- trees

@dataclass
class Node:
    guess: int                                   # row index into game.guesses
    children: dict[int, "Node"] = field(default_factory=dict)   # pattern -> subtree

    def pretty(self, game: Game, depth: int = 0, max_depth: int = 2) -> str:
        pad = "  " * depth
        lines = [f"{pad}{game.guesses[self.guess].upper()}"]
        if depth < max_depth:
            for pat, ch in sorted(self.children.items()):
                lines.append(f"{pad}  [{decode(pat)}]")
                lines.append(ch.pretty(game, depth + 2, max_depth))
        return "\n".join(lines)


def cost_vector(game: Game, tree: Node) -> np.ndarray:
    """c[w] = guesses tree needs to solve answer w."""
    c = np.zeros(game.n, dtype=np.int64)
    for w in range(game.n):
        node, d = tree, 1
        while True:
            pat = int(game.P[node.guess, w])
            if pat == ALL_GREEN:
                break
            node = node.children[pat]     # KeyError => tree does not cover w (a bug)
            d += 1
        c[w] = d
    return c


# --------------------------------------------------------------------------- DP

class _DP:
    """Memoised exact DP for one weight vector p. Sets are tuples of answer indices
    in a fixed global order (descending p), so a set's first element is its max."""

    def __init__(self, game: Game, p: np.ndarray):
        self.game, self.p = game, np.asarray(p, dtype=float)
        self.memo: dict[tuple[int, ...], tuple[float, Node]] = {}
        self.order = np.lexsort((np.arange(game.n), -self.p))   # desc p, ties by index
        self.nodes = 0

    def solve(self, S: tuple[int, ...]) -> tuple[float, Node]:
        hit = self.memo.get(S)
        if hit is not None:
            return hit
        self.nodes += 1
        g, p = self.game, self.p
        if len(S) == 1:
            res = (p[S[0]], Node(int(g.ans_row[S[0]])))
        elif len(S) == 2:
            a, b = S                          # p[a] >= p[b]: guess a first
            ga = int(g.ans_row[a])
            res = (p[a] + 2 * p[b], Node(ga, {int(g.P[ga, b]): Node(int(g.ans_row[b]))}))
        else:
            res = self._solve_general(S)
        self.memo[S] = res
        return res

    def _solve_general(self, S: tuple[int, ...]) -> tuple[float, Node]:
        game, p = self.game, self.p
        idx = np.fromiter(S, dtype=np.int64, count=len(S))
        pS = p[idx]
        W = float(pS.sum())
        M = game.P[:, idx]                                     # (G, n)

        # Per-guess lower bound: sort each row's patterns stably; the first element of
        # each group has the smallest position in S, i.e. the largest p in that bucket.
        order = np.argsort(M, axis=1, kind="stable")
        Ms = np.take_along_axis(M, order, axis=1)
        first = np.ones(M.shape, dtype=bool)
        first[:, 1:] = Ms[:, 1:] != Ms[:, :-1]
        n_buckets = first.sum(1)
        sum_max = (pS[order] * first).sum(1)                   # sum of bucket maxima (incl. green)
        p_green = ((M == ALL_GREEN) * pS).sum(1)
        lb = W + 2 * W - p_green - sum_max                     # W + sum_B (2 W_B - max_B)

        cand = np.nonzero(n_buckets > 1)[0]                    # non-splitting guesses are useless
        cand = cand[np.argsort(lb[cand], kind="stable")]

        best_val, best_node = np.inf, None
        for gi in cand:
            if lb[gi] >= best_val - 1e-12:
                break
            row = M[gi]
            buckets: dict[int, list[int]] = {}
            for w, pat in zip(S, row.tolist()):
                if pat != ALL_GREEN:
                    buckets.setdefault(pat, []).append(w)
            child_lb = {pat: 2 * p[ws].sum() - p[ws[0]] for pat, ws in buckets.items()}
            remaining_lb = sum(child_lb.values())
            total = W
            children: dict[int, Node] = {}
            # solve heaviest buckets first: they are most likely to trigger the cutoff
            for pat in sorted(buckets, key=lambda k: -child_lb[k]):
                remaining_lb -= child_lb[pat]
                v, node = self.solve(tuple(buckets[pat]))
                total += v
                children[pat] = node
                if total + remaining_lb >= best_val - 1e-12:
                    break
            else:
                if total < best_val - 1e-12:
                    best_val, best_node = total, Node(int(gi), children)
        return best_val, best_node


# --------------------------------------------------------------------------- public API

@dataclass
class BestResponse:
    value: float          # sum_w p(w) c(T, w) for the returned tree: = min_T E_p[c] (exact)
    cost: np.ndarray      # c(T, w) for every answer w, including p(w) = 0 words
    tree: Node
    nodes_expanded: int


def best_response(game: Game, p, tol: float = 1e-12) -> BestResponse:
    """Exact best-response decision tree for the guesser against host distribution p.

    The DP runs only over words with p(w) > tol. Zero-probability words are then
    grafted onto the tree (solved by a uniform-weight DP inside the buckets they fall
    into) so the tree has a defined cost for every word, which the LP needs.
    Grafting never changes the path of a positive-probability word, so optimality
    with respect to p is preserved.
    """
    p = np.clip(np.asarray(p, dtype=float), 0, None)
    p = p / p.sum()
    dp = _DP(game, p)
    support = tuple(int(w) for w in dp.order if p[w] > tol)
    zero = [int(w) for w in range(game.n) if p[w] <= tol]
    value, tree = dp.solve(support)
    nodes = dp.nodes
    if zero:
        filler = _DP(game, np.ones(game.n))
        tree = _graft(game, tree, zero, filler)
        nodes += filler.nodes
    cost = cost_vector(game, tree)
    assert abs(float(p @ cost) - value) < 1e-9, "grafting changed the value: bug"
    return BestResponse(float(p @ cost), cost, tree, nodes)


def _graft(game: Game, node: Node, words: list[int], filler: _DP) -> Node:
    """Return a copy of `node` that also covers `words` (none of them on existing paths)."""
    new = Node(node.guess, dict(node.children))
    buckets: dict[int, list[int]] = {}
    for w in words:
        pat = int(game.P[node.guess, w])
        if pat != ALL_GREEN:
            buckets.setdefault(pat, []).append(w)
    for pat, ws in buckets.items():
        if pat in new.children:
            new.children[pat] = _graft(game, new.children[pat], ws, filler)
        else:
            new.children[pat] = filler.solve(tuple(sorted(ws)))[1]
    return new
