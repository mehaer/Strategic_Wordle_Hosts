"""Check the DP best response and double oracle against brute force on tiny instances,
where every decision tree can be enumerated and the full game solved by one LP."""
import itertools

import numpy as np
import pytest

from wordle import ALL_GREEN, Game, best_response, double_oracle, solve_host_lp

IGHT = ["fight", "light", "might", "night", "right", "sight", "tight", "eight"]


def all_cost_vectors(game: Game, S=None):
    """Every decision tree over S (guesses from game.guesses), as {w: depth} dicts.
    Only splitting guesses are used (non-splitting ones are dominated)."""
    S = tuple(range(game.n)) if S is None else S
    if len(S) == 1:
        return [{S[0]: 1}]
    out = []
    for g in range(len(game.guesses)):
        buckets = {}
        for w in S:
            buckets.setdefault(int(game.P[g, w]), []).append(w)
        if len(buckets) == 1 and ALL_GREEN not in buckets:
            continue
        solved = buckets.pop(ALL_GREEN, [])
        child_options = [all_cost_vectors(game, tuple(ws)) for ws in buckets.values()]
        for combo in itertools.product(*child_options):
            c = {w: 1 for w in solved}
            for sub in combo:
                c.update({w: d + 1 for w, d in sub.items()})
            out.append(c)
    return out


def brute_matrix(game):
    vecs = all_cost_vectors(game)
    return np.unique(np.array([[c[w] for w in range(game.n)] for c in vecs]), axis=0)


@pytest.fixture(scope="module")
def tiny_games():
    games = [Game(IGHT[:5]), Game(["light", "might", "sight", "crane", "slate"])]
    rng = np.random.default_rng(1)
    from wordle import load_words
    pool = load_words("answers")
    for _ in range(3):
        games.append(Game(sorted(rng.choice(pool, 5, replace=False).tolist())))
    return games


def test_best_response_matches_brute_force(tiny_games):
    rng = np.random.default_rng(0)
    for game in tiny_games:
        C = brute_matrix(game)
        for _ in range(20):
            p = rng.dirichlet(np.ones(game.n) * 0.5)
            if rng.random() < 0.3:              # exercise zero-probability words
                p[rng.integers(game.n)] = 0
                p /= p.sum()
            br = best_response(game, p)
            assert br.value == pytest.approx((C @ p).min(), abs=1e-9)
            assert br.value == pytest.approx(p @ br.cost, abs=1e-12)


@pytest.mark.parametrize("host_oracle", [False, True])
def test_double_oracle_matches_full_lp(tiny_games, host_oracle):
    for game in tiny_games:
        _, v_full, _ = solve_host_lp(brute_matrix(game).astype(float))
        res = double_oracle(game, eps=1e-9, verbose=False, host_oracle=host_oracle, init_words=2)
        assert res.lower <= v_full + 1e-7 and v_full <= res.upper + 1e-7
        assert res.upper - res.lower <= 1e-6
        assert res.p.sum() == pytest.approx(1) and res.q.sum() == pytest.approx(1)
        # q* guarantees the upper bound against every word
        assert (res.q @ res.costs).max() <= res.upper + 1e-7


def test_both_modes_agree_on_larger_subset():
    game = Game.subset(n=40, seed=0, guesses="answers")
    a = double_oracle(game, eps=1e-6, verbose=False)
    b = double_oracle(game, eps=1e-6, verbose=False, host_oracle=True)
    for res in (a, b):
        assert res.v_uniform <= res.lower + 1e-9 <= res.upper + 1e-9
        assert res.gap <= 1e-6
    assert a.lower == pytest.approx(b.lower, abs=2e-6)


def test_all_guesses_at_least_as_strong():
    small = Game.subset(n=20, seed=3, guesses="answers")
    big = Game.subset(n=20, seed=3, guesses="all")
    p = np.full(20, 1 / 20)
    assert best_response(big, p).value <= best_response(small, p).value + 1e-12
