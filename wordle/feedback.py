"""Wordle feedback rules and precomputed pattern matrices.

A feedback pattern is encoded as a base-3 integer in [0, 242]:
digit i (weight 3**i) is the colour of letter i: 0 = gray, 1 = yellow, 2 = green.
ALL_GREEN = 242 means the guess was the answer.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ALL_GREEN = 242
_WEIGHTS = np.array([1, 3, 9, 27, 81], dtype=np.int32)


def load_words(name: str) -> list[str]:
    """Load data/<name>.txt ('answers' -> 2,315 words, 'allowed' -> 12,972 words)."""
    path = DATA_DIR / f"{name}.txt"
    return [w.strip().lower() for w in path.read_text().split() if w.strip()]


def feedback(guess: str, answer: str) -> int:
    """Reference (scalar) Wordle feedback. Greens first, then yellows from leftover counts."""
    res = [0] * 5
    remaining: dict[str, int] = {}
    for i in range(5):
        if guess[i] == answer[i]:
            res[i] = 2
        else:
            remaining[answer[i]] = remaining.get(answer[i], 0) + 1
    for i in range(5):
        if res[i] == 0 and remaining.get(guess[i], 0) > 0:
            res[i] = 1
            remaining[guess[i]] -= 1
    return sum(r * 3**i for i, r in enumerate(res))


def decode(pattern: int) -> str:
    """Pattern int -> string like 'GY..G' (G green, Y yellow, . gray)."""
    out = []
    for _ in range(5):
        out.append(".YG"[pattern % 3])
        pattern //= 3
    return "".join(out)


def _encode(words: list[str]) -> np.ndarray:
    return np.array([[ord(c) for c in w] for w in words], dtype=np.uint8)


def pattern_matrix(guesses: list[str], answers: list[str], chunk: int = 256) -> np.ndarray:
    """P[g, a] = feedback(guesses[g], answers[a]) as uint8, vectorised over chunks of guesses."""
    G, A = _encode(guesses), _encode(answers)
    out = np.empty((len(guesses), len(answers)), dtype=np.uint8)
    for s in range(0, len(guesses), chunk):
        g = G[s : s + chunk]                                   # (c, 5)
        green = g[:, None, :] == A[None, :, :]                 # (c, A, 5)
        res = 2 * green.astype(np.int32)
        for i in range(5):
            letter = g[:, i]                                   # (c,)
            # copies of this letter in the answer that are not already used by a green
            avail = ((A[None, :, :] == letter[:, None, None]) & ~green).sum(-1)
            # earlier non-green positions in the guess with the same letter claim yellows first
            if i:
                same = (g[:, :i] == letter[:, None])[:, None, :]      # (c, 1, i)
                used = (same & ~green[:, :, :i]).sum(-1)
            else:
                used = 0
            res[:, :, i] += (~green[:, :, i]) & (used < avail)
        out[s : s + chunk] = (res @ _WEIGHTS).astype(np.uint8)
    return out


def full_pattern_matrix(cache: bool = True) -> tuple[np.ndarray, list[str], list[str]]:
    """12,972 x 2,315 matrix for the real game (~30 MB), cached at data/patterns.npy."""
    guesses, answers = load_words("allowed"), load_words("answers")
    path = DATA_DIR / "patterns.npy"
    if cache and path.exists():
        P = np.load(path)
        if P.shape == (len(guesses), len(answers)):
            return P, guesses, answers
    P = pattern_matrix(guesses, answers)
    if cache:
        np.save(path, P)
    return P, guesses, answers
