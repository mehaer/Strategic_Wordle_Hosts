import numpy as np
import pytest

from wordle import decode, feedback, load_words, pattern_matrix


@pytest.mark.parametrize("guess,answer,expected", [
    ("speed", "abide", "..Y.Y"),   # only one E in ABIDE -> second E is gray
    ("erase", "speed", "Y..YY"),   # two Es in SPEED -> both Es yellow
    ("crane", "crane", "GGGGG"),
    ("llama", "hello", "YY..."),   # L at 0 and 1: both yellow (HELLO has two Ls at 2,3)
    ("hello", "llama", "..YY."),
    ("geese", "eerie", ".GY.G"),   # green E's consume first, one E left for a yellow
    ("abbey", "kebab", "YYGY."),   # second B green, first B yellow, E yellow
    ("aaaaa", "abcde", "G...."),
])
def test_feedback_cases(guess, answer, expected):
    assert decode(feedback(guess, answer)) == expected


def test_word_lists():
    answers, allowed = load_words("answers"), load_words("allowed")
    assert len(answers) == 2315 and len(allowed) == 12972
    assert set(answers) <= set(allowed)


def test_vectorised_matches_scalar():
    rng = np.random.default_rng(0)
    allowed, answers = load_words("allowed"), load_words("answers")
    g = rng.choice(allowed, 300).tolist() + ["speed", "geese", "llama", "abbey", "eerie"]
    a = rng.choice(answers, 200).tolist() + ["abide", "hello", "kebab", "eerie", "speed"]
    P = pattern_matrix(g, a)
    for i, gw in enumerate(g):
        for j, aw in enumerate(a):
            assert P[i, j] == feedback(gw, aw), (gw, aw)
