"""Offline tests for the LLM-host harness (no network, no API key)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from wordle.llm_host import build_prompt, fake_chat, parse_explicit, parse_sample

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
WORDS = ["light", "might", "crane", "slate", "pound"]


@pytest.mark.parametrize("text,expected", [
    ("Thinking...\nANSWER: LIGHT", ("light", "ok")),
    ("ANSWER: **Might**", ("might", "ok")),
    ("answer:crane", ("crane", "ok")),
    ("ANSWER: SLATE\n...wait.\nANSWER: POUND", ("pound", "ok")),      # last answer wins
    ("ANSWER: ZEBRA", (None, "not_in_list")),
    ("I would pick light or might.", (None, "no_answer")),
    ("Slate.", ("slate", "ok")),                                       # bare one-word reply
    ("", (None, "no_answer")),
])
def test_parse_sample(text, expected):
    assert parse_sample(text, WORDS) == expected


def test_parse_explicit_json_with_noise():
    text = 'Reasoning... {"LIGHT": 0.5, "might": 0.25, "zebra": 0.25}'
    p, info = parse_explicit(text, WORDS)
    assert info["status"] == "ok"
    assert p == pytest.approx({"light": 2 / 3, "might": 1 / 3})
    assert info["off_list_mass"] == pytest.approx(0.25)


def test_parse_explicit_code_fence_and_unnormalised():
    text = 'Here:\n```json\n{"light": 2, "crane": 2}\n```\nDone.'
    p, info = parse_explicit(text, WORDS)
    assert p == pytest.approx({"light": 0.5, "crane": 0.5}) and info["raw_sum"] == 4


def test_parse_explicit_lenient_and_failures():
    p, info = parse_explicit('{"light": 0.6, "pound": 0.4,}', WORDS)       # trailing comma
    assert info["status"] == "ok" and p["light"] == pytest.approx(0.6)
    assert parse_explicit("no numbers here", WORDS)[1]["status"] == "no_json"
    assert parse_explicit('{"zebra": 1.0}', WORDS)[1]["status"] == "zero_mass"


def test_prompt_shuffles_but_keeps_all_words():
    a = build_prompt(WORDS, "optimal", "sample", seed=1)
    b = build_prompt(WORDS, "plain", "explicit", seed=2)
    for w in WORDS:
        assert w.upper() in a and w.upper() in b
    assert "optimally" in a and "optimally" not in b
    assert "ANSWER:" in a and "JSON" in b


def test_fake_end_to_end(tmp_path):
    """Full pipeline with the fake model: run (resumable) then score."""
    import run_llm_hosts
    import score_llm_hosts
    cfg = json.loads((ROOT / "configs" / "llm_hosts.json").read_text())
    cfg.update(word_lists=["results/main/n25_s0.json", "results/cluster/n50_ight.json"],
               samples=6, explicit_repeats=2, models=[{"id": "fake/model", "budget_usd": 1.0}])
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    out = tmp_path / "out"
    args = ["--config", str(cfg_path), "--out", str(out), "--fake"]
    run_llm_hosts.main(["run", *args])
    raw = out / "raw" / "fake__model.jsonl"
    n_calls = 2 * 2 * (6 + 2)                       # conditions x lists x calls per unit
    assert len(raw.read_text().splitlines()) == n_calls
    run_llm_hosts.main(["run", *args])               # resume: nothing new
    assert len(raw.read_text().splitlines()) == n_calls

    score_llm_hosts.main(["--config", str(cfg_path), "--out", str(out)])
    rows = list(__import__("csv").DictReader((out / "llm_scores.csv").open()))
    fake = [r for r in rows if r["model"] == "fake/model"]
    assert len(fake) == 2 * 2 * 2 and all(r["n_ok"] == r["n_calls"] for r in fake)
    for r in rows:
        assert float(r["v_llm"]) >= float(r["v_unif"]) - 1 - 1e-9          # sanity: finite values
        assert float(r["v_llm"]) <= float(r["v_star"]) + 1e-9              # nobody beats p*
        assert 0 <= float(r["tv"]) <= 1
    assert any(r["model"].startswith("ref: p*") for r in rows)
