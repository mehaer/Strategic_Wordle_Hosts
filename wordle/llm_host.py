"""LLMs as Wordle hosts: prompts, an OpenRouter client, and parsing of model replies.

Two ways to get a host distribution p out of a model:

    sample:   ask "pick one word" many times at temperature 1; p = empirical frequencies.
    explicit: ask for a probability for every word, returned as a JSON object.

Two prompt conditions:

    optimal:  the model is told the guesser plays optimally and knows the host's probabilities
              (the equilibrium setting: the host commits to p, the guesser best-responds).
    plain:    the model is only told its goal (make the guesser need many guesses).
"""
from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.error
import urllib.request
import zlib

API_URL = "https://openrouter.ai/api/v1"

INTRO = (
    "We are playing Wordle. You are the host: you choose the secret answer, and a guesser tries "
    "to find it in as few guesses as possible.\n\n"
    "The answer must be one of the {n} words listed below, and the guesser knows this list. "
    "The guesser may guess any valid 5-letter English word. After each guess they see standard "
    "Wordle feedback for every letter: green (right letter, right position), yellow (the letter "
    "is in the answer but in a different position), or gray (the letter is not in the answer).\n\n"
    "Your goal is to make the guesser need as many guesses as possible on average."
)

CONDITIONS = {
    "optimal": (
        "Assume the guesser plays optimally: they know exactly how likely you are to choose each "
        "word, and they use the guessing strategy that minimizes their expected number of guesses "
        "against your choice."
    ),
    "plain": "",
}

ASK = {
    "sample": (
        "Choose one word from the list. You may think it through first, but end your reply with a "
        "final line of exactly this form:\nANSWER: <word>"
    ),
    "explicit": (
        "Give the probability with which you would choose each word as the answer. End your reply "
        "with a single JSON object mapping words to probabilities, for example "
        '{"crane": 0.1, "slate": 0.05}. Words you leave out get probability 0. '
        "The probabilities must sum to 1."
    ),
}


def order_seed(call_id: str) -> int:
    """Deterministic per-call seed (used to shuffle the word list shown to the model)."""
    return zlib.crc32(call_id.encode())


def build_prompt(words: list[str], condition: str, method: str, seed: int) -> str:
    """The full user message. The word list is shuffled per call to avoid position bias."""
    order = list(words)
    random.Random(seed).shuffle(order)
    parts = [INTRO.format(n=len(words))]
    if CONDITIONS[condition]:
        parts.append(CONDITIONS[condition])
    parts.append("Words:\n" + "\n".join(w.upper() for w in order))
    parts.append(ASK[method])
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- parsing

_ANSWER_RE = re.compile(r"ANSWER\s*:\s*[*_`\"']*\s*([A-Za-z]{5})\b", re.IGNORECASE)
_PAIR_RE = re.compile(r"[\"']?([A-Za-z]{5})[\"']?\s*:\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")


def parse_sample(text: str, words: list[str]) -> tuple[str | None, str]:
    """Return (word, status). status: ok | not_in_list | no_answer."""
    found = _ANSWER_RE.findall(text or "")
    if not found:
        stripped = (text or "").strip().strip(".*_`\"'").lower()
        if len(stripped) == 5 and stripped.isalpha():      # bare one-word reply
            found = [stripped]
    if not found:
        return None, "no_answer"
    w = found[-1].lower()
    return (w, "ok") if w in set(words) else (None, "not_in_list")


def parse_explicit(text: str, words: list[str]) -> tuple[dict[str, float] | None, dict]:
    """Return (p, info). p maps list words to normalised probabilities (None if unusable).

    info: status (ok | no_json | zero_mass), raw_sum (sum of all numbers the model gave),
    off_list_mass (share of that sum on words not in the list, which are dropped).
    """
    text = text or ""
    raw: dict[str, float] | None = None
    end = text.rfind("}")
    start = text.rfind("{", 0, end) if end >= 0 else -1
    if start >= 0:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                raw = {str(k).strip().lower(): float(v) for k, v in obj.items()
                       if isinstance(v, (int, float)) and not isinstance(v, bool)}
        except (ValueError, TypeError):
            raw = None
    if raw is None:                                       # lenient fallback: "word": number pairs
        pairs = _PAIR_RE.findall(text[start:] if start >= 0 else text)
        raw = {k.lower(): float(v) for k, v in pairs} if pairs else None
    if not raw:
        return None, dict(status="no_json", raw_sum=0.0, off_list_mass=0.0)
    raw = {k: max(v, 0.0) for k, v in raw.items()}
    total = sum(raw.values())
    ws = set(words)
    on = {k: v for k, v in raw.items() if k in ws and v > 0}
    on_sum = sum(on.values())
    info = dict(raw_sum=total, off_list_mass=(1 - on_sum / total) if total > 0 else 0.0)
    if on_sum <= 0:
        return None, dict(status="zero_mass", **info)
    return {k: v / on_sum for k, v in on.items()}, dict(status="ok", **info)


# --------------------------------------------------------------------------- OpenRouter

def api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is not set. Run: export OPENROUTER_API_KEY=sk-or-...")
    return key


def _request(url: str, body: dict | None, key: str | None, timeout: float) -> dict:
    headers = {"Content-Type": "application/json", "X-Title": "Strategic Wordle Hosts"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


class ApiError(RuntimeError):
    def __init__(self, msg: str, fatal: bool = False):
        super().__init__(msg)
        self.fatal = fatal          # True: stop the whole run (bad key, no credits, bad model id)


def chat(model: str, prompt: str, *, key: str, temperature: float = 1.0, max_tokens: int = 4000,
         reasoning: dict | None = None, timeout: float = 300, retries: int = 4) -> dict:
    """One chat completion. Returns text, finish_reason, cost (USD) and token counts."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature, "max_tokens": max_tokens}
    if reasoning:
        body["reasoning"] = reasoning
    for attempt in range(retries):
        try:
            data = _request(f"{API_URL}/chat/completions", body, key, timeout)
            if "error" in data:
                raise ApiError(f"API error: {json.dumps(data['error'])[:500]}")
            break
        except urllib.error.HTTPError as e:
            msg = f"HTTP {e.code}: {e.read().decode(errors='replace')[:500]}"
            if e.code in (401, 402, 403, 404):
                raise ApiError(msg, fatal=True) from None
            if attempt == retries - 1:
                raise ApiError(msg) from None
        except (urllib.error.URLError, TimeoutError, ApiError) as e:
            if isinstance(e, ApiError) and e.fatal or attempt == retries - 1:
                raise ApiError(str(e)) from None
        time.sleep(5 * 2**attempt)
    choice = data["choices"][0]
    usage = data.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    return dict(
        text=(choice.get("message") or {}).get("content") or "",
        finish_reason=choice.get("finish_reason"),
        cost=float(usage.get("cost") or 0.0),
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        reasoning_tokens=details.get("reasoning_tokens"),
        provider=data.get("provider"),
        model_used=data.get("model"),
    )


def list_models(timeout: float = 60) -> dict[str, dict]:
    """Public model catalogue: id -> metadata (pricing per token, supported parameters, ...)."""
    return {m["id"]: m for m in _request(f"{API_URL}/models", None, None, timeout)["data"]}


def key_status(key: str, timeout: float = 60) -> dict:
    """Spend and limit of the API key (GET /api/v1/key)."""
    return _request(f"{API_URL}/key", None, key, timeout).get("data", {})


# --------------------------------------------------------------------------- offline fake

def fake_chat(model: str, prompt: str, **_) -> dict:
    """Deterministic stand-in for `chat`, for dry runs and tests (no network, zero cost)."""
    words = [ln.strip().lower() for ln in prompt.split("Words:\n", 1)[1].split("\n\n", 1)[0].split("\n")]
    rng = random.Random(zlib.crc32(prompt.encode()))
    favourites = sorted(words)[:8]                       # a fake "taste": early-alphabet words
    if "ANSWER:" in prompt:
        text = f"Some thoughts.\nANSWER: {rng.choice(favourites).upper()}"
    else:
        ws = rng.sample(favourites, 5)
        text = "Here you go:\n" + json.dumps({w: 0.2 for w in ws})
    return dict(text=text, finish_reason="stop", cost=0.0, prompt_tokens=len(prompt) // 4,
                completion_tokens=len(text) // 4, reasoning_tokens=0, provider="fake",
                model_used=model)
