"""Ask LLMs to act as the Wordle host and log every reply (scored later by score_llm_hosts.py).

    python scripts/run_llm_hosts.py check            # verify model ids, prices, key limit (free)
    python scripts/run_llm_hosts.py pilot            # a few calls per model; prints projected cost
    python scripts/run_llm_hosts.py run              # the full experiment (resumable, budget-capped)
    python scripts/run_llm_hosts.py run --models anthropic/claude-haiku-5.5
    python scripts/run_llm_hosts.py run --fake       # offline dry run with a fake model, $0

Settings live in configs/llm_hosts.json. Every call is appended to results/llm/raw/<model>.jsonl
(pilot calls to results/llm/pilot/). Re-running `run` skips calls that already succeeded, so an
interrupted run can simply be restarted.

Work is done in units of (condition, word list): `samples` pick-one-word calls plus
`explicit_repeats` give-a-distribution calls. Units run in config order (all lists under the
first condition, then the second), so if a model's budget runs out, what is finished is complete
units of the most important condition. Before each unit the script estimates its cost from the
model's average cost per call so far and stops that model if the unit would exceed
the model's `budget_usd` or the overall `total_budget_usd`.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from wordle.llm_host import (  # noqa: E402
    ApiError, api_key, build_prompt, chat, fake_chat, key_status, list_models, order_seed,
    parse_explicit, parse_sample,
)

METHODS = ("sample", "explicit")


def slug(model_id: str) -> str:
    return model_id.replace("/", "__").replace(":", "_")


def list_name(path: str) -> str:
    p = Path(path)
    return f"{p.parent.name}/{p.stem}"                  # e.g. main/n50_s0, cluster/n50_ight


def load_lists(cfg: dict) -> dict[str, list[str]]:
    out = {}
    for f in cfg["word_lists"]:
        path = ROOT / f
        if not path.exists():
            raise SystemExit(f"word list file missing: {f}")
        out[list_name(f)] = json.loads(path.read_text())["words"]
    return out


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def all_spend(out: Path) -> float:
    return sum(r.get("cost", 0.0) for f in out.rglob("*.jsonl") for r in read_jsonl(f))


class Logger:
    """Thread-safe JSONL appender that also tracks spend."""

    def __init__(self, path: Path, spent: float):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path, self.spent, self.lock = path, spent, threading.Lock()

    def write(self, rec: dict) -> None:
        with self.lock:
            with self.path.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            self.spent += rec.get("cost", 0.0)


def one_call(model: dict, cfg: dict, words: list[str], lname: str, cond: str, method: str, i: int,
             call_fn, key: str | None) -> dict:
    call_id = f"{lname}|{cond}|{method}|{i}"
    seed = order_seed(call_id)
    prompt = build_prompt(words, cond, method, seed)
    rec = dict(call_id=call_id, model=model["id"], list=lname, condition=cond, method=method,
               index=i, order_seed=seed, time=time.strftime("%Y-%m-%d %H:%M:%S"))
    try:
        res = call_fn(model["id"], prompt, key=key, temperature=cfg["temperature"],
                      max_tokens=cfg["max_tokens"], reasoning=model.get("reasoning", cfg.get("reasoning")))
    except ApiError as e:
        rec.update(status="api_error", error=str(e), fatal=e.fatal, cost=0.0)
        return rec
    rec.update(res)
    if method == "sample":
        word, status = parse_sample(res["text"], words)
        rec.update(status=status, word=word)
    else:
        p, info = parse_explicit(res["text"], words)
        rec.update(info, p=p)
    return rec


def unit_calls(cfg: dict) -> list[tuple[str, int]]:
    return [("sample", i) for i in range(cfg["samples"])] + \
           [("explicit", i) for i in range(cfg["explicit_repeats"])]


def avg_cost(records: list[dict]) -> float | None:
    paid = [r["cost"] for r in records if r.get("status") != "api_error"]
    return sum(paid) / len(paid) if paid else None


def run_model(model: dict, cfg: dict, lists: dict, out: Path, call_fn, key: str | None,
              pilot: bool) -> dict:
    sub = "pilot" if pilot else "raw"
    path = out / sub / f"{slug(model['id'])}.jsonl"
    existing = read_jsonl(path)
    done = {r["call_id"] for r in existing if r.get("status") != "api_error"}
    log = Logger(path, sum(r.get("cost", 0.0) for r in existing))
    history = existing + read_jsonl(out / "pilot" / f"{slug(model['id'])}.jsonl")
    budget = model.get("budget_usd", float("inf"))
    total_budget = cfg.get("total_budget_usd", float("inf"))
    total_spent_other = all_spend(out) - log.spent

    if pilot:
        first = next(iter(lists))
        units = [(cfg["conditions"][0], first)]
        calls = [("sample", i) for i in range(4)] + [("explicit", 0)]
    else:
        units = [(c, l) for c in cfg["conditions"] for l in lists]
        calls = unit_calls(cfg)

    stopped, errors, fatal = None, 0, None
    with ThreadPoolExecutor(max_workers=cfg.get("concurrency", 4)) as pool:
        for cond, lname in units:
            todo = [(m, i) for m, i in calls if f"{lname}|{cond}|{m}|{i}" not in done]
            if not todo:
                continue
            a = avg_cost(history)
            est = (a or 0.0) * len(todo)
            if log.spent + est > budget or total_spent_other + log.spent + est > total_budget:
                stopped = f"budget: next unit ({cond}, {lname}) est ${est:.2f}, spent ${log.spent:.2f}"
                break
            print(f"  {model['id']}: {cond:8s} {lname:16s} {len(todo):3d} calls "
                  f"(spent ${log.spent:.3f}, est unit ${est:.3f})", flush=True)
            futs = [pool.submit(one_call, model, cfg, lists[lname], lname, cond, m, i, call_fn, key)
                    for m, i in todo]
            for f in futs:
                rec = f.result()
                log.write(rec)
                history.append(rec)
                if rec.get("status") == "api_error":
                    errors += 1
                    if rec.get("fatal"):
                        fatal = rec["error"]
            if fatal:
                stopped = f"fatal API error: {fatal}"
                break
            if log.spent > budget * 1.05:                 # hard stop if estimates were badly off
                stopped = f"budget overshoot: spent ${log.spent:.2f} > ${budget:.2f}"
                break
    return dict(model=model["id"], spent=log.spent, errors=errors, stopped=stopped,
                records=read_jsonl(path))


def summarize_pilot(res: dict, cfg: dict, n_lists: int) -> None:
    recs = [r for r in res["records"] if r.get("status") != "api_error"]
    by = {m: [r for r in recs if r["method"] == m] for m in METHODS}
    cs = {m: (sum(r["cost"] for r in v) / len(v) if v else float("nan")) for m, v in by.items()}
    ok = {m: sum(r.get("status") == "ok" for r in v) for m, v in by.items()}
    units = len(cfg["conditions"]) * n_lists
    full = units * (cfg["samples"] * cs["sample"] + cfg["explicit_repeats"] * cs["explicit"])
    toks = [r.get("completion_tokens") or 0 for r in recs]
    print(f"\n{res['model']}")
    print(f"  cost/call: sample ${cs['sample']:.5f}  explicit ${cs['explicit']:.5f}   "
          f"avg completion tokens {sum(toks) / max(len(toks), 1):.0f}")
    print(f"  parsed OK: sample {ok['sample']}/{len(by['sample'])}  "
          f"explicit {ok['explicit']}/{len(by['explicit'])}   API errors {res['errors']}")
    print(f"  projected full run: ${full:.2f}  ({units} units)   budget ${res['budget']:.2f}")
    for r in recs[:1] + by["explicit"][:1]:
        print(f"  example {r['method']} reply (last 300 chars): {r['text'][-300:]!r}")
    if res["stopped"]:
        print(f"  STOPPED: {res['stopped']}")


def cmd_check(cfg: dict, args) -> None:
    catalogue = list_models()
    ok = True
    for m in cfg["models"]:
        meta = catalogue.get(m["id"])
        if meta is None:
            ok = False
            fam = m["id"].split("/")[0]
            near = [i for i in catalogue if i.startswith(fam + "/")
                    and any(t in i for t in m["id"].split("/")[1].split("-")[:2])][:10]
            print(f"MISSING  {m['id']}\n         similar ids: {near}")
            continue
        pr = meta.get("pricing", {})
        params = meta.get("supported_parameters", [])
        print(f"OK       {m['id']:32s} in ${float(pr.get('prompt', 0)) * 1e6:.3f}/M  "
              f"out ${float(pr.get('completion', 0)) * 1e6:.3f}/M  "
              f"reasoning={'reasoning' in params}  temperature={'temperature' in params}  "
              f"budget ${m.get('budget_usd', 0):.2f}")
    if not args.fake:
        try:
            ks = key_status(api_key())
            print(f"\nAPI key: usage ${ks.get('usage', 0):.4f}, limit {ks.get('limit')}, "
                  f"limit remaining {ks.get('limit_remaining')}")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"\nAPI key check failed: {e}")
    total = sum(m.get("budget_usd", 0) for m in cfg["models"])
    print(f"\nsum of model budgets ${total:.2f}, total cap ${cfg.get('total_budget_usd')}")
    if not ok:
        sys.exit(1)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["check", "pilot", "run"])
    ap.add_argument("--config", default=str(ROOT / "configs" / "llm_hosts.json"))
    ap.add_argument("--out", default=str(ROOT / "results" / "llm"))
    ap.add_argument("--models", nargs="*", help="only these model ids (default: all in config)")
    ap.add_argument("--fake", action="store_true", help="offline fake model, no API key, $0")
    args = ap.parse_args(argv)

    cfg = json.loads(Path(args.config).read_text())
    if args.command == "check":
        if args.fake:
            print("check needs the network; nothing to check with --fake")
            return
        return cmd_check(cfg, args)

    models = [m for m in cfg["models"] if not args.models or m["id"] in args.models]
    if not models:
        raise SystemExit(f"no configured model matches {args.models}")
    lists = load_lists(cfg)
    out = Path(args.out)
    key = None if args.fake else api_key()
    call_fn = fake_chat if args.fake else chat
    if not args.fake:                                     # validate ids; send `reasoning` only where supported
        catalogue = list_models()
        for m in models:
            meta = catalogue.get(m["id"])
            if meta is None:
                raise SystemExit(f"unknown model id {m['id']}: run `check` and fix configs/llm_hosts.json")
            if "reasoning" not in meta.get("supported_parameters", []):
                m["reasoning"] = None
    pilot = args.command == "pilot"

    results = []
    for m in models:
        print(f"== {m['id']} ({'pilot' if pilot else 'run'})", flush=True)
        res = run_model(m, cfg, lists, out, call_fn, key, pilot)
        res["budget"] = m.get("budget_usd", float("inf"))
        results.append(res)
        if pilot:
            summarize_pilot(res, cfg, len(lists))
        elif res["stopped"]:
            print(f"  STOPPED: {res['stopped']}")
        if res["stopped"] and res["stopped"].startswith("fatal"):
            break

    print(f"\ntotal spend in {out}: ${all_spend(out):.4f}")
    if not pilot:
        for r in results:
            recs = r["records"]
            ok = sum(x.get("status") == "ok" for x in recs)
            print(f"  {r['model']:32s} calls {len(recs):4d}  parsed ok {ok:4d}  "
                  f"api errors {r['errors']:3d}  spent ${r['spent']:.3f}"
                  + (f"  STOPPED ({r['stopped']})" if r["stopped"] else ""))


if __name__ == "__main__":
    main()
