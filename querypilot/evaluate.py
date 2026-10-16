"""Benchmark runner: BIRD Mini-Dev (SQLite), three setups, same inputs.

    python -m querypilot.evaluate --data data/minidev/MINIDEV/mini_dev_sqlite.json \
        --db-root data/minidev/MINIDEV/dev_databases

Results are appended to results/<setup>.jsonl after every question, so an
interrupted run resumes where it stopped. results/summary.json and
results/summary.md are rebuilt from the logs at the end (or with --summary-only).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .agent import Agent, one_shot
from .llm import QuotaExhausted
from .safe_db import SafeDB
from .scoring import score

SETUPS = ("one_shot", "agent", "routed")


def load_questions(path: Path, limit: int | None, sample: int | None = None,
                   seed: int = 42) -> list[dict]:
    items = json.loads(path.read_text(encoding="utf-8"))
    for i, q in enumerate(items):
        q.setdefault("question_id", i)
        q.setdefault("difficulty", "unknown")
    if sample and sample < len(items):
        # Fixed random subset (same seed -> same questions every run), spread
        # over all databases and difficulties instead of the first N in file order.
        import random
        picked = random.Random(seed).sample(items, sample)
        items = sorted(picked, key=lambda q: q["question_id"])
    return items[:limit] if limit else items


def find_db(root: Path, db_id: str) -> Path:
    for candidate in (root / db_id / f"{db_id}.sqlite", root / f"{db_id}.sqlite"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No database for {db_id} under {root}")


def done_ids(path: Path) -> set:
    if not path.exists():
        return set()
    return {json.loads(line)["question_id"] for line in path.open(encoding="utf-8") if line.strip()}


def run_one(setup: str, llm, db: SafeDB, q: dict, schema: str, use_evidence: bool) -> dict:
    evidence = q.get("evidence", "") if use_evidence else ""
    trace: list[dict] = []
    start = time.monotonic()
    try:
        if setup == "one_shot":
            out = one_shot(llm, db, q["question"], evidence, schema)
        else:
            agent = Agent(llm, db, routed=(setup == "routed"), emit=trace.append)
            out = agent.run(q["question"], evidence, schema)
        api_error = ""
    except QuotaExhausted:
        raise                      # stop the run; nothing is recorded for this question
    except Exception as err:  # API failures are recorded, never dropped
        out, api_error = {"final_sql": "", "cost": 0.0, "tokens": 0, "calls": 0, "models_used": [],
                          "escalated": False, "repairs": 0}, f"{type(err).__name__}: {err}"
    latency = time.monotonic() - start
    s = score(db, out.get("final_sql", ""), q["SQL"])
    if api_error:
        s.update(correct=False, category="api_error", error=api_error)
    return {
        "question_id": q["question_id"], "db_id": q["db_id"], "difficulty": q["difficulty"],
        "question": q["question"], "evidence": evidence, "gold_sql": q["SQL"],
        "pred_sql": out.get("final_sql", ""), **s,
        "cost_usd": round(out.get("cost", 0.0), 6), "tokens": out.get("tokens", 0),
        "latency_s": round(latency, 2),
        "llm_calls": out.get("calls", 0), "repairs": out.get("repairs", 0),
        "escalated": out.get("escalated", False), "models": out.get("models_used", []),
        "trace": [{k: v for k, v in e.items() if k != "output"} for e in trace],
    }


def summarize(out_dir: Path, meta: dict) -> dict:
    summary = {"meta": meta, "setups": {}}
    for setup in SETUPS:
        path = out_dir / f"{setup}.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
        if not rows:
            continue
        n = len(rows)
        by_diff = {}
        for d in sorted({r["difficulty"] for r in rows}):
            sub = [r for r in rows if r["difficulty"] == d]
            by_diff[d] = {"n": len(sub), "ex": round(100 * sum(r["correct"] for r in sub) / len(sub), 1)}
        lat = [r["latency_s"] for r in rows]
        summary["setups"][setup] = {
            "n": n,
            "ex": round(100 * sum(r["correct"] for r in rows) / n, 1),
            "by_difficulty": by_diff,
            "categories": dict(Counter(r["category"] for r in rows).most_common()),
            "near_misses_not_counted": sum(r["near_miss"] for r in rows),
            "gold_empty": sum(r["gold_empty"] for r in rows),
            "api_errors": sum(r["category"] == "api_error" for r in rows),
            "total_cost_usd": round(sum(r["cost_usd"] for r in rows), 4),
            "cost_per_question_usd": round(sum(r["cost_usd"] for r in rows) / n, 6),
            "tokens_per_question": round(statistics.mean(r.get("tokens", 0) for r in rows)),
            "avg_latency_s": round(statistics.mean(lat), 2),
            "median_latency_s": round(statistics.median(lat), 2),
            "avg_llm_calls": round(statistics.mean(r["llm_calls"] for r in rows), 2),
            "escalation_rate": round(100 * sum(r["escalated"] for r in rows) / n, 1),
            "models": sorted({m for r in rows for m in r["models"]}),
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(to_markdown(summary), encoding="utf-8")
    return summary


def to_markdown(summary: dict) -> str:
    m = summary["meta"]
    lines = [f"# QueryPilot — BIRD Mini-Dev results",
             "",
             f"Run: {m.get('date', '?')} · questions: {m.get('sample', 'all')} · evidence hints: {m.get('evidence', '?')} · "
             f"temperature 0 · timeout 30 s · scoring: official BIRD EX (set equality)",
             "",
             f"Models: {m.get('models', '?')} · provider: {m.get('provider', '?')}",
             "",
             "| Setup | Questions | EX | Tokens / question | Cost / question | Median time | LLM calls | Escalated |",
             "|---|---|---|---|---|---|---|---|"]
    for name, s in summary["setups"].items():
        lines.append(f"| {name} | {s['n']} | **{s['ex']}%** | {s.get('tokens_per_question', 0)} | "
                     f"${s['cost_per_question_usd']:.4f} | {s['median_latency_s']} s | {s['avg_llm_calls']} | "
                     f"{s['escalation_rate']}% |")
    lines += ["", "## By difficulty", "", "| Setup | " + " | ".join(
        sorted({d for s in summary['setups'].values() for d in s['by_difficulty']})) + " |"]
    diffs = sorted({d for s in summary['setups'].values() for d in s['by_difficulty']})
    lines.append("|---|" + "---|" * len(diffs))
    for name, s in summary["setups"].items():
        cells = [f"{s['by_difficulty'].get(d, {}).get('ex', '–')}%" for d in diffs]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "## Where answers go wrong (every question counted)", ""]
    for name, s in summary["setups"].items():
        cats = ", ".join(f"{k}: {v}" for k, v in s["categories"].items())
        lines.append(f"- **{name}**: {cats}. Near-misses (not counted as correct): "
                     f"{s['near_misses_not_counted']}. Reference answer empty: {s['gold_empty']}. "
                     f"API errors: {s['api_errors']}. Models: {', '.join(s['models'])}.")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/minidev/MINIDEV/mini_dev_sqlite.json"))
    ap.add_argument("--db-root", type=Path, default=Path("data/minidev/MINIDEV/dev_databases"))
    ap.add_argument("--setups", default=",".join(SETUPS))
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions")
    ap.add_argument("--sample", type=int, default=None,
                    help="a fixed random subset of N questions (same for every setup and every run)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-evidence", action="store_true", help="hide BIRD's evidence hints from all setups")
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--yes", action="store_true", help="don't ask before continuing past the cost check")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--fake", action="store_true", help="offline dry run with a fake model (no API cost)")
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    meta = {"date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "evidence": not args.no_evidence, "data": str(args.data),
            "sample": f"{args.sample} questions, seed {args.seed}" if args.sample else "all"}
    if args.summary_only:
        print(to_markdown(summarize(args.out, meta)))
        return

    if args.fake:
        from .demo import offline_llm
        llm = offline_llm()
    else:
        from .llm import make_client
        llm = make_client()
    from .llm import LARGE, PROVIDER, SMALL
    meta.update(provider="fake" if args.fake else PROVIDER, models=f"small={SMALL}, large={LARGE}")

    setups = [s.strip() for s in args.setups.split(",") if s.strip() in SETUPS]
    questions = load_questions(args.data, args.limit, args.sample, args.seed)
    finished = {s: done_ids(args.out / f"{s}.jsonl") for s in setups}
    todo = [q for q in questions if any(q["question_id"] not in finished[s] for s in setups)]
    print(f"{len(questions)} questions · {len(todo)} still to run · setups: {', '.join(setups)}")

    dbs, schemas = {}, {}
    total = sum(1 for s_ in setups for q in todo if q["question_id"] not in finished[s_])
    spent, done, checked, t0 = 0.0, 0, False, time.monotonic()
    stop = False
    # One setup at a time: on a small GPU this keeps a single model loaded
    # instead of swapping models on every question.
    for setup in setups:
        if stop:
            break
        for q in todo:
            if q["question_id"] in finished[setup]:
                continue
            if q["db_id"] not in dbs:
                dbs[q["db_id"]] = SafeDB(find_db(args.db_root, q["db_id"]))
                schemas[q["db_id"]] = dbs[q["db_id"]].schema_context()
            try:
                rec = run_one(setup, llm, dbs[q["db_id"]], q, schemas[q["db_id"]], not args.no_evidence)
            except QuotaExhausted as err:
                print(f"\n{err}")
                stop = True
                break
            with (args.out / f"{setup}.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")
            spent += rec["cost_usd"]
            done += 1
            mark = "✓" if rec["correct"] else "✗"
            print(f"[{done}/{total}] {setup:8} {mark} {rec['category']:14} "
                  f"{rec['tokens']:>6} tok  {rec['latency_s']}s  q{q['question_id']}")
            if done == 20 and not checked and total > 20:
                checked = True
                hours = (time.monotonic() - t0) / done * total / 3600
                msg = f"\nCheck: 20 runs done → about {hours:.1f} h for all {total} runs"
                if spent > 0:
                    msg += f", about ${spent / done * total:.2f}"
                print(msg + " (the first setup is usually the fastest).")
                if spent > 0 and not args.yes and input("Continue? [y/N] ").strip().lower() != "y":
                    print("Stopped. Run the same command again later to resume.")
                    stop = True
                    break
    summary = summarize(args.out, meta)
    print("\n" + to_markdown(summary))


if __name__ == "__main__":
    sys.exit(main())
