"""Tests for the benchmark runner."""

import json

from querypilot import evaluate


def test_fake_run_writes_logs_summary_and_resumes(shop, tmp_path, capsys):
    """Check that fake run writes logs summary and resumes."""
    out = tmp_path / "results"
    args = ["--data", str(shop["questions"]), "--db-root", str(shop["root"]),
            "--out", str(out), "--fake", "--yes"]
    evaluate.main(args)
    for setup in evaluate.SETUPS:
        lines = (out / f"{setup}.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2
        rec = json.loads(lines[0])
        assert {"correct", "category", "pred_sql", "gold_sql", "cost_usd", "trace"} <= rec.keys()
    summary = json.loads((out / "summary.json").read_text())
    assert set(summary["setups"]) == set(evaluate.SETUPS)
    assert (out / "summary.md").exists()
    evaluate.main(args)
    assert "0 still to run" in capsys.readouterr().out


def test_sample_is_fixed_and_spread(tmp_path):
    """Check that sample is fixed and spread."""
    import json as _j
    qs = [{"question_id": i, "db_id": f"db{i % 5}", "question": "q", "SQL": "SELECT 1"} for i in range(100)]
    f = tmp_path / "q.json"
    f.write_text(_j.dumps(qs))
    a = evaluate.load_questions(f, None, 20, 42)
    b = evaluate.load_questions(f, None, 20, 42)
    assert [q["question_id"] for q in a] == [q["question_id"] for q in b]
    assert len(a) == 20 and len({q["db_id"] for q in a}) == 5
