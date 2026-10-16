# QueryPilot — BIRD Mini-Dev results

Run: 2026-10-16 08:58 UTC · questions: 200 questions, seed 42 · evidence hints: True · temperature 0 · timeout 30 s · scoring: official BIRD EX (set equality)

Models: qwen2.5-coder:1.5b, qwen2.5-coder:7b · provider: ollama (local)

| Setup | Questions | EX | Tokens / question | Cost / question | Median time | LLM calls | Escalated |
|---|---|---|---|---|---|---|---|
| one_shot | 200 | **47.0%** | 2109 | $0.0000 | 14.16 s | 1 | 0.0% |
| agent | 200 | **51.0%** | 3188 | $0.0000 | 13.04 s | 1.33 | 0.0% |
| routed | 200 | **40.0%** | 4990 | $0.0000 | 21.61 s | 1.92 | 56.0% |

## By difficulty

| Setup | challenging | moderate | simple |
|---|---|---|---|
| one_shot | 38.7% | 43.9% | 54.9% |
| agent | 41.9% | 49.0% | 57.7% |
| routed | 41.9% | 31.6% | 50.7% |

## Where answers go wrong (every question counted)

- **one_shot**: correct: 94, wrong_logic: 37, sql_error: 27, wrong_tables: 21, empty_result: 13, wrong_columns: 8. Near-misses (not counted as correct): 6. Reference answer empty: 0. API errors: 0. Models: qwen2.5-coder:7b.
- **agent**: correct: 102, wrong_logic: 39, wrong_tables: 29, sql_error: 10, wrong_columns: 10, empty_result: 10. Near-misses (not counted as correct): 6. Reference answer empty: 0. API errors: 0. Models: qwen2.5-coder:7b.
- **routed**: correct: 80, wrong_logic: 38, wrong_tables: 33, sql_error: 21, empty_result: 19, wrong_columns: 9. Near-misses (not counted as correct): 6. Reference answer empty: 0. API errors: 0. Models: qwen2.5-coder:1.5b, qwen2.5-coder:7b.
