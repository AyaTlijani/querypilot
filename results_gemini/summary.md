# QueryPilot — BIRD Mini-Dev results

Run: 2026-10-16 08:54 UTC · questions: all · evidence hints: True · temperature 0 · timeout 30 s · scoring: official BIRD EX (set equality)

Models: ? · provider: ?

| Setup | Questions | EX | Tokens / question | Cost / question | Median time | LLM calls | Escalated |
|---|---|---|---|---|---|---|---|
| one_shot | 200 | **62.5%** | 2285 | $0.0000 | 4.19 s | 1 | 0.0% |
| agent | 200 | **64.0%** | 2502 | $0.0000 | 4.27 s | 1.06 | 0.0% |

## By difficulty

| Setup | challenging | moderate | simple |
|---|---|---|---|
| one_shot | 58.1% | 60.2% | 67.6% |
| agent | 61.3% | 63.3% | 66.2% |

## Where answers go wrong (every question counted)

- **one_shot**: correct: 125, wrong_logic: 33, wrong_tables: 20, empty_result: 11, wrong_columns: 9, sql_error: 2. Near-misses (not counted as correct): 5. Reference answer empty: 0. API errors: 0. Models: gemini-3.5-flash-lite.
- **agent**: correct: 128, wrong_logic: 38, wrong_tables: 22, wrong_columns: 10, empty_result: 2. Near-misses (not counted as correct): 4. Reference answer empty: 0. API errors: 0. Models: gemini-3.5-flash-lite.
