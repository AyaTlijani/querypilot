# QueryPilot

**An autonomous data analyst agent.** Ask a business question in plain language; QueryPilot explores the database, writes SQL, runs it, checks whether the result makes sense, and repairs its own mistakes. Every step is visible, and every query is read-only.

It is evaluated on **BIRD Mini-Dev**, a public text-to-SQL benchmark of 500 questions over real-world databases, comparing three setups on accuracy, cost and latency.

`LangGraph` · `Ollama` · `Qwen2.5-Coder` · `MCP` · `FastAPI` · `SQLite` · `Docker`

It runs **entirely locally on open-weight models**: no API cost, and the data never leaves the machine.

---

## What it does

```
question ─▶ think ─▶ use tools ─▶ think ─▶ submit SQL ─▶ verify ─┬─▶ answer
              ▲                                                  │
              └──────────── repair (error / 0 rows) ◀────────────┘
```

- **Explores before answering.** Tools: `list_tables`, `describe_table`, `sample_rows`, `run_sql` (exploration) and `submit_sql` (final answer). It checks real values, for example that a status is stored as `'delivered'` and not `'Delivered'`, before filtering on them.
- **Verifies and self-repairs.** A submitted query is executed. If it fails or returns nothing, the agent gets the error back and fixes the query (up to 3 repairs).
- **Routes between models for efficiency.** It starts on a small model (Qwen2.5-Coder 3B) and escalates to the larger one (7B) only when verification fails or the step budget runs out.
- **Read-only by construction** (see [Safety](#safety)).
- **Speaks MCP.** The same tools are exposed as a Model Context Protocol server, so Claude Desktop, IDEs or other agents can use the sandbox directly.

The web app shows the agent's trace live: every tool call, failed query, repair and model escalation, then the answer with a table, a chart and the final SQL.

## Evaluation

**Benchmark:** BIRD Mini-Dev, SQLite version (500 questions, 11 databases, three difficulty levels). The demo database is never used for any number.

**Setups compared:**

| Setup | Description |
|---|---|
| `one_shot` | The 7B model writes the SQL in one call. No tools, no repair. |
| `agent` | Full agent loop on the 7B model. |
| `routed` | Same agent, starting on the 3B model, escalating to 7B when needed. |

**Fairness rules:**
- All setups get the same question, the same evidence hint and the same schema text: table definitions plus 2 example rows per table. The agent's only advantage is its tools and self-repair.
- `one_shot` and `agent` use the same model, so the difference measures the agent loop, not the model.
- Temperature is 0 with a fixed seed. The models and the run date are recorded.
- **Efficiency is measured in tokens and time per question** (local models have no per-call price).
- Prompts were not tuned on the 500 test questions.

**Scoring** follows BIRD's official `evaluation_ex.py`:
- A prediction is correct if `set(predicted_rows) == set(reference_rows)`, so row order and duplicates are ignored.
- Errors, blocked queries and timeouts (30 s) count as wrong.
- API failures count as wrong and are reported separately.

**Diagnostics** are reported but never counted as correct:
- **Near-misses:** the right rows with extra columns, or a rounding difference.
- **Empty reference:** questions whose reference answer is empty.

**Failure categories:** every wrong answer is tagged `wrong_logic`, `wrong_tables`, `wrong_columns`, `empty_result`, `sql_error`, `timeout`, `blocked`, `no_sql` or `api_error`.

**Results:** see [`results/summary.md`](results/summary.md) and the *Benchmark* tab of the app. Per-question logs for every setup (`results/*.jsonl`) are published too. They include each SQL attempt, error, cost and the agent's trace.

## Safety

Every query, from the agent, the MCP server or the benchmark, runs through `SafeDB`, which stacks independent protections:

1. **Read-only file mode:** the file is opened with SQLite `mode=ro`.
2. **`PRAGMA query_only = ON`:** the connection itself refuses writes.
3. **Engine-level permission filter:** an SQLite authorizer allows only read operations, denying INSERT, UPDATE, DELETE, DROP, CREATE, ATTACH and setting changes inside the engine.
4. **One statement, SELECT only:** queries are parsed with `sqlglot`, so `SELECT 1; DROP TABLE x` is rejected.
5. **Limits:** a 30-second time limit and a row cap.
6. **OS level, in Docker:** the database files are read-only and the app runs as an unprivileged user.

`tests/test_safety.py` attacks the sandbox with 16 write and injection attempts. It checks that each one is blocked and that the database file is byte-for-byte unchanged, including with the text check switched off, so the engine layers are tested on their own.

## Run it

Requires Python 3.11+ and [Ollama](https://ollama.com). The defaults fit a 6 GB GPU. Install the models once:

```bash
ollama pull qwen2.5-coder:7b
ollama pull qwen2.5-coder:3b
```

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
copy .env.example .env            # macOS/Linux: cp .env.example .env
```

**Web app** (demo database: a European online shop with customers, orders, deliveries and reviews):

```bash
uvicorn querypilot.server:app --port 8000      # open http://localhost:8000
```

If Ollama isn't running, the app starts in a clearly labelled offline mode that replays a placeholder run.

**Benchmark:**

```bash
python scripts/get_bird.py                      # downloads BIRD Mini-Dev into data/minidev/
python -m querypilot.evaluate --limit 20        # quick check
python -m querypilot.evaluate                   # full run: 500 questions × 3 setups
```

- **Time check:** after 20 runs, the runner prints the projected total time. Setups run one after another, so a small GPU keeps a single model loaded.
- **Resuming:** an interrupted run resumes where it stopped.
- **Offline dry run:** `--fake` tests the pipeline without a model.
- **Paid API option:** set `QP_PROVIDER=mistral` and `MISTRAL_API_KEY` in `.env` to use Mistral's API instead.

**MCP server:**

```bash
QP_DB=data/demo/shop.sqlite python -m querypilot.mcp_server
```

**Tests:**

```bash
python -m pytest -q
```

**Docker:**

```bash
docker build -t querypilot .
docker run -p 8000:8000 --env-file .env querypilot
```

## Project layout

```
querypilot/
  safe_db.py     read-only sandbox (all layers above)
  agent.py       LangGraph agent, self-repair, model routing, one-shot baseline
  llm.py         Ollama client (default) and Mistral client, token + time tracking, fake model for tests
  scoring.py     official BIRD execution accuracy + diagnostics + failure categories
  evaluate.py    benchmark runner (resumable, cost check, summary)
  server.py      FastAPI app, live agent trace over server-sent events, rate limit
  mcp_server.py  the same tools over the Model Context Protocol
  demo.py        demo shop database generator
web/index.html   the interface
tests/           safety, scoring, agent loop and runner tests
```

## Limits

- **Scope of the numbers:** BIRD Mini-Dev measures text-to-SQL on its own databases. The numbers don't promise the same accuracy on any company's data.
- **Demo data:** the demo shop is synthetic (generated with a fixed seed).
- **Model size:** results are for small local models (3B/7B). Larger models would score higher; the comparison between setups is what this project measures.
