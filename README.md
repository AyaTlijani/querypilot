# QueryPilot

**An autonomous data analyst agent.** Ask a business question in plain language; QueryPilot explores the database, writes SQL, runs it, checks whether the result makes sense, and repairs its own mistakes. Every step is visible, and every query is read-only.

It is evaluated on **BIRD Mini-Dev**, a public text-to-SQL benchmark over real-world databases, comparing three setups on accuracy, tokens and time.

`LangGraph` · `Ollama` · `Qwen2.5-Coder` · `MCP` · `FastAPI` · `SQLite` · `Docker`

It runs **entirely locally on open-weight models**: no API cost, and the data never leaves the machine.

---

## What it does

```
question ─▶ draft ─▶ execute ─┬─ rows returned ───────────────────────────────────▶ answer
                              └─ error / 0 rows ─▶ diagnose (database tools) ─▶ revise ─┐
                                        ▲                                               │
                                        └──────────────── execute ◀─────────────────────┘
```

- **Execution-guided.** The query is drafted exactly as a one-shot model would write it, then run. A query that returns rows is kept as it is, so a working answer is never rewritten.
- **Diagnoses failures with database tools.** On an error or an empty result, the workflow gathers evidence: similarly named columns for "no such column", the real table list for "no such table", the actual values of a filtered column when a query returns 0 rows, and sample rows. The model then revises the query from that evidence, up to 3 times. The workflow runs the tools itself, because small local models are unreliable at choosing tools on their own.

- **Read-only by construction** (see [Safety](#safety)).
- **Speaks MCP.** The same tools are exposed as a Model Context Protocol server, so AI assistants, IDEs or other agents can use the sandbox directly.

- **Works on your own data.** Upload a CSV or Excel file and it becomes a read-only SQLite database:
  - Each Excel sheet becomes a table.
  - Delimiters, encodings and French decimal commas are detected.
  - Column names are cleaned into safe SQL identifiers.
  - Starter questions are suggested from your columns.
  - Uploads are deleted after 24 hours.

The web app shows the agent's trace live: every diagnostic, failed query and repair, then the answer with a table, a chart and the final SQL.

## Evaluation

**Benchmark:** BIRD Mini-Dev, SQLite version (500 questions, 11 databases, three difficulty levels; a fixed 200-question sample was run). The demo database is never used for any number.

**Setups compared:**

| Setup | Description |
|---|---|
| `one_shot` | The 7B model writes the SQL in one call. No tools, no repair. |
| `agent` | Draft, execute, diagnose and revise, all on the 7B model. |
| `routed` | Same workflow: the small 1.5B model drafts, and the 7B model takes over to repair. |

**Fairness rules:**
- All setups get the same question, the same evidence hint and the same schema text: table definitions plus 2 example rows per table. The agent's only advantage is its tools and self-repair.
- `one_shot` and `agent` use the same model, so the difference measures the agent loop, not the model.
- Temperature is 0 with a fixed seed. The models and the run date are recorded.
- **Efficiency is measured in tokens and time per question** (local models have no per-call price).
- Prompts were not tuned on the test questions.

**Design changes, stated openly.** Two earlier designs were tried in pilot runs on test questions:
1. An "explore first" agent rewrote correct answers into wrong ones.
2. A version where the model chose its own tools to repair failures fixed 0 of the first 7 failed queries: the 7B model wandered through exploration steps without ever submitting a fix.

The final design keeps working answers and runs the diagnostics itself. These are general rules, not tuned to any specific question. All reported agent numbers come from a fresh run of the final design.

**Scoring** follows BIRD's official `evaluation_ex.py`:
- A prediction is correct if `set(predicted_rows) == set(reference_rows)`, so row order and duplicates are ignored.
- Errors, blocked queries and timeouts (30 s) count as wrong.
- API failures count as wrong and are reported separately.

**Diagnostics** are reported but never counted as correct:
- **Near-misses:** the right rows with extra columns, or a rounding difference.
- **Empty reference:** questions whose reference answer is empty.

**Failure categories:** every wrong answer is tagged `wrong_logic`, `wrong_tables`, `wrong_columns`, `empty_result`, `sql_error`, `timeout`, `blocked`, `no_sql` or `api_error`.

## Results

200 questions, a fixed random sample of BIRD Mini-Dev (seed 42) covering all 11 databases. Every setup answered the same 200 questions.

**Local: Qwen2.5-Coder 7B on a laptop GPU (RTX 3050, 6 GB), via Ollama**

| Setup | Execution accuracy | SQL errors left | Median time |
|---|---|---|---|
| One-shot | 47.0% | 27 | 14 s |
| **Agent** | **51.0%** (+4.0 pts) | 10 | 13 s |
| Routed (1.5B drafts, 7B repairs) | 40.0% | 21 | 22 s |

- **The agent fixed 8 questions one-shot got wrong and broke none.** The model is deterministic here: all 200 drafts were identical to the one-shot answers, so the +4.0 points come entirely from the repair loop. 8 fixes against 0 regressions is unlikely to be chance (sign test, p < 0.01).
- **The repair loop turned 17 of 27 crashing queries into running ones,** and 8 of those into correct answers. 40 questions entered repair.
- **Routing didn't pay off.** The 1.5B model's drafts often *run* but return the wrong answer, and a running query is never sent to repair. It lost 28 answers the 7B model got right and cost more tokens. "The query executes" is too weak a signal to decide when to escalate. That's a useful negative result, and it's why the web app uses the 7B agent, not routing.

**Cloud: Gemini 3.5 Flash-Lite (free tier)**

| Setup | Execution accuracy | SQL errors left | Median time |
|---|---|---|---|
| One-shot | 62.5% | 2 | 4 s |
| Agent | 64.0% | 0 | 4 s |

- **The stronger model makes far fewer crashing queries,** so there is less for the repair loop to fix: 12 questions entered repair, and 5 were fixed.
- **Gemini isn't fully deterministic,** even at temperature 0: only 100 of the 200 agent drafts were identical to its one-shot answers. The +1.5 points are within that noise, so the reliable figure for Gemini is the repair success rate (5 of 12).

**Where both still fail:** the remaining errors are mostly *wrong logic* (right tables, wrong filter, join or aggregation) and *wrong tables*. These are queries that run fine, so execution feedback can't catch them. That's the clearest next step (see Limits).

Full tables: [`results/summary.md`](results/summary.md) (local) and [`results_gemini/summary.md`](results_gemini/summary.md) (Gemini). Per-question logs (`*.jsonl`) include every SQL attempt, diagnostic and error.

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
ollama pull qwen2.5-coder:1.5b
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
python -m querypilot.evaluate --sample 200      # the reported run: 200 questions × 3 setups
```

- **Time check:** after 20 runs, the runner prints the projected total time. Setups run one after another, so a small GPU keeps a single model loaded.
- **Resuming:** an interrupted run resumes where it stopped.
- **Offline dry run:** `--fake` tests the pipeline without a model.
- **Gemini (free tier):** set `GEMINI_API_KEY` in `.env`, then run with `QP_PROVIDER=gemini` and `--out results_gemini`. Requests are paced for the free tier, and the run stops cleanly when the daily quota is used up, so you can resume the next day.
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
docker run -p 8000:8000 --env-file .env -e OLLAMA_URL=http://host.docker.internal:11434 querypilot
```

Ollama runs on the host, so the container reaches it through `host.docker.internal` (on Linux, add `--add-host=host.docker.internal:host-gateway`). The Docker setup has not been tested end to end.

## Project layout

```
querypilot/
  safe_db.py     read-only sandbox (all layers above)
  agent.py       LangGraph agent (draft, run, diagnose, revise), one-shot baseline
  llm.py         Ollama client (default), Gemini and Mistral clients, fake model for tests
  scoring.py     official BIRD execution accuracy + diagnostics + failure categories
  evaluate.py    benchmark runner (fixed sample, resumable, summary)
  server.py      FastAPI app, live agent trace over server-sent events, rate limit
  mcp_server.py  the same tools over the Model Context Protocol
  demo.py        demo shop database generator
  uploads.py     CSV / Excel -> SQLite conversion (type inference, name cleaning, limits)
web/index.html   the interface
tests/           safety, scoring, agent loop and runner tests
```

## Limits

- **The repair loop only sees failures it can detect** (errors and empty results). Wrong-but-running queries, the largest error category, need a different signal. Self-consistency voting across several drafts, or BIRD's column descriptions in the schema, are the natural next steps.

- **Scope of the numbers:** BIRD Mini-Dev measures text-to-SQL on its own databases. The numbers don't promise the same accuracy on any company's data.
- **Demo data:** the demo shop is synthetic (generated with a fixed seed).
- **Model size:** results are for small local models (1.5B/7B) and Gemini Flash-Lite. Larger models would score higher; the comparison between setups is what this project measures.
