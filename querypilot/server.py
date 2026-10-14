"""Web app: ask questions about the demo database and watch the agent work live.

    uvicorn querypilot.server:app --port 8000
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agent import Agent, summarize
from .demo import DEMO_PATH, EXAMPLES, build, offline_llm
from .safe_db import SafeDB

load_dotenv()
ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
RESULTS = ROOT / "results" / "summary.json"
DB_PATH = Path(os.getenv("QP_DEMO_DB", ROOT / DEMO_PATH))
RATE_LIMIT = int(os.getenv("QP_RATE_LIMIT_PER_HOUR", "20"))

if not DB_PATH.exists():
    build(DB_PATH)
db = SafeDB(DB_PATH)

if os.getenv("MISTRAL_API_KEY"):
    from .llm import MistralClient
    llm, MODE = MistralClient(), "live"
else:
    llm, MODE = offline_llm(), "offline"

app = FastAPI(title="QueryPilot")
app.mount("/vendor", StaticFiles(directory=WEB / "vendor"), name="vendor")
_hits: dict[str, deque] = defaultdict(deque)


class Ask(BaseModel):
    question: str


def _allowed(ip: str) -> bool:
    now, window = time.time(), _hits[ip]
    while window and now - window[0] > 3600:
        window.popleft()
    if len(window) >= RATE_LIMIT:
        return False
    window.append(now)
    return True


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/info")
def info():
    tables = [{"name": t, "columns": [c for c, _ in db.describe(t)]} for t in db.tables()]
    return {"mode": MODE, "examples": EXAMPLES, "tables": tables}


@app.get("/api/results")
def results():
    if RESULTS.exists():
        return JSONResponse(json.loads(RESULTS.read_text(encoding="utf-8")))
    return JSONResponse({"setups": {}})


@app.post("/api/ask")
def ask(body: Ask, request: Request):
    question = body.question.strip()[:500]
    if not question:
        raise HTTPException(400, "Empty question.")
    if not _allowed(request.client.host if request.client else "?"):
        raise HTTPException(429, "Too many questions this hour — try again later.")

    events: queue.Queue = queue.Queue()

    def work():
        try:
            agent = Agent(llm, db, routed=True, emit=events.put)
            out = agent.run(question)
            answer = ""
            if out.get("rows") and MODE == "live":
                answer = summarize(llm, question, out["columns"], out["rows"])
            elif MODE == "offline":
                answer = "Offline mode: this is a placeholder run. Add a Mistral API key for real answers."
            events.put({"type": "answer", "text": answer, "sql": out.get("final_sql", ""),
                        "columns": out.get("columns", []), "rows": out.get("rows", [])[:200],
                        "status": out.get("status"), "cost": round(out.get("cost", 0), 5),
                        "calls": out.get("calls", 0), "escalated": out.get("escalated", False),
                        "models": out.get("models_used", [])})
        except Exception as err:
            events.put({"type": "error", "text": str(err)})
        finally:
            events.put(None)

    threading.Thread(target=work, daemon=True).start()

    def stream():
        while True:
            event = events.get()
            if event is None:
                break
            yield f"data: {json.dumps(event, default=str)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")
