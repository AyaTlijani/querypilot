"""The web app: ask questions, upload files, and see benchmark results."""

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
from .uploads import UploadError, build_database, path_for, suggest_questions

load_dotenv()
ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
RESULTS = ROOT / "results" / "summary.json"
DB_PATH = Path(os.getenv("QP_DEMO_DB", ROOT / DEMO_PATH))
RATE_LIMIT = int(os.getenv("QP_RATE_LIMIT_PER_HOUR", "20"))

if not DB_PATH.exists():
    build(DB_PATH)
db = SafeDB(DB_PATH)

from .llm import LARGE, PROVIDER, SMALL, make_client

try:
    llm, MODE = make_client(), "live"
except Exception as err:
    print(f"[querypilot] Running offline: {err}")
    llm, MODE = offline_llm(), "offline"

app = FastAPI(title="QueryPilot")
app.mount("/vendor", StaticFiles(directory=WEB / "vendor"), name="vendor")
_hits: dict[str, deque] = defaultdict(deque)


class Ask(BaseModel):
    """The body of an /api/ask request."""
    question: str
    db_id: str | None = None


_upload_hits: dict[str, deque] = defaultdict(deque)
UPLOAD_LIMIT = int(os.getenv("QP_UPLOADS_PER_HOUR", "10"))


def _db_for(db_id: str | None) -> SafeDB:
    """Return the demo database or an uploaded one."""
    if not db_id:
        return db
    path = path_for(db_id)
    if path is None:
        raise HTTPException(404, "That dataset has expired. Please upload the file again.")
    return SafeDB(path)


def _tables(target: SafeDB) -> list[dict]:
    """List tables and columns for the schema panel."""
    return [{"name": t, "columns": [c for c, _ in target.describe(t)]} for t in target.tables()]


def _allowed(ip: str) -> bool:
    """Limit how many questions one visitor can ask per hour."""
    now, window = time.time(), _hits[ip]
    while window and now - window[0] > 3600:
        window.popleft()
    if len(window) >= RATE_LIMIT:
        return False
    window.append(now)
    return True


@app.get("/")
def index():
    """Serve the web page."""
    return FileResponse(WEB / "index.html")


@app.get("/api/info")
def info():
    """Return the model mode, demo tables and example questions."""
    tables = _tables(db)
    return {"mode": MODE, "provider": PROVIDER, "models": [SMALL, LARGE],
            "examples": EXAMPLES, "tables": tables}


@app.get("/api/results")
def results():
    """Return the latest benchmark summary."""
    if RESULTS.exists():
        return JSONResponse(json.loads(RESULTS.read_text(encoding="utf-8")))
    return JSONResponse({"setups": {}})


@app.post("/api/upload")
async def upload(request: Request):
    """Receive a CSV or Excel file (size-limited) and turn it into a database."""
    ip = request.client.host if request.client else "?"
    window, now = _upload_hits[ip], time.time()
    while window and now - window[0] > 3600:
        window.popleft()
    if len(window) >= UPLOAD_LIMIT:
        raise HTTPException(429, "Too many uploads this hour — try again later.")
    window.append(now)
    from urllib.parse import unquote
    filename = unquote(request.headers.get("x-filename", "upload.csv"))
    from .uploads import MAX_BYTES
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_BYTES:
        raise HTTPException(413, f"Files up to {MAX_BYTES // (1024 * 1024)} MB are supported.")
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_BYTES:
            raise HTTPException(413, f"Files up to {MAX_BYTES // (1024 * 1024)} MB are supported.")
        chunks.append(chunk)
    data = b"".join(chunks)
    try:
        info = build_database(data, filename)
    except UploadError as err:
        raise HTTPException(400, str(err))
    return {"db_id": info["db_id"], "file": info["file"], "truncated": info["truncated"],
            "tables": [{"name": t["name"], "columns": t["columns"], "rows": t["rows"]}
                       for t in info["tables"]],
            "examples": suggest_questions(info["tables"])}


@app.post("/api/ask")
def ask(body: Ask, request: Request):
    """Run the agent on a question and stream each step to the browser."""
    question = body.question.strip()[:500]
    if not question:
        raise HTTPException(400, "Empty question.")
    if not _allowed(request.client.host if request.client else "?"):
        raise HTTPException(429, "Too many questions this hour — try again later.")

    target = _db_for(body.db_id)
    events: queue.Queue = queue.Queue()

    def work():
        """Run the agent in the background and queue its events."""
        try:
            agent = Agent(llm, target, emit=events.put)
            out = agent.run(question)
            answer = ""
            if out.get("rows") and MODE == "live":
                answer = summarize(llm, question, out["columns"], out["rows"])
            elif MODE == "offline":
                answer = "Offline mode: this is a placeholder run. Start Ollama for real answers."
            events.put({"type": "answer", "text": answer, "sql": out.get("final_sql", ""),
                        "columns": out.get("columns", []), "rows": out.get("rows", [])[:200],
                        "status": out.get("status"), "cost": round(out.get("cost", 0), 5),
                        "tokens": out.get("tokens", 0),
                        "calls": out.get("calls", 0), "escalated": out.get("escalated", False),
                        "models": out.get("models_used", [])})
        except Exception as err:
            events.put({"type": "error", "text": str(err)})
        finally:
            events.put(None)

    threading.Thread(target=work, daemon=True).start()

    def stream():
        """Send queued events to the browser as they happen."""
        while True:
            event = events.get()
            if event is None:
                break
            yield f"data: {json.dumps(event, default=str)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")
