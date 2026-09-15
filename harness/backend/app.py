"""FastAPI app wiring model registry, library import, and inference."""

from __future__ import annotations

import json
import logging
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import DATA_DIR

from . import db, inference
from .library_import import parse_apple_library
from .matching import MATCH_THRESHOLD, match_many
from .models_registry import scan_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
)
logger = logging.getLogger("harness")

HARNESS_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = HARNESS_ROOT / "frontend"
UPLOAD_DIR = Path(DATA_DIR) / "harness_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Melody Matcher Harness", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    db.init_db()
    # Pre-flight: tell us at boot if any checkpoint is broken, instead of at
    # first click. Cheap — just torch.load + key checks, no instantiation.
    try:
        errors = inference.validate_all_models()
        if errors:
            for mid, msg in errors.items():
                logger.warning("startup: model %s broken — %s", mid, msg)
    except Exception as exc:
        logger.warning("startup: model validation skipped (%s)", exc)


# ---------------------------------------------------------------------------
# Pre-warm: run inference for a list of model_ids in the background so that
# subsequent /api/runs requests for them are instant. Single-user app, so a
# single global lock + state dict is fine.
# ---------------------------------------------------------------------------

_PREWARM_LOCK = threading.Lock()
_PREWARM_STATE: Dict[str, Any] = {
    "running": False,
    "library_id": None,
    "total": 0,
    "done": 0,
    "current": None,         # model_id currently being warmed
    "current_name": None,
    "completed_ids": [],     # finished successfully
    "errors": {},            # {model_id: error_msg}
    "started_at": None,
    "finished_at": None,
}


def _prewarm_worker(library_id: int, model_ids: List[str]) -> None:
    models = {m.id: m for m in scan_models()}
    try:
        for mid in model_ids:
            with _PREWARM_LOCK:
                _PREWARM_STATE["current"] = mid
                _PREWARM_STATE["current_name"] = (
                    models[mid].theory_tag if mid in models else mid
                )
            try:
                inference.run_inference(mid, library_id, top_k=50)
                with _PREWARM_LOCK:
                    _PREWARM_STATE["completed_ids"].append(mid)
            except Exception as exc:
                logger.warning("prewarm failed for %s: %s", mid, exc)
                with _PREWARM_LOCK:
                    _PREWARM_STATE["errors"][mid] = f"{type(exc).__name__}: {exc}"
            with _PREWARM_LOCK:
                _PREWARM_STATE["done"] += 1
    finally:
        with _PREWARM_LOCK:
            _PREWARM_STATE["running"] = False
            _PREWARM_STATE["current"] = None
            _PREWARM_STATE["current_name"] = None
            _PREWARM_STATE["finished_at"] = datetime.now(timezone.utc).isoformat()


class PrewarmRequest(BaseModel):
    library_id: int
    model_ids: List[str]


@app.post("/api/runs/prewarm")
def start_prewarm(req: PrewarmRequest) -> Dict[str, Any]:
    if not req.model_ids:
        raise HTTPException(400, "model_ids must be non-empty")
    with _PREWARM_LOCK:
        if _PREWARM_STATE["running"]:
            raise HTTPException(409, "prewarm already running")
        _PREWARM_STATE.update({
            "running": True,
            "library_id": req.library_id,
            "total": len(req.model_ids),
            "done": 0,
            "current": None,
            "current_name": None,
            "completed_ids": [],
            "errors": {},
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
        })
    threading.Thread(
        target=_prewarm_worker,
        args=(req.library_id, list(req.model_ids)),
        daemon=True,
    ).start()
    return {"status": "started", "total": len(req.model_ids)}


@app.get("/api/runs/prewarm/status")
def prewarm_status() -> Dict[str, Any]:
    with _PREWARM_LOCK:
        return dict(_PREWARM_STATE)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@app.get("/api/models")
def list_models() -> Dict[str, Any]:
    records = scan_models()
    return {"models": [r.to_dict() for r in records]}


# ---------------------------------------------------------------------------
# Library import
# ---------------------------------------------------------------------------

@app.post("/api/library/import")
async def import_library(
    file: UploadFile = File(...),
    match_policy: str = Form("aggressive"),
    display_name: Optional[str] = Form(None),
) -> Dict[str, Any]:
    # Save upload
    ext = ".xml"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    saved = UPLOAD_DIR / f"library_{stamp}_{uuid.uuid4().hex[:8]}{ext}"
    with open(saved, "wb") as out:
        shutil.copyfileobj(file.file, out)

    # Parse
    try:
        imported = parse_apple_library(saved)
    except Exception as exc:
        raise HTTPException(400, f"failed to parse library XML: {exc}")
    if not imported:
        raise HTTPException(400, "no tracks found in library XML")

    # Match
    matches = match_many(imported)
    matched_count = sum(1 for _, r in matches if r.matched)

    # Persist
    with db.get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO harness_libraries
               (created_at, source, match_policy, total_count, matched_count,
                raw_xml_path, display_name)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                "apple_music_xml",
                match_policy,
                len(imported),
                matched_count,
                str(saved),
                display_name or file.filename or saved.name,
            ),
        )
        library_id = cur.lastrowid
        conn.executemany(
            """INSERT OR REPLACE INTO harness_library_tracks
               (library_id, apple_track_id, artist, title, album, year, genre,
                duration_ms, play_count, embedding_index, embedding_key, match_score)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            [
                (
                    library_id, t.apple_track_id, t.artist, t.title, t.album,
                    t.year, t.genre, t.duration_ms, t.play_count,
                    r.embedding_index, r.embedding_key, r.match_score,
                ) for t, r in matches
            ],
        )
        conn.commit()

    sample_misses = [
        {"artist": t.artist, "title": t.title, "play_count": t.play_count}
        for t, r in matches if not r.matched
    ][:20]

    return {
        "library_id": library_id,
        "total": len(imported),
        "matched": matched_count,
        "match_threshold": MATCH_THRESHOLD,
        "sample_misses": sample_misses,
    }


@app.get("/api/library/{library_id}")
def get_library(library_id: int) -> Dict[str, Any]:
    with db.get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM harness_libraries WHERE id = ?", (library_id,)
        ).fetchone()
    if row is None:
        raise HTTPException(404, "library not found")
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "source": row["source"],
        "match_policy": row["match_policy"],
        "trackCount": row["total_count"],
        "matchedCount": row["matched_count"],
        "topCount": row["matched_count"],
        "display_name": row["display_name"],
        "user": "you",
    }


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

class RunRequest(BaseModel):
    model_id: str
    library_id: int
    top_k: int = 50
    # Hurdle-only knobs; ignored for scoring models.
    ranking_mode: str = "p_played"   # "p_played" | "joint" | "threshold"
    threshold: float = 0.5


_RMODE_LABEL = {
    "p_played":  "P(played)",
    "joint":     "joint",
    "threshold": "threshold",
}


def _run_to_history(row) -> Dict[str, Any]:
    feedback = {"up": row["up_count"] or 0, "down": row["down_count"] or 0}
    row_keys = set(row.keys())
    rmode = row["ranking_mode"] if "ranking_mode" in row_keys else None
    model_label = row["model_name"]
    if rmode:
        model_label = f"{model_label} · {_RMODE_LABEL.get(rmode, rmode)}"
    return {
        "id": row["id"],
        "model": model_label,
        "modelId": row["model_id"],
        "at": _relative_time(row["created_at"]),
        "seeds": row["library_matched"] or 0,
        "feedback": feedback,
        "ranking_mode": rmode,
    }


def _relative_time(iso: str) -> str:
    try:
        dt = datetime.fromisoformat(iso)
    except Exception:
        return iso
    now = datetime.now(timezone.utc)
    delta = now - dt
    if delta.total_seconds() < 60:
        return "Just now"
    mins = int(delta.total_seconds() // 60)
    if mins < 60:
        return f"{mins}m ago"
    hours = mins // 60
    if hours < 24:
        return f"{hours}h ago"
    return dt.strftime("%b %d, %H:%M")


@app.post("/api/runs")
def create_run(req: RunRequest) -> Dict[str, Any]:
    models = {m.id: m for m in scan_models()}
    record = models.get(req.model_id)
    if record is None:
        raise HTTPException(400, f"unknown model_id: {req.model_id}")

    try:
        payload = inference.run_inference(
            req.model_id, req.library_id, req.top_k,
            ranking_mode=req.ranking_mode, threshold=req.threshold,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    # Persist ranking_mode/threshold only for hurdle runs (NULL for scoring).
    is_hurdle = record.model_type == "hurdle"
    rmode = req.ranking_mode if is_hurdle else None
    rthr = req.threshold if is_hurdle else None

    run_id = f"run-{uuid.uuid4().hex[:8]}"
    now = datetime.now(timezone.utc).isoformat()
    with db.get_connection() as conn:
        lib = conn.execute(
            "SELECT matched_count FROM harness_libraries WHERE id = ?",
            (req.library_id,),
        ).fetchone()
        seeds = lib["matched_count"] if lib else 0
        conn.execute(
            """INSERT INTO harness_runs (id, model_id, model_name, model_path,
               library_id, created_at, latency_ms, results_json,
               ranking_mode, threshold)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                run_id, record.id, record.name, record.path,
                req.library_id, now, payload["latency_ms"],
                json.dumps(payload["results"]),
                rmode, rthr,
            ),
        )
        conn.commit()

    run = _build_run_response(
        run_id=run_id, record=record, created_at=now,
        latency_ms=payload["latency_ms"], seeds=seeds,
        results=payload["results"], feedback={"up": 0, "down": 0},
        ranking_mode=rmode, threshold=rthr,
    )
    return {"run": run}


def _build_run_response(run_id, record, created_at, latency_ms, seeds,
                        results, feedback,
                        ranking_mode=None, threshold=None) -> Dict[str, Any]:
    return {
        "id": run_id,
        "model": record.name,
        "modelId": record.id,
        "modelStatus": record.status,
        "modelType": getattr(record, "model_type", "scoring"),
        "at": _relative_time(created_at),
        "created_at": created_at,
        "seeds": seeds,
        "latencyMs": latency_ms,
        "feedback": feedback,
        "results": results,
        "ranking_mode": ranking_mode,
        "threshold": threshold,
    }


@app.get("/api/runs")
def list_runs() -> Dict[str, Any]:
    with db.get_connection() as conn:
        rows = conn.execute(
            """SELECT r.*, l.matched_count as library_matched,
                      (SELECT COUNT(*) FROM harness_feedback f
                       WHERE f.run_id = r.id AND f.vote = 'up')   as up_count,
                      (SELECT COUNT(*) FROM harness_feedback f
                       WHERE f.run_id = r.id AND f.vote = 'down') as down_count
               FROM harness_runs r
               LEFT JOIN harness_libraries l ON l.id = r.library_id
               ORDER BY r.created_at DESC LIMIT 200"""
        ).fetchall()
    return {"runs": [_run_to_history(r) for r in rows]}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> Dict[str, Any]:
    with db.get_connection() as conn:
        row = conn.execute(
            """SELECT r.*, l.matched_count as library_matched,
                      (SELECT COUNT(*) FROM harness_feedback f
                       WHERE f.run_id = r.id AND f.vote = 'up')   as up_count,
                      (SELECT COUNT(*) FROM harness_feedback f
                       WHERE f.run_id = r.id AND f.vote = 'down') as down_count
               FROM harness_runs r
               LEFT JOIN harness_libraries l ON l.id = r.library_id
               WHERE r.id = ?""",
            (run_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(404, "run not found")
        votes = conn.execute(
            "SELECT track_key, vote FROM harness_feedback WHERE run_id = ?",
            (run_id,),
        ).fetchall()
    models = {m.id: m for m in scan_models()}
    record = models.get(row["model_id"])
    results = json.loads(row["results_json"])
    vote_map = {v["track_key"]: v["vote"] for v in votes}
    for r in results:
        v = vote_map.get(r["embedding_key"])
        if v:
            r["vote"] = v

    # SQLite Row supports .keys(); use it to guard against historical rows
    # missing the new columns.
    row_keys = set(row.keys())
    rmode = row["ranking_mode"] if "ranking_mode" in row_keys else None
    rthr = row["threshold"] if "threshold" in row_keys else None

    run = _build_run_response(
        run_id=row["id"],
        record=record or _fallback_record(row),
        created_at=row["created_at"],
        latency_ms=row["latency_ms"],
        seeds=row["library_matched"] or 0,
        results=results,
        feedback={"up": row["up_count"] or 0, "down": row["down_count"] or 0},
        ranking_mode=rmode, threshold=rthr,
    )
    return {"run": run}


def _fallback_record(row):
    class _R:
        id = row["model_id"]; name = row["model_name"]
        status = "unknown"; path = row["model_path"]
        model_type = "scoring"
    return _R()


class FeedbackBody(BaseModel):
    track_key: str
    vote: Optional[str]  # "up" | "down" | None


@app.post("/api/runs/{run_id}/feedback")
def post_feedback(run_id: str, body: FeedbackBody) -> Dict[str, Any]:
    if body.vote not in ("up", "down", None):
        raise HTTPException(400, "vote must be 'up', 'down', or null")
    with db.get_connection() as conn:
        if body.vote is None:
            conn.execute(
                "DELETE FROM harness_feedback WHERE run_id = ? AND track_key = ?",
                (run_id, body.track_key),
            )
        else:
            conn.execute(
                """INSERT INTO harness_feedback (run_id, track_key, vote, created_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(run_id, track_key) DO UPDATE SET vote = excluded.vote""",
                (run_id, body.track_key, body.vote,
                 datetime.now(timezone.utc).isoformat()),
            )
        up = conn.execute(
            "SELECT COUNT(*) c FROM harness_feedback WHERE run_id=? AND vote='up'",
            (run_id,),
        ).fetchone()["c"]
        down = conn.execute(
            "SELECT COUNT(*) c FROM harness_feedback WHERE run_id=? AND vote='down'",
            (run_id,),
        ).fetchone()["c"]
        conn.commit()
    return {"up": up, "down": down}


@app.get("/api/runs/{run_id}/export")
def export_run(run_id: str):
    with db.get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM harness_runs WHERE id = ?", (run_id,)
        ).fetchone()
    if row is None:
        raise HTTPException(404, "run not found")
    payload = {
        "run_id": row["id"],
        "model_id": row["model_id"],
        "model_name": row["model_name"],
        "library_id": row["library_id"],
        "created_at": row["created_at"],
        "latency_ms": row["latency_ms"],
        "results": json.loads(row["results_json"]),
    }
    headers = {"Content-Disposition": f'attachment; filename="{run_id}.json"'}
    return JSONResponse(payload, headers=headers)


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------

app.mount("/src", StaticFiles(directory=str(FRONTEND_DIR / "src")), name="src")


@app.get("/")
def root() -> FileResponse:
    index = FRONTEND_DIR / "index.html"
    return FileResponse(str(index))
