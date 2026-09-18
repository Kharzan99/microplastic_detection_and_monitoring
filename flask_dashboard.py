"""AquaTrace — Flask dashboard server.

Session behavior:
- When this server starts, the dashboard is RESET to zero.
- Firebase history is NEVER loaded on startup.
- A new run of main_demo.py publishes an active run ID through a local
  state file; Flask then serves only Firebase records belonging to that run.
- No demo/fallback readings are used.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from flask import Flask, jsonify, send_from_directory
import firebase_admin
from firebase_admin import credentials, db

PROJECT_ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
FIREBASE_KEY = PROJECT_ROOT / "models" / "firebase_key.json"
FIREBASE_DB_URL = (
    "https://aquatrace-63edb-default-rtdb.asia-southeast1.firebasedatabase.app/"
)
STATE_FILE = PROJECT_ROOT / ".aquatrace_dashboard_state.json"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")


def reset_dashboard_state() -> dict:
    """Every Flask launch creates a brand-new dashboard session.

    Old Firebase runs can never appear in this session. main_demo.py must
    explicitly handshake with this session before its results are shown.
    """
    state = {
        "dashboard_session_id": f"dash_{uuid.uuid4().hex}",
        "active_run_id": None,
        "run_mode": None,
        "run_started_timestamp": None,
        "run_started_human": None,
    }
    tmp = STATE_FILE.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)
    tmp.replace(STATE_FILE)
    return state


def _load_state() -> dict:
    if not STATE_FILE.exists():
        return {"dashboard_session_id": None, "active_run_id": None}
    try:
        with STATE_FILE.open("r", encoding="utf-8") as fh:
            value = json.load(fh)
        return value if isinstance(value, dict) else {"dashboard_session_id": None, "active_run_id": None}
    except Exception:
        return {"dashboard_session_id": None, "active_run_id": None}


def init_firebase() -> None:
    if firebase_admin._apps:
        return
    if not FIREBASE_KEY.exists():
        raise FileNotFoundError(
            f"Firebase service account key not found: {FIREBASE_KEY}"
        )
    cred = credentials.Certificate(str(FIREBASE_KEY))
    firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})


def _load_records() -> list[dict]:
    payload = db.reference("sensor_readings").get()
    payload = payload if isinstance(payload, dict) else {}
    records = []
    for key, value in payload.items():
        if isinstance(value, dict):
            record = dict(value)
            record["_key"] = key
            records.append(record)
    records.sort(key=lambda x: float(x.get("timestamp") or 0), reverse=True)
    return records


@app.get("/api/health")
def health():
    try:
        init_firebase()
        state = _load_state()
        active = state.get("active_run_id")
        dashboard_session_id = state.get("dashboard_session_id")
        db.reference("sensor_readings").limit_to_last(1).get()
        return jsonify({
            "ok": True,
            "firebase": "connected",
            "active_run_id": active,
            "dashboard_session_id": dashboard_session_id,
        })
    except Exception as exc:
        return jsonify({
            "ok": False,
            "firebase": "unavailable",
            "error": str(exc).split("\n")[0][:240],
        }), 503


@app.get("/api/readings")
def readings():
    try:
        state = _load_state()
        active_id = state.get("active_run_id")
        dashboard_session_id = state.get("dashboard_session_id")

        # Critical rule: if Flask was just started, show ZERO.
        # Do not even expose historical Firebase records to the browser.
        if not active_id:
            response = jsonify({
                "ok": True,
                "count": 0,
                "readings": [],
                "current_run_id": None,
                "dashboard_session_id": dashboard_session_id,
                "current_run_count": 0,
                "current_run_mode": None,
                "current_run_started_human": None,
            })
            response.headers["Cache-Control"] = "no-store, max-age=0"
            return response

        init_firebase()
        all_records = _load_records()
        current_records = [
            r for r in all_records
            if (
                r.get("analysis_run_id") == active_id
                and r.get("dashboard_session_id") == dashboard_session_id
            )
        ]

        response = jsonify({
            "ok": True,
            "count": len(current_records),
            "readings": current_records,
            "current_run_id": active_id,
            "dashboard_session_id": dashboard_session_id,
            "current_run_count": len(current_records),
            "current_run_mode": state.get("run_mode"),
            "current_run_started_human": state.get("run_started_human"),
        })
        response.headers["Cache-Control"] = "no-store, max-age=0"
        return response
    except Exception as exc:
        return jsonify({
            "ok": False,
            "error": str(exc).split("\n")[0][:500],
            "readings": [],
            "current_run_id": None,
            "dashboard_session_id": _load_state().get("dashboard_session_id"),
            "current_run_count": 0,
        }), 503


@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/<path:path>")
def assets(path: str):
    file_path = FRONTEND_DIR / path
    if file_path.exists() and file_path.is_file():
        return send_from_directory(FRONTEND_DIR, path)
    return send_from_directory(FRONTEND_DIR, "index.html")


if __name__ == "__main__":
    # Reset BEFORE serving the first request. This guarantees a fresh dashboard
    # every time flask_dashboard.py is launched.
    print("\nAquaTrace Dashboard")
    print("-------------------")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Firebase key : {FIREBASE_KEY}")
    print("Dashboard    : http://127.0.0.1:5000")
    print("Data source  : Firebase RTDB /sensor_readings")
    fresh = reset_dashboard_state()
    print(f"Dashboard session: {fresh['dashboard_session_id']}")
    print("Session      : EMPTY on Flask start; populated only by a new main_demo.py run")
    print("Data mode    : REAL DATA ONLY — no demo/fallback readings\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
