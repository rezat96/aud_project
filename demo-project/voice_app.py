from __future__ import annotations

import base64
import json
import os
import tempfile
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import Any

from flask import Flask, jsonify, request

from agent import DecisionError
from local_stt import transcribe_audio_file
from scheduler import utc_now
from session_manager import SessionManager

app = Flask(__name__, static_folder="static")
app.config["JSON_SORT_KEYS"] = False

_session_lock = Lock()
_session_store: dict[str, Any] = {}
_manager = SessionManager()


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/api/session", methods=["POST"])
def create_session():
    session_id = request.json.get("session_id") if request.is_json else None
    with _session_lock:
        if session_id is None:
            session_id = "demo-session"
        if session_id not in _session_store:
            _session_store[session_id] = _manager.get_or_create(session_id)
            app.logger.info("session_created session_id=%s", session_id)
        else:
            app.logger.info("session_reused session_id=%s", session_id)
        return jsonify({"session_id": session_id})


@app.route("/api/session", methods=["DELETE"])
def delete_session():
    payload = request.get_json(silent=True) or {}
    session_id = payload.get("session_id") or "demo-session"
    with _session_lock:
        removed = _manager.delete_session(session_id)
        _session_store.pop(session_id, None)
    app.logger.info("session_deleted session_id=%s existed=%s", session_id, removed)
    return jsonify({"session_id": session_id, "deleted": removed})


@app.route("/api/transcribe", methods=["POST"])
def transcribe():
    payload = request.get_json(silent=True) or {}
    audio_base64 = payload.get("audio_base64")
    if not audio_base64:
        return jsonify({"error": "missing audio_base64"}), 400

    audio_bytes = base64.b64decode(audio_base64)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(audio_bytes)
        try:
            transcript, metrics, _ = transcribe_audio_file(temp_path, model="small", language="en") #fr
        except Exception as exc:  # pragma: no cover - surfaced to browser for operator feedback
            app.logger.exception("transcription_failed session_id=%s", payload.get("session_id") or "demo-session")
            return jsonify({"error": f"transcription failed: {exc}"}), 500
        session_id = payload.get("session_id") or "demo-session"
        app.logger.info(
            "transcription_completed session_id=%s characters=%d language=%s duration_seconds=%.2f",
            session_id,
            len(transcript),
            metrics["language"],
            metrics["transcription_seconds"],
        )
        return jsonify({"transcript": transcript})
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)


@app.route("/api/turn", methods=["POST"])
def process_turn():
    payload = request.get_json(silent=True) or {}
    session_id = payload.get("session_id") or "demo-session"
    transcript = (payload.get("transcript") or "").strip()
    if not transcript:
        return jsonify({"error": "empty transcript"}), 400

    if os.environ.get("DEMO_ALLOW_PAID_CALLS") != "1":
        return jsonify({"error": "Paid calls disabled. Set DEMO_ALLOW_PAID_CALLS=1 and OPENAI_API_KEY before routing browser speech to the scheduler agent."}), 403

    with _session_lock:
        session = _session_store.setdefault(session_id, _manager.get_or_create(session_id))
        turn_started = monotonic()
        app.logger.info("agent_interaction session_id=%s user=%r", session_id, transcript)
        result = session.process_turn(transcript, now=utc_now())
        app.logger.info(
            "agent_interaction session_id=%s event=%s service=%s slot_id=%s booking_id=%s reply=%r duration_seconds=%.2f",
            session_id,
            result.get("event", "unknown"),
            result.get("service", ""),
            result.get("slot_id", ""),
            result.get("booking_id", ""),
            result.get("reply", ""),
            monotonic() - turn_started,
        )
        return jsonify(result)


if __name__ == "__main__":
    cert_path = Path(__file__).resolve().parent / "certs" / "cert.pem"
    key_path = Path(__file__).resolve().parent / "certs" / "key.pem"
    if cert_path.exists() and key_path.exists():
        app.run(host="127.0.0.1", port=5000, debug=True, ssl_context=(str(cert_path), str(key_path)))
    else:
        app.run(host="127.0.0.1", port=5000, debug=True)
