from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable
from uuid import uuid4

from agent import TextAgent
from scheduler import Scheduler, utc_now


@dataclass
class ConversationSession:
    session_id: str
    scheduler: Scheduler
    agent: TextAgent
    created_at: datetime
    last_activity: datetime

    def process_turn(self, transcript: str, now: datetime | None = None) -> dict:
        event_time = now or utc_now()
        result = self.agent.turn(transcript, event_time)
        self.last_activity = event_time
        return result


class SessionManager:
    def __init__(self, decide_factory: Callable[[], object] | None = None):
        self.decide_factory = decide_factory or self._default_decide_factory
        self._sessions: dict[str, ConversationSession] = {}
        self._temp_dirs: dict[str, TemporaryDirectory[str]] = {}

    @staticmethod
    def _default_decide_factory() -> object:
        from model import OpenAIRouter

        return OpenAIRouter()

    def create_session(self, session_id: str | None = None) -> ConversationSession:
        session_key = session_id or uuid4().hex
        if session_key in self._sessions:
            return self._sessions[session_key]

        temp_dir = TemporaryDirectory(prefix="voice-session-")
        self._temp_dirs[session_key] = temp_dir
        db_path = Path(temp_dir.name) / "session.sqlite3"
        scheduler = Scheduler(db_path)
        scheduler.seed_slots(utc_now())
        session = ConversationSession(
            session_id=session_key,
            scheduler=scheduler,
            agent=TextAgent(scheduler, self.decide_factory()),
            created_at=utc_now(),
            last_activity=utc_now(),
        )
        self._sessions[session_key] = session
        return session

    def get_or_create(self, session_id: str | None = None) -> ConversationSession:
        session_key = session_id or "default"
        if session_key in self._sessions:
            return self._sessions[session_key]
        return self.create_session(session_key)

    def process_turn(self, transcript: str, session_id: str | None = None, now: datetime | None = None) -> dict:
        session = self.get_or_create(session_id)
        return session.process_turn(transcript, now)

    def delete_session(self, session_id: str) -> bool:
        self._sessions.pop(session_id, None)
        temp_dir = self._temp_dirs.pop(session_id, None)
        if temp_dir is None:
            return False
        temp_dir.cleanup()
        return True

    def close(self) -> None:
        for temp_dir in self._temp_dirs.values():
            temp_dir.cleanup()
        self._sessions.clear()
        self._temp_dirs.clear()
