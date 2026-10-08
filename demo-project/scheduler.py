import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4


class BookingError(ValueError):
    pass


@dataclass(frozen=True)
class Booking:
    booking_id: str
    slot_id: str
    service: str
    customer_id: str


class Scheduler:
    SERVICES = frozenset({"oil_change", "tire_rotation"})

    def __init__(self, database: Path):
        self.database = database
        with closing(self._connect()) as connection, connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS slots (
                    slot_id TEXT PRIMARY KEY,
                    starts_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS offers (
                    offer_id TEXT PRIMARY KEY,
                    slot_id TEXT NOT NULL REFERENCES slots(slot_id),
                    service TEXT NOT NULL,
                    customer_id TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS bookings (
                    booking_id TEXT PRIMARY KEY,
                    slot_id TEXT NOT NULL UNIQUE REFERENCES slots(slot_id),
                    offer_id TEXT NOT NULL UNIQUE REFERENCES offers(offer_id),
                    request_key TEXT NOT NULL UNIQUE
                );
            """)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def seed_slots(self, now: datetime) -> None:
        tomorrow = (now + timedelta(days=1)).replace(
            hour=9, minute=0, second=0, microsecond=0
        )
        with closing(self._connect()) as connection, connection:
            for hour in (9, 10):
                start = tomorrow.replace(hour=hour).isoformat()
                connection.execute(
                    "INSERT OR IGNORE INTO slots VALUES (?, ?)", (start, start)
                )

    def available_slots(self, now: datetime) -> list[dict[str, str]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """SELECT slot_id, starts_at FROM slots
                   WHERE starts_at > ?
                   AND slot_id NOT IN (SELECT slot_id FROM bookings)
                   ORDER BY starts_at""", (now.isoformat(),)
            ).fetchall()
            return [dict(row) for row in rows]

    def prepare_offer(
        self, slot_id: str, service: str, customer_id: str, now: datetime
    ) -> str:
        if service not in self.SERVICES:
            raise BookingError("Unsupported service; hand off to staff.")
        if not customer_id.strip():
            raise BookingError("A synthetic customer identifier is required.")
        offer_id = str(uuid4())
        with closing(self._connect()) as connection, connection:
            if slot_id not in {slot["slot_id"] for slot in self.available_slots(now)}:
                raise BookingError("Slot unavailable; fetch availability again.")
            connection.execute(
                "INSERT INTO offers VALUES (?, ?, ?, ?, ?)",
                (offer_id, slot_id, service, customer_id,
                 (now + timedelta(minutes=5)).isoformat()),
            )
        return offer_id

    def confirm_booking(
        self, offer_id: str, request_key: str, customer_id: str,
        confirmed: bool, now: datetime
    ) -> Booking:
        if confirmed is not True:
            raise BookingError("Explicit customer confirmation is required.")
        if not request_key.strip():
            raise BookingError("A request key is required for safe retries.")
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            replay = connection.execute(
                """SELECT b.*, o.service, o.customer_id FROM bookings b
                   JOIN offers o USING (offer_id) WHERE request_key = ?""",
                (request_key,),
            ).fetchone()
            if replay is not None:
                if replay["offer_id"] != offer_id or replay["customer_id"] != customer_id:
                    raise BookingError("Request key belongs to a different request.")
                return Booking(replay["booking_id"], replay["slot_id"],
                               replay["service"], replay["customer_id"])
            offer = connection.execute(
                "SELECT * FROM offers WHERE offer_id = ?", (offer_id,)
            ).fetchone()
            if offer is None or offer["customer_id"] != customer_id:
                raise BookingError("Unknown offer for this customer.")
            if offer["expires_at"] <= now.isoformat() or offer["slot_id"] <= now.isoformat():
                raise BookingError("Offer expired; fetch availability again.")
            booking = Booking(str(uuid4()), offer["slot_id"],
                              offer["service"], customer_id)
            try:
                connection.execute(
                    "INSERT INTO bookings VALUES (?, ?, ?, ?)",
                    (booking.booking_id, booking.slot_id, offer_id, request_key),
                )
            except sqlite3.IntegrityError as error:
                raise BookingError("Slot or offer already booked; recheck before retrying.") from error
            return booking


def utc_now() -> datetime:
    return datetime.now(timezone.utc)