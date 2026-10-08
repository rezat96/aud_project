from datetime import datetime
from typing import Callable
from uuid import uuid4

from scheduler import Booking, BookingError, Scheduler
class DecisionError(ValueError):
    pass


ACTIONS = ("availability", "offer", "clarify", "handoff", "cancel")


def validate_decision(value: object) -> dict:
    if not isinstance(value, dict) or set(value) != {"action", "service", "slot_number"}:
        raise DecisionError("Unexpected decision structure.")
    if not isinstance(value["action"], str) or value["action"] not in ACTIONS:
        raise DecisionError("Unsupported action.")
    if value["action"] == "offer":
        if not isinstance(value["service"], str) or value["service"] not in Scheduler.SERVICES:
            raise DecisionError("Unsupported service.")
        if type(value["slot_number"]) is not int or value["slot_number"] not in (1, 2):
            raise DecisionError("Invalid slot number.")
    elif value["service"] is not None or value["slot_number"] is not None:
        raise DecisionError("Unused action arguments must be null.")
    return value


class TextAgent:
    def __init__(self, scheduler: Scheduler, decide: Callable[[dict], object]):
        self.scheduler = scheduler
        self.decide = decide
        self.offer_id: str | None = None
        self.request_key: str | None = None
        self.last_booking: Booking | None = None
        self.customer_id = "demo-customer"
        self.turns = 0

    def turn(self, text: str, now: datetime) -> dict:
        if self.turns >= 20:
            return {"event": "limit", "reply": "Session turn limit reached."}
        self.turns += 1
        if not text.strip() or len(text) > 1000:
            return {"event": "rejected", "reply": "Use 1 to 1000 characters."}
        command = text.strip().casefold()
        if command in ("confirm", "retry"):
            if not self.offer_id or not self.request_key:
                return {"event": "rejected", "reply": "Prepare and review an offer first."}
            if command == "retry" and self.last_booking is None:
                return {"event": "rejected", "reply": "Use confirm first; retry only repeats a confirmed booking."}
            try:
                booking = self.scheduler.confirm_booking(
                    self.offer_id, self.request_key, self.customer_id, True, now
                )
                self.last_booking = booking
                return {"event": "booked", "booking_id": booking.booking_id,
                        "slot_id": booking.slot_id, "service": booking.service,
                        "reply": f"Confirmed {booking.service} at {booking.slot_id}. Booking {booking.booking_id}."}
            except BookingError as error:
                return {"event": "rejected", "reply": str(error)}
        self.offer_id = None
        self.request_key = None
        self.last_booking = None
        slots = self.scheduler.available_slots(now)
        try:
            decision = validate_decision(self.decide({"utterance": text, "slots": slots}))
            action = decision["action"]
            if action == "offer":
                index = decision["slot_number"] - 1
                if index >= len(slots):
                    raise BookingError("Slot unavailable. Ask for current availability.")
                slot_id = slots[index]["slot_id"]
                self.offer_id = self.scheduler.prepare_offer(
                    slot_id, decision["service"], self.customer_id, now
                )
                self.request_key = str(uuid4())
                self.last_booking = None
                return {"event": "offer", "slot_id": slot_id, "service": decision["service"],
                        "reply": f"Proposed {decision['service']} at {slot_id} UTC. Type confirm to book. Not reserved; offer expires in five minutes."}
            if action == "availability":
                return {"event": action, "slots": slots,
                        "reply": "Available slots (UTC): " + "; ".join(
                            f"{index}: {slot['starts_at']}" for index, slot in enumerate(slots, 1)
                        )}
            if action in ("cancel", "handoff"):
                self.offer_id = None
                self.request_key = None
                self.last_booking = None
            replies = {
                "clarify": "Which service (oil change or tire rotation) and which available slot (1 or 2)?",
                "handoff": "This request needs staff. Mock handoff only; no person was contacted.",
                "cancel": "Pending offer cleared. Existing bookings were not cancelled.",
            }
            return {"event": action, "reply": replies[action]}
        except (DecisionError, BookingError) as error:
            return {"event": "rejected", "reply": str(error)}