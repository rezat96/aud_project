import re
from datetime import date, datetime, timedelta
from typing import Callable
from uuid import uuid4

from scheduler import Booking, BookingError, Scheduler
class DecisionError(ValueError):
    pass


ACTIONS = ("availability", "offer", "clarify", "handoff", "cancel", "booking_status")
ACKNOWLEDGEMENTS = {"ok", "okay", "yes", "yeah", "yep", "sounds good", "got it"}
CONFIRMATIONS = {"confirm", "confirmed", "yes", "yeah", "yep", "yes please", "go ahead", "book it", "thats correct"}
WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}


def _normalize_command(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.casefold()).split())


def _requested_date(text: str, now: datetime) -> date | None:
    normalized = text.casefold()
    if "tomorrow" in normalized:
        return now.date() + timedelta(days=1)
    if "today" in normalized:
        return now.date()
    explicit_date = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", normalized)
    if explicit_date:
        try:
            return date.fromisoformat(explicit_date.group(1))
        except ValueError:
            return None
    weekday_match = re.search(
        r"\b(?:(next|this)\s+)?(" + "|".join(WEEKDAYS) + r")\b", normalized
    )
    if weekday_match:
        qualifier, weekday_name = weekday_match.groups()
        days_ahead = (WEEKDAYS[weekday_name] - now.weekday()) % 7
        if days_ahead == 0 or qualifier == "next":
            days_ahead += 7
        return now.date() + timedelta(days=days_ahead)
    return None


def validate_decision(value: object) -> dict:
    if not isinstance(value, dict) or set(value) != {"action", "service", "slot_id"}:
        raise DecisionError("Unexpected decision structure.")
    if not isinstance(value["action"], str) or value["action"] not in ACTIONS:
        raise DecisionError("Unsupported action.")
    if value["action"] == "booking_status":
        return {"action": "booking_status", "service": None, "slot_id": None}
    if value["service"] is not None:
        if not isinstance(value["service"], str) or value["service"] not in Scheduler.SERVICES:
            raise DecisionError("Unsupported service.")
    if value["slot_id"] is not None and (
        not isinstance(value["slot_id"], str) or not value["slot_id"]
    ):
        raise DecisionError("Invalid slot identifier.")
    if value["action"] not in ("availability", "offer", "clarify") and (
        value["service"] is not None or value["slot_id"] is not None
    ):
        raise DecisionError("Unused action arguments must be null.")
    if value["action"] == "availability" and value["slot_id"] is not None:
        raise DecisionError("Unused action arguments must be null.")
    return value


class TextAgent:
    def __init__(self, scheduler: Scheduler, decide: Callable[[dict], object]):
        self.scheduler = scheduler
        self.decide = decide
        self.offer_id: str | None = None
        self.request_key: str | None = None
        self.offer_confirmed = False
        self.last_booking: Booking | None = None
        self.remembered_service: str | None = None
        self.remembered_slot_id: str | None = None
        self.remembered_date: date | None = None
        self.confirmation_uncertain = False
        self.customer_id = "demo-customer"
        self.turns = 0

    def turn(self, text: str, now: datetime) -> dict:
        if self.turns >= 20:
            return {"event": "limit", "reply": "Session turn limit reached."}
        self.turns += 1
        if not text.strip() or len(text) > 1000:
            return {"event": "rejected", "reply": "Use 1 to 1000 characters."}
        command = _normalize_command(text)
        confirmation = command in CONFIRMATIONS
        retry = command == "retry"
        if (confirmation or retry) and not self.offer_id:
            return {"event": "rejected", "reply": "Prepare and review an offer first."}
        if self.offer_id and (confirmation or retry):
            if not self.offer_id or not self.request_key:
                return {"event": "rejected", "reply": "Prepare and review an offer first."}
            if retry and not self.offer_confirmed and not self.confirmation_uncertain:
                return {"event": "rejected", "reply": "Use confirm first; retry only repeats a confirmed booking."}
            try:
                booking = self.scheduler.confirm_booking(
                    self.offer_id, self.request_key, self.customer_id, True, now
                )
                self.last_booking = booking
                self.offer_confirmed = True
                self.confirmation_uncertain = False
                self.remembered_service = None
                self.remembered_slot_id = None
                self.remembered_date = None
                return {"event": "booked", "booking_id": booking.booking_id,
                        "slot_id": booking.slot_id, "service": booking.service,
                        "reply": f"Confirmed {booking.service} at {booking.slot_id}. Booking {booking.booking_id}."}
            except TimeoutError:
                self.confirmation_uncertain = True
                return {
                    "event": "confirmation_uncertain",
                    "reply": "I could not verify whether the booking completed. Say retry to safely check the same request; the request key prevents a duplicate booking.",
                }
            except BookingError as error:
                self._clear_pending_offer()
                return {"event": "rejected", "reply": str(error)}
        if self.offer_id and command in ACKNOWLEDGEMENTS:
            return self._awaiting_confirmation()

        if self.last_booking is not None and _asks_booking_status(text):
            return self._booking_status(_asks_for_travel_estimate(text))

        had_pending_offer = self.offer_id is not None and not self.offer_confirmed
        mentioned_date = _requested_date(text, now)
        if mentioned_date is not None:
            self.remembered_date = mentioned_date
        slots = self.scheduler.available_slots(now)
        available_slot_ids = {slot["slot_id"] for slot in slots}
        if self.remembered_slot_id not in available_slot_ids:
            self.remembered_slot_id = None
        try:
            raw_decision = self.decide({
                "utterance": text,
                "slots": slots,
                "memory": {
                    "service": self.remembered_service,
                    "slot_id": self.remembered_slot_id,
                    "requested_date": self.remembered_date.isoformat() if self.remembered_date else None,
                    "latest_booking": (
                        {"service": self.last_booking.service,
                         "slot_id": self.last_booking.slot_id}
                        if self.last_booking is not None else None
                    ),
                },
            })
            if isinstance(raw_decision, dict) and set(raw_decision) == {
                "action", "service", "slot_number"
            }:
                slot_number = raw_decision["slot_number"]
                if type(slot_number) is int and 1 <= slot_number <= len(slots):
                    slot_id = slots[slot_number - 1]["slot_id"]
                elif slot_number is None:
                    slot_id = None
                else:
                    slot_id = slot_number
                raw_decision = {
                    "action": raw_decision["action"],
                    "service": raw_decision["service"],
                    "slot_id": slot_id,
                }
            decision = validate_decision(raw_decision)
            action = decision["action"]
            service = decision["service"] or self.remembered_service
            slot_id = self.remembered_slot_id
            if decision["slot_id"] is not None:
                if decision["slot_id"] not in available_slot_ids:
                    raise BookingError("Slot unavailable. Ask for current availability.")
                slot_id = decision["slot_id"]
            if decision["service"] is not None:
                self.remembered_service = decision["service"]
            if action == "offer" and self.remembered_date and slot_id:
                offered_date = datetime.fromisoformat(slot_id).date()
                if offered_date != self.remembered_date:
                    self.remembered_slot_id = None
                    return {
                        "event": "date_mismatch",
                        "service": self.remembered_service,
                        "requested_date": self.remembered_date.isoformat(),
                        "slot_id": slot_id,
                        "reply": f"That appointment is on {offered_date.isoformat()}, not {self.remembered_date.isoformat()}. No offer was created. Please choose an available time on the requested date.",
                    }
            if decision["service"] is not None:
                self.remembered_service = decision["service"]
            if decision["slot_id"] is not None:
                self.remembered_slot_id = slot_id
            if action == "offer":
                service = self.remembered_service
                slot_id = self.remembered_slot_id
                if service is None or slot_id is None:
                    return self._awaiting_confirmation() if had_pending_offer else self._clarification()
                new_offer_id = self.scheduler.prepare_offer(
                    slot_id, service, self.customer_id, now
                )
                self.offer_id = new_offer_id
                self.request_key = str(uuid4())
                self.offer_confirmed = False
                self.confirmation_uncertain = False
                return {"event": "offer", "slot_id": slot_id, "service": service,
                    "reply": f"Proposed {service} at {slot_id} UTC. Type confirm to book. Not reserved; offer expires in five minutes."}
            if action == "booking_status":
                return self._booking_status(_asks_for_travel_estimate(text))
            if action == "availability":
                service_label = f" for {self.remembered_service}" if self.remembered_service else ""
                return {"event": action, "slots": slots,
                        "reply": f"Available appointments{service_label} (UTC): " + "; ".join(
                            f"{index}: {slot['starts_at']}" for index, slot in enumerate(slots, 1)
                        )}
            if action in ("cancel", "handoff"):
                self._clear_pending_offer()
                self.remembered_service = None
                self.remembered_slot_id = None
                self.remembered_date = None
            if action == "clarify":
                return self._awaiting_confirmation() if had_pending_offer else self._clarification()
            replies = {
                "handoff": "This request needs staff. Mock handoff only; no person was contacted.",
                "cancel": "Pending offer cleared. Existing bookings were not cancelled.",
            }
            return {"event": action, "reply": replies[action]}
        except (DecisionError, BookingError) as error:
            return {"event": "rejected", "reply": str(error)}

    def _clear_pending_offer(self) -> None:
        self.offer_id = None
        self.request_key = None
        self.offer_confirmed = False
        self.confirmation_uncertain = False

    def _booking_status(self, includes_travel_estimate: bool = False) -> dict:
        if self.last_booking is None:
            return {"event": "booking_status", "reply": "There is no confirmed booking in this session."}
        booking = self.last_booking
        reply = f"Your latest confirmed booking is {booking.service} at {booking.slot_id} UTC. Booking {booking.booking_id}."
        if includes_travel_estimate:
            reply += " I can't estimate when you will get home because I don't know the service duration, your destination, or travel conditions."
        return {
            "event": "booking_status",
            "booking_id": booking.booking_id,
            "service": booking.service,
            "slot_id": booking.slot_id,
            "reply": reply,
        }

    def _awaiting_confirmation(self) -> dict:
        return {
            "event": "awaiting_confirmation",
            "reply": f"Your proposed {self.remembered_service} at {self.remembered_slot_id} UTC is still awaiting confirmation. No booking has been made. Say confirm to book, or cancel to clear the offer.",
        }

    def _clarification(self, previous_offer_cleared: bool = False) -> dict:
        if previous_offer_cleared:
            return {
                "event": "clarify",
                "reply": "No booking was made. That offer was cleared by your new message. Repeat the service and slot to request a new offer, then type confirm to book.",
            }
        missing = []
        if self.remembered_service is None:
            missing.append("service (oil change or tire rotation)")
        if self.remembered_slot_id is None:
            missing.append("available day and time")
        if not missing:
            reply = "What would you like to change about your request?"
        else:
            reply = "Which " + " and which ".join(missing) + "?"
        return {"event": "clarify", "reply": reply}


def _asks_booking_status(text: str) -> bool:
    normalized = _normalize_command(text)
    asks_for_detail = re.search(r"\b(what|when|which|remind|recall|remember|tell)\b", normalized)
    refers_to_booking = re.search(r"\b(booked|booking|appointment|slot)\b", normalized)
    asks_for_time = re.search(r"\b(time|date|day)\b", normalized)
    return bool(asks_for_detail and (refers_to_booking or asks_for_time))


def _asks_for_travel_estimate(text: str) -> bool:
    normalized = _normalize_command(text)
    return bool(re.search(r"\b(home|drive|driving|travel|trip|commute|arrive)\b", normalized))