import argparse
import json
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from scheduler import BookingError, Scheduler, utc_now


def emit(event: str, **details) -> None:
    print(json.dumps({"event": event, **details}))


def smoke(scheduler: Scheduler) -> None:
    now = utc_now()
    slots = scheduler.available_slots(now)
    offer = scheduler.prepare_offer(
        slots[0]["slot_id"], "oil_change", "demo-customer", now
    )
    request_key = str(uuid4())
    try:
        scheduler.confirm_booking(offer, request_key, "demo-customer", False, now)
    except BookingError as error:
        emit("confirmation_blocked", reason=str(error))
    booking = scheduler.confirm_booking(offer, request_key, "demo-customer", True, now)
    replay = scheduler.confirm_booking(offer, request_key, "demo-customer", True, now)
    assert booking == replay
    assert len(scheduler.available_slots(now)) == len(slots) - 1
    emit("booking_created", **asdict(booking))
    emit("retry_verified", same_booking=booking == replay)


def console(scheduler: Scheduler) -> None:
    offer = None
    request_key = None
    print("Local mock dealership. All times UTC. No AI or network calls.")
    print("Commands: slots | offer oil_change 1 | offer tire_rotation 2 | confirm | retry | quit")
    print("An offer is not a reservation. Confirm commits the latest offered details.")
    for _ in range(20):
        try:
            command = input("> ").strip().split()
        except (EOFError, KeyboardInterrupt):
            break
        if not command:
            continue
        if command == ["quit"]:
            break
        try:
            if command == ["slots"]:
                emit("availability", slots=scheduler.available_slots(utc_now()))
            elif len(command) == 3 and command[0] == "offer":
                slots = scheduler.available_slots(utc_now())
                index = int(command[2]) - 1
                if index < 0 or index >= len(slots):
                    raise BookingError("Choose a slot number from current availability.")
                proposed_offer = scheduler.prepare_offer(
                    slots[index]["slot_id"], command[1], "demo-customer", utc_now()
                )
                offer, request_key = proposed_offer, str(uuid4())
                emit("awaiting_confirmation", service=command[1],
                     slot_id=slots[index]["slot_id"], expires_in_seconds=300)
            elif command in (["confirm"], ["retry"]):
                if offer is None or request_key is None:
                    raise BookingError("Prepare an offer first.")
                booking = scheduler.confirm_booking(
                    offer, request_key, "demo-customer", True, utc_now()
                )
                emit("booking_result", **asdict(booking))
            else:
                emit("unsupported_command", handoff="staff")
        except (BookingError, ValueError) as error:
            emit("request_rejected", reason=str(error))
    print("Session ended. Maximum 20 commands per local session.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    with TemporaryDirectory() as directory:
        scheduler = Scheduler(Path(directory) / "demo.sqlite3")
        scheduler.seed_slots(utc_now())
        if arguments.smoke:
            smoke(scheduler)
        else:
            console(scheduler)


if __name__ == "__main__":
    main()