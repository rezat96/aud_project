import argparse
import json
import math
import sqlite3
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

from agent import DecisionError, TextAgent
from model import OpenAIRouter, PROMPT_VERSION
from scheduler import Scheduler


class TimeoutAfterCommitScheduler(Scheduler):
    def __init__(self, database: Path):
        super().__init__(database)
        self.timeout_after_next_commit = True

    def confirm_booking(self, *args, **kwargs):
        booking = super().confirm_booking(*args, **kwargs)
        if self.timeout_after_next_commit:
            self.timeout_after_next_commit = False
            raise TimeoutError("Injected lost confirmation response")
        return booking


def _percentiles(values: list[float]) -> dict | None:
    if not values:
        return None
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    return {
        "samples": len(ordered),
        "p50": round(percentile(0.50), 2),
        "p95": round(percentile(0.95), 2),
    }


def evaluate(live: bool = False) -> dict:
    cases = json.loads(Path(__file__).with_name("eval_cases.json").read_text())
    router = OpenAIRouter() if live else None
    results = []
    turn_latencies_ms = []
    model_call_latencies_ms = []
    now = datetime(2026, 10, 8, 12, tzinfo=timezone.utc)
    for case in cases:
        with TemporaryDirectory() as directory:
            database = Path(directory) / "eval.sqlite3"
            scheduler_type = TimeoutAfterCommitScheduler if case.get("timeout_after_commit") else Scheduler
            scheduler = scheduler_type(database)
            scheduler.seed_slots(now)
            original_slots = scheduler.available_slots(now)
            scripted = iter(case["decisions"])
            observed_decisions = []

            def decide(context):
                call_started = monotonic()
                decision = router(context) if router is not None else next(scripted)
                if router is not None:
                    model_call_latencies_ms.append((monotonic() - call_started) * 1000)
                observed_decisions.append(decision)
                return decision

            agent = TextAgent(scheduler, decide)
            started = monotonic()
            responses = []
            external_bookings = []
            for turn_index, text in enumerate(case["turns"]):
                conflict = case.get("competing_booking_before_turn")
                if conflict and turn_index == conflict["turn"]:
                    competing_slot = original_slots[conflict["slot_number"] - 1]["slot_id"]
                    competing_offer = scheduler.prepare_offer(
                        competing_slot, conflict["service"], "competing-customer", now
                    )
                    competing_booking = scheduler.confirm_booking(
                        competing_offer, f"external-{case['name']}", "competing-customer", True, now
                    )
                    external_bookings.append((competing_booking.service, competing_booking.slot_id))
                turn_started = monotonic()
                responses.append(agent.turn(text, now))
                turn_latencies_ms.append((monotonic() - turn_started) * 1000)
            with closing(sqlite3.connect(database)) as connection:
                persisted = connection.execute(
                    """SELECT o.service, b.slot_id FROM bookings b
                       JOIN offers o USING (offer_id) ORDER BY b.slot_id"""
                ).fetchall()
            expected = sorted(
                (booking["service"], original_slots[booking["slot_number"] - 1]["slot_id"])
                for booking in case["bookings"]
            )
            events = [result["event"] for result in responses]
            combined_replies = " ".join(result.get("reply", "") for result in responses).casefold()
            reply_ok = all(
                expected_reply.casefold() in combined_replies
                for expected_reply in case.get("reply_contains", [])
            )
            actual_all = sorted(persisted)
            actual_counts = Counter(actual_all)
            actual_counts.subtract(external_bookings)
            actual = sorted(actual_counts.elements())
            external_booking_outcome_ok = all(
                Counter(actual_all)[booking] >= Counter(external_bookings)[booking]
                for booking in set(external_bookings)
            )
            booking_matches = sum(
                (Counter(expected) & Counter(actual)).values()
            )
            decision_ok = observed_decisions == case["decisions"]
            event_sequence_ok = events == case["events"]
            booking_outcome_ok = actual == expected and external_booking_outcome_ok
            outcome_ok = event_sequence_ok and booking_outcome_ok and reply_ok
            results.append({
                "case": case["name"], "passed": decision_ok and outcome_ok,
                "decision_ok": decision_ok, "outcome_ok": outcome_ok,
                "event_sequence_ok": event_sequence_ok,
                "booking_outcome_ok": booking_outcome_ok,
                "reply_ok": reply_ok,
                "external_booking_outcome_ok": external_booking_outcome_ok,
                "expected_bookings": len(expected),
                "actual_bookings": len(actual),
                "booking_matches": booking_matches,
                "events": events, "expected_events": case["events"],
                "persisted_bookings": len(actual_all),
                "latency_ms": round((monotonic() - started) * 1000),
            })

    decision_passes = sum(result["decision_ok"] for result in results)
    event_passes = sum(result["event_sequence_ok"] for result in results)
    booking_passes = sum(result["booking_outcome_ok"] for result in results)
    cases_requiring_booking = [case for case in cases if case["bookings"]]
    cases_requiring_no_booking = [case for case in cases if not case["bookings"]]
    no_booking_safe = sum(
        result["booking_outcome_ok"]
        for case, result in zip(cases, results)
        if not case["bookings"]
    )
    booking_true_positives = sum(result["booking_matches"] for result in results)
    booking_false_positives = sum(
        result["actual_bookings"] - result["booking_matches"] for result in results
    )
    booking_false_negatives = sum(
        result["expected_bookings"] - result["booking_matches"] for result in results
    )
    case_latencies_ms = [result["latency_ms"] for result in results]
    metrics = {
        "cases_passed": sum(result["passed"] for result in results),
        "decision_pass_rate": round(decision_passes / len(results), 4) if results else None,
        "event_sequence_pass_rate": round(event_passes / len(results), 4) if results else None,
        "exact_booking_outcome_rate": round(booking_passes / len(results), 4) if results else None,
        "booking_task_success_rate": (
            round(sum(
                result["booking_outcome_ok"]
                for case, result in zip(cases, results)
                if case["bookings"]
            ) / len(cases_requiring_booking), 4)
            if cases_requiring_booking else None
        ),
        "no_booking_safety_rate": (
            round(no_booking_safe / len(cases_requiring_no_booking), 4)
            if cases_requiring_no_booking else None
        ),
        "unexpected_booking_cases": len(cases_requiring_no_booking) - no_booking_safe,
        "booking_precision": (
            round(booking_true_positives / (booking_true_positives + booking_false_positives), 4)
            if booking_true_positives + booking_false_positives else None
        ),
        "booking_recall": (
            round(booking_true_positives / (booking_true_positives + booking_false_negatives), 4)
            if booking_true_positives + booking_false_negatives else None
        ),
        "latency_ms": {
            "case_end_to_end": _percentiles(case_latencies_ms),
            "agent_turn": _percentiles(turn_latencies_ms),
            "model_call": _percentiles(model_call_latencies_ms),
        },
    }
    return {
        "mode": "live_model" if live else "scripted_controller_only",
        "prompt_version": PROMPT_VERSION,
        "passed": metrics["cases_passed"], "total": len(results),
        "metrics": metrics,
        "model_calls": router.calls if router is not None else 0,
        "usage": router.usage if router is not None else {},
        "estimated_usd": router.estimated_cost_usd() if router is not None else 0,
        "cases": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Opt in to paid model evaluation")
    arguments = parser.parse_args()
    try:
        report = evaluate(arguments.live)
    except DecisionError as error:
        print(error)
        raise SystemExit(2) from None
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] == report["total"] else 1)


if __name__ == "__main__":
    main()