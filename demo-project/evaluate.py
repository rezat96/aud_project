import argparse
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

from agent import DecisionError, TextAgent
from model import OpenAIRouter, PROMPT_VERSION
from scheduler import Scheduler


def evaluate(live: bool = False) -> dict:
    cases = json.loads(Path(__file__).with_name("eval_cases.json").read_text())
    router = OpenAIRouter() if live else None
    results = []
    now = datetime(2026, 10, 8, 12, tzinfo=timezone.utc)
    for case in cases:
        with TemporaryDirectory() as directory:
            database = Path(directory) / "eval.sqlite3"
            scheduler = Scheduler(database)
            scheduler.seed_slots(now)
            original_slots = scheduler.available_slots(now)
            scripted = iter(case["decisions"])
            observed_decisions = []

            def decide(context):
                decision = router(context) if router is not None else next(scripted)
                observed_decisions.append(decision)
                return decision

            agent = TextAgent(scheduler, decide)
            started = monotonic()
            responses = [agent.turn(text, now) for text in case["turns"]]
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
            actual = sorted(persisted)
            decision_ok = observed_decisions == case["decisions"]
            outcome_ok = events == case["events"] and actual == expected
            results.append({
                "case": case["name"], "passed": decision_ok and outcome_ok,
                "decision_ok": decision_ok, "outcome_ok": outcome_ok,
                "events": events, "expected_events": case["events"],
                "persisted_bookings": len(persisted),
                "latency_ms": round((monotonic() - started) * 1000),
            })
    return {
        "mode": "live_model" if live else "scripted_controller_only",
        "prompt_version": PROMPT_VERSION,
        "passed": sum(result["passed"] for result in results), "total": len(results),
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