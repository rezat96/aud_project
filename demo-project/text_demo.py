import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

from agent import DecisionError, TextAgent
from model import OpenAIRouter
from scheduler import Scheduler, utc_now


def main() -> None:
    try:
        router = OpenAIRouter()
    except DecisionError as error:
        print(error)
        return
    print("Paid OpenAI text demo. Synthetic data only. All times UTC.")
    print("20 turns/requests maximum; 10-minute session; no automatic retries.")
    print("Ask naturally, e.g. 'Oil change in slot 1'. Type confirm, retry, or quit.")
    with TemporaryDirectory() as directory:
        scheduler = Scheduler(Path(directory) / "text.sqlite3")
        scheduler.seed_slots(utc_now())
        agent = TextAgent(scheduler, router)
        for _ in range(20):
            try:
                text = input("You: ")
            except (EOFError, KeyboardInterrupt):
                break
            if text.strip().casefold() == "quit" or monotonic() - router.started >= 600:
                break
            router.trace = {}
            result = agent.turn(text, utc_now())
            print("Agent:", result["reply"])
            print(json.dumps({"event": result["event"], "trace": router.trace,
                              "estimated_session_usd": round(router.estimated_cost_usd(), 6)}))
    print("Session ended. Database deleted. Check actual usage in your provider dashboard.")


if __name__ == "__main__":
    main()