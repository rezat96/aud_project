import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from scheduler import BookingError, Scheduler


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.scheduler = Scheduler(Path(self.directory.name) / "test.sqlite3")
        self.now = datetime(2026, 10, 7, 12, tzinfo=timezone.utc)
        self.scheduler.seed_slots(self.now)
        self.slot_id = self.scheduler.available_slots(self.now)[0]["slot_id"]
        self.offer = self.scheduler.prepare_offer(
            self.slot_id, "oil_change", "demo-customer", self.now
        )

    def book(self, **overrides):
        parameters = dict(offer_id=self.offer, request_key="request-1",
                          customer_id="demo-customer", confirmed=True, now=self.now)
        parameters.update(overrides)
        return self.scheduler.confirm_booking(**parameters)

    def test_confirmed_booking_removes_slot(self):
        self.assertEqual(self.book().slot_id, self.slot_id)
        self.assertEqual(len(self.scheduler.available_slots(self.now)), 1)

    def test_unconfirmed_request_does_not_book(self):
        with self.assertRaisesRegex(BookingError, "confirmation"):
            self.book(confirmed=False)
        self.assertEqual(len(self.scheduler.available_slots(self.now)), 2)

    def test_replay_returns_original_booking_even_after_offer_expiry(self):
        original = self.book()
        self.assertEqual(self.book(now=self.now + timedelta(minutes=10)), original)

    def test_reused_key_for_different_offer_is_rejected(self):
        self.book()
        other_slot = self.scheduler.available_slots(self.now)[0]["slot_id"]
        other_offer = self.scheduler.prepare_offer(
            other_slot, "oil_change", "demo-customer", self.now
        )
        with self.assertRaisesRegex(BookingError, "different request"):
            self.book(offer_id=other_offer)

    def test_expired_offer_is_rejected(self):
        with self.assertRaisesRegex(BookingError, "expired"):
            self.book(now=self.now + timedelta(minutes=6))

    def test_wrong_customer_is_rejected(self):
        with self.assertRaisesRegex(BookingError, "Unknown offer"):
            self.book(customer_id="someone-else")

    def test_empty_request_key_is_rejected(self):
        with self.assertRaisesRegex(BookingError, "request key"):
            self.book(request_key="")

    def test_unsupported_service_is_rejected(self):
        with self.assertRaisesRegex(BookingError, "Unsupported"):
            self.scheduler.prepare_offer(self.slot_id, "repair", "demo", self.now)

    def test_concurrent_bookings_have_one_winner(self):
        other_offer = self.scheduler.prepare_offer(
            self.slot_id, "tire_rotation", "demo-customer", self.now
        )

        def attempt(offer_id):
            try:
                return self.book(offer_id=offer_id, request_key=offer_id)
            except BookingError:
                return None

        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(executor.map(attempt, (self.offer, other_offer)))
        self.assertEqual(sum(outcome is not None for outcome in outcomes), 1)


if __name__ == "__main__":
    unittest.main()