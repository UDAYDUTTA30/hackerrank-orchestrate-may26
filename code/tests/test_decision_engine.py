import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decision_engine import decide


class DecisionEngineTests(unittest.TestCase):
    def test_risk_override_wins(self) -> None:
        result = decide(0.95, 0.95, 0.8, escalation_required=False)
        self.assertEqual(result.status, "escalated")
        self.assertEqual(result.decision_mode, "risk_override")

    def test_low_confidence_escalates(self) -> None:
        result = decide(0.2, 0.2, 0.1, escalation_required=False)
        self.assertEqual(result.status, "escalated")
        self.assertEqual(result.decision_mode, "low_confidence")

    def test_mid_confidence_disclaimer(self) -> None:
        result = decide(0.6, 0.65, 0.3, escalation_required=False)
        self.assertEqual(result.status, "replied")
        self.assertEqual(result.decision_mode, "reply_with_disclaimer")

    def test_high_confidence_full_resolution(self) -> None:
        result = decide(0.95, 0.95, 0.1, escalation_required=False)
        self.assertEqual(result.status, "replied")
        self.assertEqual(result.decision_mode, "full_resolution")


if __name__ == "__main__":
    unittest.main()
