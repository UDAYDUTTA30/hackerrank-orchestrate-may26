import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import ClassifierOutput, TicketInput
from risk_engine import assess_risk


class RiskEngineTests(unittest.TestCase):
    def test_fraud_triggers_immediate_escalation(self) -> None:
        ticket = TicketInput(
            issue="I did not make this purchase and my card was stolen",
            subject="fraud payment",
            company="Visa",
            row_id=1,
        )
        cls = ClassifierOutput(
            domain="visa",
            product_area="Fraud Report",
            request_type="product_issue",
            classifier_score=0.8,
            reasons=[],
        )
        result = assess_risk(ticket, cls)
        self.assertTrue(result.escalation_required)
        self.assertGreaterEqual(result.risk_score, 0.9)
        self.assertIn("fraud_payment", result.risk_flags)

    def test_invalid_not_forced_escalation(self) -> None:
        ticket = TicketInput(
            issue="What is the actor name in Iron Man?",
            subject="off topic",
            company="None",
            row_id=2,
        )
        cls = ClassifierOutput(
            domain="none",
            product_area="general_support",
            request_type="invalid",
            classifier_score=0.2,
            reasons=[],
        )
        result = assess_risk(ticket, cls)
        self.assertFalse(result.escalation_required)
        self.assertGreaterEqual(result.risk_score, 0.35)
        self.assertIn("invalid_request", result.risk_flags)

    def test_outage_forces_escalation(self) -> None:
        ticket = TicketInput(
            issue="none of the submissions are working right now",
            subject="site is down",
            company="HackerRank",
            row_id=3,
        )
        cls = ClassifierOutput(
            domain="hackerrank",
            product_area="Platform Bug",
            request_type="bug",
            classifier_score=0.6,
            reasons=[],
        )
        result = assess_risk(ticket, cls)
        self.assertTrue(result.escalation_required)
        self.assertIn("platform_outage", result.risk_flags)


if __name__ == "__main__":
    unittest.main()
