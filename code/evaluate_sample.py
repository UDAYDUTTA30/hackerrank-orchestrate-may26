import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from config import DATA_ROOT, TAXONOMY_PATH
from models import TicketInput
from pipeline import run_ticket
from retriever import Retriever
from taxonomy import ProductTaxonomy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sample", required=True)
    p.add_argument("--report", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    taxonomy = ProductTaxonomy(TAXONOMY_PATH)
    retriever = Retriever(DATA_ROOT)

    confusion = defaultdict(Counter)
    domain_breakdown = defaultdict(Counter)
    false_escalations = 0
    false_replies = 0
    risky_expected = 0
    safe_expected = 0
    low_retrieval = 0
    detailed = []

    with Path(args.sample).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            expected_status = (row.get("Status") or "").strip().lower()
            expected_type = (row.get("Request Type") or "").strip().lower()
            ticket = TicketInput(
                issue=(row.get("Issue") or "").strip(),
                subject=(row.get("Subject") or "").strip(),
                company=(row.get("Company") or "").strip(),
                row_id=i,
            )
            pred = run_ticket(ticket, taxonomy, retriever)

            confusion[expected_type][pred["request_type"]] += 1
            domain = (ticket.company or "None").lower()
            domain_breakdown[domain][pred["status"]] += 1
            if "retrieval=0.00" in pred["justification"]:
                low_retrieval += 1

            expected_risky = expected_status == "escalated"
            if expected_risky:
                risky_expected += 1
                if pred["status"] != "escalated":
                    false_replies += 1
            else:
                safe_expected += 1
                if pred["status"] == "escalated":
                    false_escalations += 1

            detailed.append(
                {
                    "ticket_id": i,
                    "expected_request_type": expected_type,
                    "pred_request_type": pred["request_type"],
                    "expected_status": expected_status,
                    "pred_status": pred["status"],
                    "justification": pred["justification"],
                }
            )

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "ticket_id",
                "expected_request_type",
                "pred_request_type",
                "expected_status",
                "pred_status",
                "justification",
            ],
        )
        writer.writeheader()
        writer.writerows(detailed)

    print("Confusion Matrix (request_type):")
    for expected, preds in confusion.items():
        print(expected, dict(preds))
    print("\nRisk Analysis:")
    fer = (false_escalations / safe_expected * 100.0) if safe_expected else 0.0
    frr = (false_replies / risky_expected * 100.0) if risky_expected else 0.0
    print(f"False Escalation Rate: {fer:.2f}%")
    print(f"False Reply Rate: {frr:.2f}%")
    print(f"Coverage gaps (retrieval score < 0.3 proxy): {low_retrieval}")
    print("\nPer-domain status breakdown:")
    for d, c in domain_breakdown.items():
        print(d, dict(c))
    print(f"\nDetailed misclassification report: {report_path}")


if __name__ == "__main__":
    main()
