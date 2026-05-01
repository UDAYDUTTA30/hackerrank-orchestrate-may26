import argparse
import csv
import time
from pathlib import Path

from config import DATA_ROOT, TAXONOMY_PATH
from llm_client import API_KEY, GROQ_KEY
from models import TicketInput
from pipeline import run_ticket
from retriever import Retriever
from taxonomy import ProductTaxonomy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Print per-ticket confidence/risk diagnostics to stdout",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    taxonomy = ProductTaxonomy(TAXONOMY_PATH)
    retriever = Retriever(DATA_ROOT)

    rows_out = []
    with Path(args.input).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            ticket = TicketInput(
                issue=(row.get("Issue") or row.get("issue") or "").strip(),
                subject=(row.get("Subject") or row.get("subject") or "").strip(),
                company=(row.get("Company") or row.get("company") or "").strip(),
                row_id=idx,
            )
            pred = run_ticket(ticket, taxonomy, retriever, explain=args.explain)
            
            # Small delay for rate stability
            if API_KEY or GROQ_KEY:
                time.sleep(0.5)

            rows_out.append(
                {
                    "issue": ticket.issue,
                    "subject": ticket.subject,
                    "company": ticket.company,
                    "response": pred["response"],
                    "product_area": pred["product_area"],
                    "status": pred["status"],
                    "request_type": pred["request_type"],
                    "justification": pred["justification"],
                }
            )
            if args.explain:
                debug = pred.get("_debug", {})
                print(
                    "row={row} status={status} confidence={conf:.2f} risk={risk:.2f} "
                    "retrieval={ret:.2f} mode={mode} top_doc={doc}".format(
                        row=ticket.row_id,
                        status=pred["status"],
                        conf=debug.get("confidence", 0.0),
                        risk=debug.get("risk_score", 0.0),
                        ret=debug.get("retrieval_relevance", 0.0),
                        mode=debug.get("decision_mode", "unknown"),
                        doc=debug.get("top_doc", ""),
                    )
                )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        fields = ["issue", "subject", "company", "response", "product_area", "status", "request_type", "justification"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Wrote {len(rows_out)} rows to {output_path}")


if __name__ == "__main__":
    main()
