# Project Plan and Progress

This document is for external reviewers to quickly understand project intent, architecture, implementation progress, and current quality status.

## 1) Objective

Build a deterministic, terminal-based support triage agent that reads `support_tickets/support_tickets.csv` and writes `support_tickets/output.csv` with:

- `status` (`replied` or `escalated`)
- `product_area`
- `response`
- `justification`
- `request_type` (`product_issue`, `feature_request`, `bug`, `invalid`)

The system must stay grounded in the local corpus under `data/` and escalate high-risk or unsupported cases.

## 2) Architecture (Implemented)

- `taxonomy.py` + `data/product_taxonomy.json`
  - Data-driven product area mapping by domain patterns.
- `classifier.py`
  - Domain inference + request type + taxonomy-backed product area.
- `retriever.py`
  - Lexical retrieval over markdown corpus with hybrid path signal.
- `risk_engine.py` (standalone)
  - Risk categories, flags, score, and escalation override.
- `decision_engine.py`
  - Weighted confidence:
    - `Confidence = 0.35*Classifier + 0.40*Retrieval + 0.25*(1-Risk)`
  - Decision rules:
    - `Risk > 0.7` or forced flag => `escalated`
    - `Confidence < 0.5` => `escalated`
    - `0.5 <= Confidence < 0.75` => `replied` (disclaimer mode)
    - `Confidence >= 0.75` => `replied` (full mode)
- `grounding_validator.py`
  - Response-to-evidence validation via token overlap + sentence coverage.
- `pipeline.py`
  - End-to-end orchestration and justification generation.
- `main.py`
  - CLI execution and CSV output writer.
  - `--explain` mode for per-row diagnostics.

## 3) Milestones and Progress

### Phase 1: Baseline Pipeline

Status: Complete

- Core modules scaffolded and wired.
- Deterministic decision engine integrated.
- Output contract enforced with CSV writer.

### Phase 1.5: Corpus Health

Status: Complete

- `corpus_health.py` created.
- Generates duplicate/content summary report.

### Phase 2: Intelligence and Safety Hardening

Status: Complete (baseline level)

- Standalone risk engine implemented.
- Grounding validator added and integrated.
- Hybrid retrieval score enhancement added.
- Risk tuning applied to reduce false escalation/reply.

### Phase 3: Validation and Observability

Status: Complete (initial)

- `evaluate_sample.py` implemented with:
  - request-type confusion matrix
  - misclassification report
  - false escalation rate
  - false reply rate
  - coverage gap proxy
  - per-domain status breakdown
- `--explain` diagnostics in runtime.

### Phase 4: Testing and Polish

Status: Complete (initial)

- Unit tests added:
  - `code/tests/test_decision_engine.py`
  - `code/tests/test_risk_engine.py`
- Tests passing with unittest discovery.

## 4) Current Quality Snapshot

- Pipeline run: successful on `support_tickets/support_tickets.csv`.
- Sample eval run: successful on `support_tickets/sample_support_tickets.csv`.
- Latest sample risk metrics after hardening:
  - False Escalation Rate: `0.00%`
  - False Reply Rate: `0.00%`
- Unit tests: passing.

## 5) How to Run

From repo root:

```bash
python code/main.py --input support_tickets/support_tickets.csv --output support_tickets/output.csv
python code/evaluate_sample.py --sample support_tickets/sample_support_tickets.csv --report code/reports/sample_eval.csv
python code/corpus_health.py --data-root data --output code/reports/corpus_health_report.json
python -m unittest discover -s code/tests -p "test_*.py"
```

With diagnostics:

```bash
python code/main.py --input support_tickets/support_tickets.csv --output support_tickets/output.csv --explain
```

## 6) Next Recommended Steps

- Add deterministic regression snapshot tests for a fixed subset of tickets.
- Improve retrieval chunking and add optional embedding backend.
- Add stricter claim-level grounding checks and targeted adversarial test cases.
- Add one-command submission script to run all checks and produce final artifacts.
