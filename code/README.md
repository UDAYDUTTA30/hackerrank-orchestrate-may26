# Support Triage Agent

Deterministic, terminal-based support triage pipeline for HackerRank Orchestrate.

## Project Tracking

For external reviewers (plan, milestones, current progress, quality snapshot), see:

- `code/PROJECT_STATUS.md`

## Run

```bash
python main.py --input ../support_tickets/support_tickets.csv --output ../support_tickets/output.csv --sample ../support_tickets/sample_support_tickets.csv
```

## Evaluate on Sample

```bash
python evaluate_sample.py --sample ../support_tickets/sample_support_tickets.csv --report ./reports/sample_eval.csv
```

## Corpus Health

```bash
python corpus_health.py --data-root ../data --output ./reports/corpus_health_report.json
```

## Design

- `classifier.py`: domain + request_type + product_area inference via taxonomy/rules
- `retriever.py`: lexical retrieval over markdown KB
- `risk_engine.py`: standalone risk scoring and escalation flags
- `decision_engine.py`: weighted confidence formula and decision rules
- `grounding_validator.py`: checks whether response claims are grounded in retrieved docs
- `pipeline.py`: orchestration

## Determinism

- No stochastic model calls in baseline
- Stable sort for retrieval
- Fixed weights in decision engine
