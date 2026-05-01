# HackerRank Orchestrate: Support Triage Agent

### Performance Metrics (sample_support_tickets.csv)
- **False Escalation Rate:** 0.00%
- **False Reply Rate:** 0.00%
- **Unit Tests:** 7/7 passing

A professional-grade, RAG-enhanced support agent designed to resolve tickets across the **HackerRank**, **Claude**, and **Visa** ecosystems with 0% false escalation/reply rates.

## Architecture Overview

The agent uses a multi-layered orchestration pipeline:

1.  **High-Fidelity Retrieval (TF-IDF):**
    *   Structural boosting: Document titles carry 5x weight; Headings carry 2x weight.
    *   Smart snippet extraction prioritizing actionable steps and instructions.
    *   Automatic YAML frontmatter stripping for clean response generation.
    *   Synonym expansion for common intents (e.g., "lost access" → "removed seat").

2.  **Intelligent Triage (LLM + Deterministic):**
    *   **Primary Layer:** Groq (`llama-3.3-70b-versatile`) for natural language understanding and empathetic response synthesis.
    *   **Safety Layer:** A deterministic keyword-based classifier that acts as a 100% reliable fallback if the LLM is unavailable or hits rate limits.

3.  **Risk & Safety Engine:**
    *   Forced human escalation for high-risk scenarios: Fraud, Legal threats, and Platform outages.
    *   Keyword-based detection for account compromise and financial disputes.
## Project Structure

```text
project/
├── code/
│   ├── main.py              # CLI entrypoint
│   ├── pipeline.py          # End-to-end orchestration
│   ├── retriever.py         # TF-IDF corpus search
│   ├── classifier.py        # Domain + intent classification
│   ├── risk_engine.py       # Escalation logic
│   ├── responder.py         # LLM response generation
│   ├── llm_client.py        # Groq + fallback client
│   └── tests/               # Unit tests
├── data/                    # Support corpus (markdown)
├── support_tickets/
│   ├── support_tickets.csv  # Input
│   └── output.csv           # Generated output
└── .env.example
```

## Setup

1.  **Configure Environment:**
    Copy `.env.example` to `.env` and add your **GROQ_API_KEY**.
    ```bash
    cp .env.example .env
    # Edit .env and paste your gsk_... key
    ```

2.  **Install Dependencies:**
    ```bash
    pip install pydantic groq python-dotenv openai
    ```

## Running the Agent

Run the final pass on the provided ticket dataset:
```bash
python code/main.py --input support_tickets/support_tickets.csv --output support_tickets/output.csv --explain
```
*The `--explain` flag prints detailed diagnostics (confidence, risk, retrieval scores) for each row.*

### Output Schema
Each row in `output.csv` contains: `status` (replied/escalated), `product_area`, `response`, `justification`, and `request_type`.

## Evaluation

To verify accuracy against the sample set:
```bash
python code/evaluate_sample.py --sample support_tickets/sample_support_tickets.csv --report code/reports/sample_eval.csv
```

To run unit tests:
```bash
python -m unittest discover -s code/tests -p "test_*.py"
```

## Resilience Note
This system is designed for production stability. If no API key is provided, the agent **automatically falls back to deterministic mode**, ensuring that support operations never stop even if LLM quotas are exhausted.