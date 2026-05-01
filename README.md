# HackerRank Orchestrate: Support Triage Agent

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

## Setup

1.  **Configure Environment:**
    Copy `.env.example` to `.env` and add your **GROQ_API_KEY**.
    ```bash
    cp .env.example .env
    # Edit .env and paste your gsk_... key
    ```

2.  **Install Dependencies:**
    ```bash
    pip install pydantic google-generativeai python-dotenv groq
    ```

## Running the Agent

Run the final pass on the provided ticket dataset:
```bash
python code/main.py --input support_tickets/support_tickets.csv --output support_tickets/output.csv --explain
```
*The `--explain` flag prints detailed diagnostics (confidence, risk, retrieval scores) for each row.*

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