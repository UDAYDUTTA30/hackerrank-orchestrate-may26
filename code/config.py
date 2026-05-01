from pathlib import Path

ALLOWED_STATUS = {"replied", "escalated"}
ALLOWED_REQUEST_TYPE = {"product_issue", "feature_request", "bug", "invalid"}

ALPHA = 0.35
BETA = 0.40
GAMMA = 0.25

RISK_OVERRIDE_THRESHOLD = 0.7
ESCALATE_CONFIDENCE_THRESHOLD = 0.5
DISCLAIMER_CONFIDENCE_THRESHOLD = 0.75

TOP_K = 5

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
TAXONOMY_PATH = DATA_ROOT / "product_taxonomy.json"
