import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    file_count = 0
    domain_count = Counter()
    hashes = defaultdict(list)

    for md in data_root.rglob("*.md"):
        file_count += 1
        parts = md.parts
        if "data" in parts:
            idx = parts.index("data")
            domain = parts[idx + 1] if idx + 1 < len(parts) else "unknown"
        else:
            domain = "unknown"
        domain_count[domain] += 1
        content = md.read_text(encoding="utf-8", errors="ignore")
        digest = hashlib.sha256(content.strip().encode("utf-8")).hexdigest()
        hashes[digest].append(str(md))

    duplicates = [paths for paths in hashes.values() if len(paths) > 1]
    report = {
        "total_markdown_files": file_count,
        "domain_counts": dict(domain_count),
        "duplicate_groups": duplicates,
        "duplicate_group_count": len(duplicates),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote corpus health report to {output}")


if __name__ == "__main__":
    main()
