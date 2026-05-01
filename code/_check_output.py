import csv

rows = list(csv.DictReader(open("support_tickets/output.csv", "r", encoding="utf-8-sig")))
for i, row in enumerate(rows[:5], 1):
    print(f"=== Row {i} ===")
    print("status:", row["status"])
    print("request_type:", row["request_type"])
    print("product_area:", row["product_area"])
    print("response:", row["response"][:400])
    print("justification:", row["justification"][:250])
    print()
