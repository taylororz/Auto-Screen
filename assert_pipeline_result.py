#!/usr/bin/env python3
import json
import sys
from pathlib import Path

def check_gate():
    # Check whether the summary comparison report exists
    summary_path = Path("diff_results/summary_report.json")
    if not summary_path.exists():
        print("❌ [CI Gate] diff_results/summary_report.json not found. Testing pipeline did not complete!")
        sys.exit(1)

    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ [CI Gate] Failed to parse summary report: {e}")
        sys.exit(1)

    # Check for uncaptured or missing screenshots (compatible with top-level fields and folder_summaries)
    total_missing = data.get("total_missing")
    if total_missing is None:
        folder_summaries = data.get("folder_summaries", [])
        total_missing = sum(folder.get("missing_count", 0) for folder in folder_summaries)

    if total_missing > 0:
        print(f"❌ [CI Gate] Detected {total_missing} missing or uncaptured page(s)!")
        sys.exit(1)

    # Optional check: AI visual bug detection gate (if AI evaluation is enabled)
    ai_failures = 0
    for comp in data.get("comparisons", []):
        for item in comp.get("results", []):
            ai_analysis = item.get("ai_analysis")
            if isinstance(ai_analysis, dict) and ai_analysis.get("severity") == "fail":
                ai_failures += 1

    if ai_failures > 0:
        print(f"⚠️ [CI Gate] Detected {ai_failures} page(s) flagged by AI with severe visual defects (fail)!")
        # Uncomment below if strict blocking is required
        # sys.exit(1)

    print("✅ [CI Gate] Visual testing completed successfully. Gate verification passed!")
    sys.exit(0)

if __name__ == "__main__":
    check_gate()
