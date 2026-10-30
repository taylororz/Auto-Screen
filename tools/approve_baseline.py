#!/usr/bin/env python3
"""Approve a run's screenshots as the reference baseline for version comparison."""
import argparse
import getpass
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def _manifest(folder):
    """SHA-256 per screenshot, so a later change of the approved files can be detected."""
    return {
        str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(folder.rglob("*.png"))
    }


def main():
    ap = argparse.ArgumentParser(description="Approve screenshots as baseline for the version comparison")
    ap.add_argument("--app", required=True, help="App name, e.g. katapp")
    ap.add_argument("--source", help="Screenshot root of the approved run, e.g. Qwen_trained/screenshots")
    ap.add_argument("--label", help="Identity of the approved app build, e.g. its commit SHA")
    ap.add_argument("--note", default="", help="Reason for the approval")
    ap.add_argument("--dest", default=None, help="Baseline directory (default: baselines/<app>)")
    ap.add_argument("--force", action="store_true", help="Replace an existing approved baseline")
    ap.add_argument("--show", action="store_true", help="Only show the current approved baseline")
    args = ap.parse_args()

    dest = Path(args.dest) if args.dest else ROOT_DIR / "baselines" / args.app
    record_path = dest / "baseline.json"

    if args.show:
        if not record_path.exists():
            print(f"No approved baseline in {dest}")
            return 1
        record = json.loads(record_path.read_text(encoding="utf-8"))
        changed = [f for f, h in record.get("files", {}).items()
                   if not (dest / "screenshots" / f).exists() or hashlib.sha256((dest / "screenshots" / f).read_bytes()).hexdigest() != h]
        print(json.dumps({k: v for k, v in record.items() if k != "files"}, ensure_ascii=False, indent=2))
        print(f"{len(record.get('files', {}))} screenshots, {len(changed)} changed since the approval")
        return 0

    if not args.source or not args.label:
        ap.error("--source and --label are required to approve a baseline")
    source = Path(args.source)
    if not source.is_dir() or not any(source.rglob("*.png")):
        ap.error(f"No screenshots in {source}")
    if record_path.exists() and not args.force:
        current = json.loads(record_path.read_text(encoding="utf-8"))
        print(f"An approved baseline exists ({current.get('label')}, {current.get('approved_at')}). Use --force to replace it.")
        return 1

    target = dest / "screenshots"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)

    record = {
        "app": args.app,
        "label": args.label,
        "approved_at": datetime.now(timezone.utc).isoformat(),
        "approved_by": getpass.getuser(),
        "note": args.note,
        "source": str(source.resolve()),
        "files": _manifest(target),
    }
    build_info = source / "app_build.json"
    if build_info.exists():
        record["app_build"] = json.loads(build_info.read_text(encoding="utf-8"))
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Approved {len(record['files'])} screenshots of {args.label} as baseline in {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
