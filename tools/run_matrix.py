#!/usr/bin/env python3
"""Run UI tests across combinations of simulator devices, appearances (light/dark), and Dynamic Type sizes."""
import argparse
import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def _slug(*parts):
    return "_".join(re.sub(r"[^a-z0-9]+", "-", str(p).lower()).strip("-") for p in parts if p)


def find_simulator(device_name, platform_version):
    """Returns the UDID of an available simulator with this name and iOS version."""
    listing = json.loads(subprocess.run(["xcrun", "simctl", "list", "devices", "available", "-j"],
                                        capture_output=True, text=True, check=True).stdout)
    runtime_suffix = "iOS-" + platform_version.replace(".", "-")
    for runtime, devices in listing.get("devices", {}).items():
        if not runtime.endswith(runtime_suffix):
            continue
        for device in devices:
            if device.get("name") == device_name:
                return device["udid"]
    return None


def main():
    ap = argparse.ArgumentParser(description="Run the test for several devices, appearances and text sizes")
    ap.add_argument("--app", required=True)
    ap.add_argument("--model", default="Qwen_trained")
    ap.add_argument("--mode", choices=["dg", "som"], default="dg")
    ap.add_argument("--loops", type=int, default=1)
    ap.add_argument("--devices", nargs="+", required=True, help="Simulator names, e.g. 'iPhone 18 Pro'")
    ap.add_argument("--platform-version", default=None, help="iOS version (default: from the app config)")
    ap.add_argument("--appearance", nargs="+", default=[None], choices=["light", "dark"])
    ap.add_argument("--content-size", nargs="+", default=[None],
                    help="Dynamic Type size, e.g. large (iOS default), extra-extra-large, accessibility-large")
    ap.add_argument("--dry-run", action="store_true", help="Only print the variants")
    args = ap.parse_args()

    config = json.loads((ROOT_DIR / "configs" / f"{args.app}.json").read_text(encoding="utf-8"))
    platform_version = args.platform_version or config["app_config"].get("platform_version", "")

    variants = []
    for device, appearance, content_size in itertools.product(args.devices, args.appearance, args.content_size):
        variants.append({"device": device, "appearance": appearance, "content_size": content_size,
                         "variant": _slug(device, appearance, content_size)})
    print(f"{len(variants)} variants for {args.app} (iOS {platform_version}):")
    for v in variants:
        print(f"  • {v['variant']}: {v['device']}, appearance={v['appearance'] or 'unchanged'}, content_size={v['content_size'] or 'unchanged'}")
    if args.dry_run:
        return 0

    results = []
    touched = set()
    for v in variants:
        udid = find_simulator(v["device"], platform_version)
        if not udid:
            print(f"❌ No available simulator '{v['device']}' with iOS {platform_version}, skipped")
            results.append({**v, "status": "no_simulator"})
            continue
        subprocess.run(["xcrun", "simctl", "boot", udid], capture_output=True)
        subprocess.run(["xcrun", "simctl", "bootstatus", udid, "-b"], capture_output=True)
        env = dict(os.environ,
                   AUTOSCREEN_DEVICE_NAME=v["device"], AUTOSCREEN_PLATFORM_VERSION=platform_version,
                   AUTOSCREEN_DEVICE_UDID=udid, AUTOSCREEN_VARIANT=v["variant"])
        if v["appearance"]:
            env["AUTOSCREEN_APPEARANCE"] = v["appearance"]
            touched.add(udid)
        if v["content_size"]:
            env["AUTOSCREEN_CONTENT_SIZE"] = v["content_size"]
            touched.add(udid)
        print(f"\n▶️ Variant {v['variant']} on {udid}")
        proc = subprocess.run([sys.executable, str(ROOT_DIR / "auto_screen.py"), "test", "--app", args.app,
                               "--model", args.model, "--mode", args.mode, "--loops", str(args.loops)],
                              cwd=ROOT_DIR, env=env)
        results.append({**v, "udid": udid, "status": "ok" if proc.returncode == 0 else f"exit {proc.returncode}",
                        "screenshots": str(ROOT_DIR / args.model / f"screenshots_{v['variant']}")})

    # Restore the iOS defaults on the simulators whose settings were changed
    for udid in touched:
        subprocess.run(["xcrun", "simctl", "ui", udid, "appearance", "light"], capture_output=True)
        subprocess.run(["xcrun", "simctl", "ui", udid, "content_size", "large"], capture_output=True)

    print("\n" + json.dumps(results, ensure_ascii=False, indent=2))
    return 0 if all(r["status"] == "ok" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
