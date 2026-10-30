#!/usr/bin/env python3
"""
🚀 Auto-Screen: Universal Visual GUI Automation, Regression Testing & Data Flywheel Suite
Master's Thesis Tool for iOS Applications.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import generic_engine


def cmd_list(args):
    """Lists available app configs, models, and datasets."""
    print("==================================================================")
    print("📱 Auto-Screen System Overview")
    print("==================================================================")

    print("\n[📁 Available App Configurations (configs/)]")
    configs_dir = ROOT_DIR / "configs"
    if configs_dir.exists():
        for f in sorted(configs_dir.glob("*.json")):
            tag = " (Template)" if "template" in f.name else ""
            print(f"  • {f.stem:<20} -> {f.name}{tag}")
    else:
        print("  (None found)")

    print("\n[🤖 Available Models & Runners]")
    model_dirs = ["Qwen_trained", "Qwen_untrain", "claude", "openAI_6.0_Luna", "openAI_6.0_Sol"]
    for md in model_dirs:
        p = ROOT_DIR / md
        exists_str = "✅ Ready" if p.exists() else "❌ Missing"
        sub_models = []
        if (p / "Qwen3-VL-4B-V2").exists():
            sub_models.append("Qwen3-VL-4B-V2")
        extra = f" ({', '.join(sub_models)})" if sub_models else ""
        print(f"  • {md:<20} [{exists_str}]{extra}")

    print("\n[📊 Datasets & Logs]")
    ds_dir = ROOT_DIR / "datasets"
    if ds_dir.exists():
        for d in ds_dir.iterdir():
            if d.is_dir():
                print(f"  • {d.name} (Directory)")
    qwen_diff = ROOT_DIR / "Qwen_trained" / "diff_results"
    if qwen_diff.exists():
        logs = list(qwen_diff.glob("*.jsonl"))
        for l in logs:
            print(f"  • Qwen_trained/{l.name} ({l.stat().st_size // 1024} KB)")

    print("==================================================================\n")


def cmd_test(args):
    """Runs GUI automation tests on an iOS application."""
    config_path = generic_engine.resolve_config_path(args.config or args.app)

    # Resolve model directory
    model_dir_name = args.model
    model_dir = ROOT_DIR / model_dir_name
    if not model_dir.exists():
        # Try case-insensitive or partial match
        matched = [d for d in ROOT_DIR.iterdir() if d.is_dir() and model_dir_name.lower() in d.name.lower()]
        if matched:
            model_dir = matched[0]
        else:
            raise FileNotFoundError(f"Model directory '{model_dir_name}' not found under {ROOT_DIR}")

    # Resolve agent mode
    mode = args.mode.lower()
    if mode in ("dg", "direct", "direct_grounding"):
        agent_file = "visual_agent_DG.py"
    else:
        agent_file = "visual_agent.py"

    target_agent_path = model_dir / agent_file
    if not target_agent_path.exists():
        print(f"⚠️ Warning: {agent_file} not found in {model_dir.name}, falling back to visual_agent.py")
        agent_file = "visual_agent.py"

    print(f"\n==================================================================")
    print(f"🚀 Auto-Screen Test Runner")
    print(f"  • App Configuration: {config_path}")
    print(f"  • Model Directory:   {model_dir.name}")
    print(f"  • Agent Mode:        {agent_file}")
    print(f"  • Test Loops:        {args.loops}")
    print(f"==================================================================\n")

    engine = generic_engine.GenericGUITestEngine(
        config_path=config_path,
        model_dir=model_dir,
        loops=args.loops,
        agent_filename=agent_file,
        reset_between_loops=not getattr(args, "no_loop_reset", False)
    )
    engine.run()


def cmd_review(args):
    """Launches the Gradio Human Review & Data Flywheel UI."""
    mode = args.mode.lower()
    port = args.port

    if mode in ("dg", "direct", "direct_grounding"):
        script = ROOT_DIR / "review_tool_DG.py"
        mode_label = "Direct Grounding (DG)"
    else:
        script = ROOT_DIR / "review_tool.py"
        mode_label = "Set-of-Mark (SoM)"

    print(f"\n🚀 Launching Auto-Screen {mode_label} Review Tool on port {port}...")
    env = os.environ.copy()
    env["GRADIO_SERVER_PORT"] = str(port)
    subprocess.run([sys.executable, str(script)], env=env)


def cmd_compare(args):
    """Executes multi-algorithm visual regression comparison."""
    compare_script = ROOT_DIR / "compare_images.py"
    cmd = [sys.executable, str(compare_script)]
    if getattr(args, "app", None):
        cmd.extend(["--app", args.app])
    if getattr(args, "model", None):
        cmd.extend(["--model", args.model])
    if getattr(args, "flow", False):
        cmd.append("--flow")
    if getattr(args, "only_diffs", False):
        cmd.append("--only-diffs")
    if getattr(args, "pages", None):
        cmd.extend(["--pages"] + args.pages)
    if args.baseline:
        cmd.extend(["--baseline", args.baseline])
    if args.actual:
        cmd.extend(["--actual", args.actual])
    if args.output:
        cmd.extend(["--output", args.output])
    subprocess.run(cmd)


def cmd_dataset(args):
    """Converts reviewed interactions into training format."""
    tool_script = ROOT_DIR / "tools" / "prepare_dataset.py"
    cmd = [sys.executable, str(tool_script)]
    if args.input:
        cmd.extend(["--input", args.input])
    if args.output:
        cmd.extend(["--output", args.output])
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Auto-Screen: Universal Visual GUI Automation & Data Flywheel Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python auto_screen.py list
  python auto_screen.py test --app bluesky --model Qwen_trained --mode dg
  python auto_screen.py test --app katapp --model claude
  python auto_screen.py test --config configs/template_app.json
  python auto_screen.py review --mode dg --port 7861
  python auto_screen.py compare --baseline ./baseline_dir --actual ./actual_dir
        """
    )

    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # 1. list
    p_list = subparsers.add_parser("list", help="List available apps, models, and datasets")
    p_list.set_defaults(func=cmd_list)

    # 2. test / run
    for cmd_alias in ["test", "run"]:
        p_test = subparsers.add_parser(cmd_alias, help="Run GUI automation test on an app")
        p_test.add_argument("--app", "-a", default=None, help="App name shortcut (e.g. 'bluesky', 'katapp')")
        p_test.add_argument("--config", "-c", default=None, help="Path to declarative test config JSON")
        p_test.add_argument("--model", "-m", default="Qwen_trained", help="Model directory (default: Qwen_trained)")
        p_test.add_argument("--mode", choices=["dg", "som"], default="dg", help="Grounding mode: 'dg' (Direct Grounding) or 'som' (Set-of-Mark)")
        p_test.add_argument("--loops", "-l", type=int, default=1, help="Number of test execution loops (default: 1)")
        p_test.add_argument("--no-loop-reset", action="store_true", help="Disable lightweight app reset between test loops")
        p_test.set_defaults(func=cmd_test)

    # 3. review
    p_rev = subparsers.add_parser("review", help="Launch Human Review & Annotation Web UI")
    p_rev.add_argument("--mode", choices=["dg", "som"], default="dg", help="Review mode: 'dg' or 'som'")
    p_rev.add_argument("--port", "-p", type=int, default=7860, help="Gradio server port (default: 7860)")
    p_rev.set_defaults(func=cmd_review)

    # 4. compare
    p_cmp = subparsers.add_parser("compare", help="Run visual regression screenshot comparison")
    p_cmp.add_argument("--app", "-A", default=None, help="Application name to compare (e.g. katapp, bluesky). Reads configuration from configs/<app>.json")
    p_cmp.add_argument("--model", "-m", default=None, help="Model directory containing screenshots (e.g. Qwen_trained, openAI_6.0_Luna)")
    p_cmp.add_argument("--flow", "-f", action="store_true", help="Compare all numbered sequential flow step screenshots (01_*, 02_*, etc.)")
    p_cmp.add_argument("--only-diffs", action="store_true", help="Only compare and report pages with visual differences, ignore identical pages")
    p_cmp.add_argument("--pages", "-p", nargs="*", default=None, help="Specific page names to compare (or 'flow', 'all')")
    p_cmp.add_argument("--baseline", "-b", help="Baseline images directory or file")
    p_cmp.add_argument("--actual", "-a", help="Actual images directory or file")
    p_cmp.add_argument("--output", "-o", help="Diff output directory")
    p_cmp.set_defaults(func=cmd_compare)

    # 5. dataset
    p_ds = subparsers.add_parser("dataset", help="Format human-reviewed interactions for fine-tuning")
    p_ds.add_argument("--input", "-i", help="Input finetune_dataset.jsonl path")
    p_ds.add_argument("--output", "-o", help="Output JSON path for LLaMA-Factory / ShareGPT")
    p_ds.set_defaults(func=cmd_dataset)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
