#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import generic_engine

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Visual GUI Test on Model")
    parser.add_argument("--config", "-c", default=None, help="App config path or name (e.g. 'bluesky', 'katapp')")
    parser.add_argument("--loops", "-l", type=int, default=int(os.environ.get("FLYWHEEL_LOOPS", 1)), help="Number of loops")
    parser.add_argument("--direct-grounding", "-dg", action="store_true", help="Use Direct Grounding mode")
    parser.add_argument("--agent-file", "-a", default=None, help="Agent script filename")
    args, unknown = parser.parse_known_args()

    model_dir = Path(__file__).resolve().parent
    agent_file = args.agent_file
    if not agent_file:
        use_dg = args.direct_grounding or os.environ.get("USE_DIRECT_GROUNDING") in ("1", "true", "True")
        dg_file = model_dir / "visual_agent_DG.py"
        agent_file = "visual_agent_DG.py" if (use_dg and dg_file.exists()) else "visual_agent.py"

    config_path = generic_engine.resolve_config_path(args.config)
    print(f"🚀 Launching Configuration-Driven Engine for {model_dir.name} (Agent: {agent_file}, Config: {config_path.name})...")
    engine = generic_engine.GenericGUITestEngine(
        config_path=config_path,
        model_dir=model_dir,
        loops=args.loops,
        agent_filename=agent_file
    )
    engine.run()
