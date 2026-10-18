"""
Untrained Qwen-VL Visual Agent - Set-of-Mark (SoM) Mode
Powered by common_agent unified engine.
"""

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common_agent.agent_facade import make_agent_facade, VisualNavigationError

_agent = make_agent_facade(
    provider="qwen_untrain",
    mode="som",
    model_name=os.environ.get("BASE_MODEL_NAME", "Qwen/Qwen3-VL-4B-Instruct"),
    model_dir=Path(__file__).resolve().parent,
)

annotate_ui_elements = _agent.annotate_ui_elements
ask_ai_for_element_id = _agent.ask_ai_for_element_id
heuristic_fallback = _agent.heuristic_fallback
is_page_changed = _agent.is_page_changed
log_interaction = _agent.log_interaction
