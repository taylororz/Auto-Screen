"""
Common Visual Agent Engine for Auto-Screen.
Unifies OpenCV computer vision, Set-of-Mark (SoM) annotations, Direct Grounding (DG),
and multi-model LLM inference across Anthropic Claude, OpenAI, and Qwen.
"""

from .base_cv import (
    annotate_ui_elements_som,
    annotate_ui_elements_dg,
    is_page_changed,
    heuristic_fallback_som,
    heuristic_fallback_dg,
    parse_dg_coordinates,
    write_interaction_log,
)
from .agent_facade import make_agent_facade, VisualNavigationError

__all__ = [
    "annotate_ui_elements_som",
    "annotate_ui_elements_dg",
    "is_page_changed",
    "heuristic_fallback_som",
    "heuristic_fallback_dg",
    "parse_dg_coordinates",
    "write_interaction_log",
    "make_agent_facade",
    "VisualNavigationError",
]
