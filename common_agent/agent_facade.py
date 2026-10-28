"""
Unified Visual Agent Facade.
Provides consistent generic_engine compatible interfaces for any model and mode (SoM or DG).
"""

import os
import json
import time
from pathlib import Path
import cv2

from .base_cv import (
    annotate_ui_elements_som,
    annotate_ui_elements_dg,
    is_page_changed as cv_is_page_changed,
    heuristic_fallback_som,
    heuristic_fallback_dg,
    parse_dg_coordinates,
    write_interaction_log,
)
from .model_client import ClaudeClient, OpenAIClient, QwenUntrainClient


class VisualNavigationError(Exception):
    """Raised when the visual agent fails to locate a required UI element."""
    pass


class AgentFacade:
    def __init__(self, provider="openai", mode="som", model_name=None, model_dir=None):
        self.provider = provider.lower()
        self.mode = mode.lower()
        self.model_name = model_name
        self.model_dir = Path(model_dir).resolve() if model_dir else Path.cwd()
        self.last_predicted_coordinates = {}

        if self.provider in ("claude", "anthropic"):
            self.client = ClaudeClient(model_name=self.model_name)
        elif self.provider in ("openai", "gpt"):
            self.client = OpenAIClient(model_name=self.model_name)
        elif self.provider in ("qwen_untrain", "untrained", "qwen_base"):
            self.client = QwenUntrainClient(model_name=self.model_name)
        else:
            raise ValueError(f"Unknown provider '{provider}'. Must be one of: anthropic, openai, qwen_untrain")

    def annotate_ui_elements(self, image_path, output_path=None, **kwargs):
        if self.mode in ("dg", "direct", "direct_grounding"):
            dynamic_map, target_img, elements = annotate_ui_elements_dg(image_path, output_path=output_path, **kwargs)
            self.last_predicted_coordinates = dynamic_map
            return self.last_predicted_coordinates, target_img, elements
        else:
            return annotate_ui_elements_som(image_path, output_path=output_path, **kwargs)

    def heuristic_fallback(self, instruction, elements=None):
        if self.mode in ("dg", "direct", "direct_grounding"):
            pt = heuristic_fallback_dg(instruction)
            self.last_predicted_coordinates["1"] = pt
            self.last_predicted_coordinates[1] = pt
            return 1
        else:
            return heuristic_fallback_som(instruction, elements or [])

    def is_page_changed(self, img_path_a, img_path_b, threshold=0.005):
        return cv_is_page_changed(img_path_a, img_path_b, threshold=threshold)

    def log_interaction(
        self,
        language,
        step_name,
        instruction,
        action,
        target_id,
        center_x,
        center_y,
        screenshot_path,
        annotated_image_path,
        decision,
        tap_x=None,
        tap_y=None,
        run_id=None,
        loop_index=None,
        step_index=None,
    ):
        log_dir = self.model_dir / "diff_results"
        write_interaction_log(
            log_dir=log_dir,
            mode=self.mode,
            language=language,
            step_name=step_name,
            instruction=instruction,
            action=action,
            target_id=target_id,
            center_x=center_x,
            center_y=center_y,
            screenshot_path=screenshot_path,
            annotated_image_path=annotated_image_path,
            decision=decision,
            tap_x=tap_x,
            tap_y=tap_y,
            run_id=run_id,
            loop_index=loop_index,
            step_index=step_index,
        )

    def ask_ai_for_element_id(self, annotated_img_path, instruction, elements=None, language=None, model=None, timeout=60):
        if self.mode in ("dg", "direct", "direct_grounding"):
            return self._ask_ai_dg(annotated_img_path, instruction, elements=elements, language=language, timeout=timeout)
        else:
            return self._ask_ai_som(annotated_img_path, instruction, elements=elements or [], language=language, timeout=timeout)

    def _ask_ai_som(self, annotated_img_path, instruction, elements, language=None, timeout=60):
        start_time = time.time()
        try:
            raw_output, err = self.client.query_som(annotated_img_path, instruction, elements, language=language, timeout=timeout)
            if err:
                print(f"⚠️ API Error ({err}). Triggering heuristic fallback.")
                target_id = heuristic_fallback_som(instruction, elements)
                return {
                    "status": "error",
                    "reason": str(err),
                    "target_id": target_id,
                    "confidence": 0.1,
                    "rationale": f"API error fallback: {err}",
                    "time_ms": (time.time() - start_time) * 1000,
                }

            # Parse JSON
            parsed = {}
            try:
                parsed = json.loads(raw_output)
            except Exception:
                start = raw_output.find("{")
                end = raw_output.rfind("}")
                if start != -1 and end != -1 and end > start:
                    try:
                        parsed = json.loads(raw_output[start:end + 1])
                    except Exception:
                        pass

            target_id = parsed.get("target_id")
            confidence = float(parsed.get("confidence", 1.0 if target_id is not None else 0.0))

            valid_ids = {el["id"] for el in elements}
            if target_id is not None and target_id not in valid_ids:
                if isinstance(target_id, str) and target_id.isdigit() and int(target_id) in valid_ids:
                    target_id = int(target_id)
                else:
                    target_id = None

            used_fallback = False
            if target_id is None:
                target_id = heuristic_fallback_som(instruction, elements)
                used_fallback = True
                print(f"⚠️ VLM output could not resolve target ID (raw: '{raw_output}'). Using fallback: target_id={target_id}")

            decision = {
                "status": "analyzed_fallback" if used_fallback else f"analyzed_{self.provider}",
                "model": self.client.model_name,
                "target_id": target_id,
                "confidence": confidence,
                "rationale": parsed.get("rationale") or raw_output,
                "candidate_ids": parsed.get("candidate_ids", []),
                "time_ms": (time.time() - start_time) * 1000,
            }
            print(f"🤖 [{self.provider.upper()} SoM] target_id={decision['target_id']} confidence={decision['confidence']}")
            return decision

        except Exception as exc:
            print(f"⚠️ [{self.provider.upper()} SoM] Inference exception: {exc}. Using fallback.")
            target_id = heuristic_fallback_som(instruction, elements)
            return {
                "status": "error",
                "reason": str(exc),
                "target_id": target_id,
                "confidence": 0.1,
                "rationale": f"Model error: {exc}",
                "time_ms": (time.time() - start_time) * 1000,
            }

    def _ask_ai_dg(self, image_path, instruction, elements=None, language=None, timeout=60):
        img = cv2.imread(str(image_path))
        if img is not None:
            img_h, img_w = img.shape[:2]
        else:
            img_w, img_h = 1206, 2622

        start_time = time.time()
        try:
            raw_output, err = self.client.query_dg(
                image_path, instruction, image_width=img_w, image_height=img_h, language=language, timeout=timeout
            )
            if err:
                print(f"⚠️ API Error ({err}). Triggering DG heuristic fallback.")
                target_x, target_y = heuristic_fallback_dg(instruction, image_size=(img_w, img_h))
                decision = {
                    "status": "error",
                    "reason": str(err),
                    "target_id": 1,
                    "point": [target_x, target_y],
                    "center_x": target_x,
                    "center_y": target_y,
                    "confidence": 0.1,
                    "rationale": f"API error fallback: {err}",
                    "time_ms": (time.time() - start_time) * 1000,
                }
                self.last_predicted_coordinates["1"] = (target_x, target_y)
                self.last_predicted_coordinates[1] = (target_x, target_y)
                return decision

            print(f"🤖 [{self.provider.upper()} Direct Grounding] Raw: {raw_output}")
            # Qwen prompts (QwenUntrainClient.query_dg) request normalized [0, 1000] coordinates,
            # identical to the LoRA agent (Qwen_trained/visual_agent_DG.py, step 4 "normalization scaling").
            # Claude/OpenAI prompts request absolute pixel coordinates and must not be rescaled.
            is_qwen = self.provider in ("qwen", "qwen_untrain", "untrained", "qwen_base")
            target_x, target_y, confidence, rationale = parse_dg_coordinates(
                raw_output, image_width=img_w, image_height=img_h, is_normalized_1000=is_qwen
            )

            used_fallback = False
            if target_x is None or target_y is None:
                print(f"⚠️ [{self.provider.upper()} DG] Coordinates not found in output. Triggering fallback.")
                target_x, target_y = heuristic_fallback_dg(instruction, image_size=(img_w, img_h))
                used_fallback = True
                confidence = 0.5

            decision = {
                "status": "analyzed_fallback" if used_fallback else "analyzed_direct_grounding",
                "model": self.client.model_name,
                "target_id": 1,
                "point": [target_x, target_y],
                "center_x": target_x,
                "center_y": target_y,
                "confidence": confidence,
                "rationale": rationale,
                "raw_output": raw_output,
                "time_ms": (time.time() - start_time) * 1000,
            }
            self.last_predicted_coordinates["1"] = (target_x, target_y)
            self.last_predicted_coordinates[1] = (target_x, target_y)
            print(f"🤖 [{self.provider.upper()} DG] Target: ({target_x}, {target_y}) confidence={confidence}")
            return decision

        except Exception as exc:
            print(f"⚠️ [{self.provider.upper()} DG] Exception: {exc}. Triggering fallback.")
            target_x, target_y = heuristic_fallback_dg(instruction, image_size=(img_w, img_h))
            decision = {
                "status": "error",
                "reason": str(exc),
                "target_id": 1,
                "point": [target_x, target_y],
                "center_x": target_x,
                "center_y": target_y,
                "confidence": 0.1,
                "rationale": f"Inference error fallback: {exc}",
                "time_ms": (time.time() - start_time) * 1000,
            }
            self.last_predicted_coordinates["1"] = (target_x, target_y)
            self.last_predicted_coordinates[1] = (target_x, target_y)
            return decision


def make_agent_facade(provider="openai", mode="som", model_name=None, model_dir=None):
    """Factory creating an AgentFacade instance."""
    return AgentFacade(provider=provider, mode=mode, model_name=model_name, model_dir=model_dir)
