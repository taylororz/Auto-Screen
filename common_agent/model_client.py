"""
Multi-Model Client Gateway for Auto-Screen.
Supports Anthropic Claude, OpenAI, and Untrained Local Qwen-VL backends
in both Set-of-Mark (SoM) and Direct Grounding (DG) modes.
"""

import os
import json
import time
import ssl
from pathlib import Path
from urllib import error, request
from .base_cv import encode_image, parse_dg_coordinates


def _get_ssl_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


# ==============================================================================
# 1. Anthropic Claude Client
# ==============================================================================
class ClaudeClient:
    def __init__(self, model_name=None):
        self.model_name = model_name or os.environ.get("CLAUDE_AGENT_MODEL", "claude-sonnet-5")
        self.api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY")
        base_url = os.environ.get("ANTHROPIC_BASE_URL") or os.environ.get("CLAUDE_BASE_URL") or "https://api.anthropic.com"
        self.endpoint = f"{base_url.rstrip('/')}/v1/messages" if not base_url.endswith("/messages") else base_url

    def query_som(self, annotated_img_path, instruction, elements, language=None, timeout=60):
        if not self.api_key:
            return None, "ANTHROPIC_API_KEY is not set"

        element_summary = [
            {"id": el["id"], "center": el["center"], "bbox": el["bbox"]}
            for el in elements
        ]
        lang_hint = f"Interface language/locale: {language}." if language else "Interface language/locale: unknown."
        prompt = (
            "You are a pure-vision mobile UI test agent. The screenshot contains red Set-of-Mark labels like [1], [2], [3]. "
            "Given the user instruction, choose the single label ID that should be clicked. "
            "Use only visible UI evidence and the numbered labels. If no label matches, return target_id null. "
            "Return JSON only with this schema: "
            '{"target_id":number|null,"confidence":0.0,"rationale":"short reason","candidate_ids":[number]}. '
            f"{lang_hint} User instruction: {instruction}. "
            f"Known label metadata: {json.dumps(element_summary, ensure_ascii=False)}"
        )

        encoded_img, mime_type, _ = encode_image(annotated_img_path)
        payload = {
            "model": self.model_name,
            "max_tokens": 1024,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": encoded_img,
                        },
                    },
                ],
            }],
        }

        req = request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        with request.urlopen(req, timeout=timeout, context=_get_ssl_context()) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            text_parts = [item["text"] for item in data.get("content", []) if item.get("type") == "text"]
            return "\n".join(text_parts), None

    def query_dg(self, image_path, instruction, image_width=1206, image_height=2622, language=None, timeout=60):
        if not self.api_key:
            return None, "ANTHROPIC_API_KEY is not set"

        lang_hint = f"Language: {language}." if language else ""
        prompt = (
            f"You are a pure-vision mobile UI agent. Locate the UI element matching this instruction: '{instruction}'. {lang_hint}\n"
            f"The screenshot has resolution {image_width}x{image_height}.\n"
            f"Return ONLY valid JSON with target click coordinates [x, y] in range x=[0, {image_width}], y=[0, {image_height}]:\n"
            '{"point": [x, y], "confidence": 1.0, "rationale": "short explanation"}'
        )

        encoded_img, mime_type, _ = encode_image(image_path)
        payload = {
            "model": self.model_name,
            "max_tokens": 512,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": encoded_img,
                        },
                    },
                ],
            }],
        }

        req = request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        with request.urlopen(req, timeout=timeout, context=_get_ssl_context()) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            text_parts = [item["text"] for item in data.get("content", []) if item.get("type") == "text"]
            return "\n".join(text_parts), None


# ==============================================================================
# 2. OpenAI Client (Supports Responses API & Chat Completions fallback)
# ==============================================================================
class OpenAIClient:
    def __init__(self, model_name=None):
        self.model_name = model_name or os.environ.get("OPENAI_AGENT_MODEL", "gpt-5.6-luna")
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.base_url = (os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")

    def _post(self, endpoint, payload, timeout=60):
        req = request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=timeout, context=_get_ssl_context()) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def query_som(self, annotated_img_path, instruction, elements, language=None, timeout=60):
        if not self.api_key:
            return None, "OPENAI_API_KEY is not set"

        element_summary = [
            {"id": el["id"], "center": el["center"], "bbox": el["bbox"]}
            for el in elements
        ]
        lang_hint = f"Interface language/locale: {language}." if language else "Interface language/locale: unknown."
        prompt = (
            "You are a pure-vision mobile UI agent. The screenshot contains red Set-of-Mark labels like [1], [2], [3]. "
            "Given the user instruction, choose the single label ID that should be clicked. "
            "Use only visible UI evidence and the numbered labels. If no label matches, return target_id null. "
            "Return JSON only with this schema: "
            '{"target_id":number|null,"confidence":0.0,"rationale":"short reason","candidate_ids":[number]}. '
            f"{lang_hint} User instruction: {instruction}. "
            f"Known label metadata: {json.dumps(element_summary, ensure_ascii=False)}"
        )

        _, _, data_url = encode_image(annotated_img_path)

        # 1. Try Responses API
        try:
            payload_resp = {
                "model": self.model_name,
                "input": [{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": data_url},
                    ],
                }],
            }
            resp = self._post(f"{self.base_url}/responses", payload_resp, timeout=timeout)
            if "output_text" in resp:
                return resp["output_text"], None
            text_parts = []
            for item in resp.get("output", []):
                for content in item.get("content", []):
                    if content.get("type") in {"output_text", "text"} and content.get("text"):
                        text_parts.append(content["text"])
            if text_parts:
                return "\n".join(text_parts), None
        except Exception:
            pass

        # 2. Fallback to standard chat completions
        payload_chat = {
            "model": self.model_name,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
        }
        resp = self._post(f"{self.base_url}/chat/completions", payload_chat, timeout=timeout)
        content = resp["choices"][0]["message"]["content"]
        return content, None

    def query_dg(self, image_path, instruction, image_width=1206, image_height=2622, language=None, timeout=60):
        if not self.api_key:
            return None, "OPENAI_API_KEY is not set"

        lang_hint = f"Language: {language}." if language else ""
        prompt = (
            f"You are a pure-vision mobile UI agent. Locate the UI element matching this instruction: '{instruction}'. {lang_hint}\n"
            f"The screenshot has resolution {image_width}x{image_height}.\n"
            f"Return ONLY valid JSON with target click coordinates [x, y] in range x=[0, {image_width}], y=[0, {image_height}]:\n"
            '{"point": [x, y], "confidence": 1.0, "rationale": "short explanation"}'
        )

        _, _, data_url = encode_image(image_path)

        # 1. Try Responses API
        try:
            payload_resp = {
                "model": self.model_name,
                "input": [{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": data_url},
                    ],
                }],
            }
            resp = self._post(f"{self.base_url}/responses", payload_resp, timeout=timeout)
            if "output_text" in resp:
                return resp["output_text"], None
            text_parts = []
            for item in resp.get("output", []):
                for content in item.get("content", []):
                    if content.get("type") in {"output_text", "text"} and content.get("text"):
                        text_parts.append(content["text"])
            if text_parts:
                return "\n".join(text_parts), None
        except Exception:
            pass

        # 2. Fallback to standard chat completions
        payload_chat = {
            "model": self.model_name,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
        }
        resp = self._post(f"{self.base_url}/chat/completions", payload_chat, timeout=timeout)
        content = resp["choices"][0]["message"]["content"]
        return content, None


# ==============================================================================
# 3. Untrained Qwen-VL Client (Local Transformers)
# ==============================================================================
_qwen_untrain_model = None
_qwen_untrain_processor = None


class QwenUntrainClient:
    def __init__(self, model_name=None):
        self.model_name = model_name or os.environ.get("BASE_MODEL_NAME", "Qwen/Qwen3-VL-4B-Instruct")

    def _get_model_and_processor(self):
        global _qwen_untrain_model, _qwen_untrain_processor
        if _qwen_untrain_model is None or _qwen_untrain_processor is None:
            print(f"🔄 Loading untrained base model: {self.model_name}...")
            import torch
            from transformers import AutoProcessor

            try:
                from transformers import Qwen3VLForConditionalGeneration
                model_cls = Qwen3VLForConditionalGeneration
            except ImportError:
                try:
                    from transformers import Qwen2_5_VLForConditionalGeneration
                    model_cls = Qwen2_5_VLForConditionalGeneration
                except ImportError:
                    from transformers import AutoModelForVision2Seq
                    model_cls = AutoModelForVision2Seq

            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
            dtype = torch.bfloat16 if torch.cuda.is_available() or torch.backends.mps.is_available() else torch.float32

            load_kwargs = {"torch_dtype": dtype}
            if torch.cuda.is_available():
                load_kwargs["device_map"] = "auto"

            # Prioritize loading from local HuggingFace cache to prevent intermittent HF Hub HTTP 500 / rate limit errors
            try:
                _qwen_untrain_model = model_cls.from_pretrained(
                    self.model_name,
                    local_files_only=True,
                    **load_kwargs
                )
            except Exception:
                _qwen_untrain_model = model_cls.from_pretrained(self.model_name, **load_kwargs)

            if not torch.cuda.is_available():
                _qwen_untrain_model.to(device)

            try:
                _qwen_untrain_processor = AutoProcessor.from_pretrained(
                    self.model_name,
                    local_files_only=True,
                    min_pixels=256 * 28 * 28,
                    max_pixels=1024 * 28 * 28
                )
            except Exception:
                _qwen_untrain_processor = AutoProcessor.from_pretrained(
                    self.model_name,
                    min_pixels=256 * 28 * 28,
                    max_pixels=1024 * 28 * 28
                )
            print("✅ Untrained base Qwen-VL model loaded successfully!")
        return _qwen_untrain_model, _qwen_untrain_processor

    def query_som(self, annotated_img_path, instruction, elements, language=None, timeout=60):
        import torch
        from qwen_vl_utils import process_vision_info

        model, processor = self._get_model_and_processor()
        prompt_text = (
            f"<image>This is a UI testing task. "
            f"Find the element that matches this instruction: '{instruction}'. "
            f"Output the target ID in JSON format: {{\"target_id\": <number>}}."
        )

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": str(annotated_img_path)},
                {"type": "text", "text": prompt_text},
            ],
        }]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        device = next(model.parameters()).device

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=160, do_sample=False)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return output_text, None

    def query_dg(self, image_path, instruction, image_width=1206, image_height=2622, language=None, timeout=60):
        import torch
        from qwen_vl_utils import process_vision_info

        model, processor = self._get_model_and_processor()
        prompt_text = (
            f"<image>This is a UI testing task. Locate the UI element that matches this instruction: '{instruction}'. "
            f"Output the target interaction point in normalized coordinates [0, 1000] in JSON format: "
            f'{{"point": [x, y], "confidence": 1.0, "rationale": "..."}}'
        )

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt_text},
            ],
        }]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        device = next(model.parameters()).device

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=160, do_sample=False)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return output_text, None
