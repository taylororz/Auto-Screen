#!/usr/bin/env python3
"""
Generic GUI Agent Test Engine 
Configuration-Driven Architecture for Mobile App Visual Regression Testing.
"""

import os
import re
import sys
import time
import json
import shutil
import argparse
import importlib.util
from pathlib import Path
from datetime import datetime

import cv2
from appium import webdriver
from appium.options.ios import XCUITestOptions


def resolve_config_path(config_path=None):
    """
    Resolves declarative test flow configuration JSON file.
    Supports:
    - Direct file path (e.g. 'configs/bluesky.json' or '/path/to/app.json')
    - App name shortcut (e.g. 'bluesky' -> 'configs/bluesky.json', 'katapp' -> 'configs/katapp.json')
    - Environment variables APP_CONFIG or TEST_CONFIG
    - Default fallback to 'configs/bluesky.json' or first .json in 'configs/'
    """
    root_dir = Path(__file__).resolve().parent
    candidates = []
    if config_path:
        cp = Path(config_path)
        if cp.exists() and cp.is_file():
            return cp.resolve()
        candidates.append(cp)
        str_path = str(config_path)
        if not str_path.endswith(".json"):
            candidates.append(root_dir / "configs" / f"{str_path}.json")
            candidates.append(root_dir / f"{str_path}.json")
        else:
            candidates.append(root_dir / "configs" / cp.name)
            candidates.append(root_dir / "configs" / cp)
            candidates.append(root_dir / cp)

    env_cfg = os.environ.get("APP_CONFIG") or os.environ.get("TEST_CONFIG")
    if env_cfg:
        candidates.append(Path(env_cfg))
        candidates.append(root_dir / "configs" / f"{env_cfg}.json")
        candidates.append(root_dir / "configs" / env_cfg)

    candidates.append(root_dir / "configs" / "bluesky.json")
    candidates.append(root_dir / "configs" / "katapp.json")
    candidates.append(root_dir / "test_config.json")

    for cand in candidates:
        if cand.exists() and cand.is_file():
            return cand.resolve()

    configs_dir = root_dir / "configs"
    if configs_dir.exists():
        json_files = sorted(configs_dir.glob("*.json"))
        for jf in json_files:
            if "template" not in jf.name:
                return jf.resolve()
        if json_files:
            return json_files[0].resolve()

    raise FileNotFoundError(f"Configuration file not found. Checked: {[str(c) for c in candidates]}")


def load_config(config_path=None):
    """Loads declarative test flow configuration JSON file."""
    config_file = resolve_config_path(config_path)
    print(f"📄 Loaded App Configuration: {config_file}")
    with open(config_file, "r", encoding="utf-8") as f:
        return json.load(f)


def import_visual_agent(model_dir, agent_filename=None):
    """Dynamically imports visual_agent.py or specified agent file (e.g. visual_agent_DG.py) from the model directory."""
    if agent_filename is None:
        agent_filename = "visual_agent_DG.py" if os.environ.get("USE_DIRECT_GROUNDING") in ("1", "true", "True") else "visual_agent.py"
    
    model_path = Path(model_dir).resolve()
    agent_file = model_path / agent_filename
    if not agent_file.exists():
        # Fallback to current script directory or root
        agent_file = Path(__file__).resolve().parent / agent_filename
    if not agent_file.exists():
        # Final fallback to standard visual_agent.py
        agent_file = model_path / "visual_agent.py"
    
    spec = importlib.util.spec_from_file_location(f"visual_agent_{model_path.name}_{Path(agent_filename).stem}", agent_file)
    agent_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = agent_module
    spec.loader.exec_module(agent_module)
    return agent_module


def load_env_file(root_dir=None):
    """Loads .env file into os.environ if present."""
    search_dirs = [Path(root_dir).resolve()] if root_dir else []
    search_dirs.append(Path(__file__).resolve().parent)
    for sdir in search_dirs:
        env_file = sdir / ".env"
        if env_file.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(env_file)
            except ImportError:
                with open(env_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            os.environ[k.strip()] = v.strip().strip("'\"")
            break


class GenericGUITestEngine:
    def __init__(self, config_path=None, model_dir=".", loops=1, agent_filename=None, reset_between_loops=True):
        self.config_path = config_path
        self.model_dir = Path(model_dir).resolve()
        load_env_file(self.model_dir.parent)
        self.config = load_config(config_path)
        self.app_cfg = self.config["app_config"]
        self.loops = loops
        self.agent = import_visual_agent(model_dir, agent_filename=agent_filename)
        self.driver = None
        self.reset_between_loops = reset_between_loops
        self._cached_window_size = None

        # Prepare log and screenshot output directories under model_dir
        self.diff_dir = self.model_dir / "diff_results"
        self.diff_dir.mkdir(parents=True, exist_ok=True)
        self.screenshot_root = self.model_dir / "screenshots"
        self.screenshot_root.mkdir(parents=True, exist_ok=True)

    def get_window_size(self):
        """Safely returns cached window size or queries driver with fallback to avoid session drop crashes."""
        if getattr(self, "_cached_window_size", None):
            return self._cached_window_size
        try:
            if self.driver:
                self._cached_window_size = self.driver.get_window_size()
                return self._cached_window_size
        except Exception as e:
            print(f"⚠️ Failed to query window_size ({e}), using default fallback.")
        self._cached_window_size = {"width": 393, "height": 852}
        return self._cached_window_size

    def init_driver(self):
        """Initializes Appium XCUITest options from config."""
        options = XCUITestOptions()
        options.platform_name = self.app_cfg.get("platform_name", "iOS")
        # Auto-detect already booted simulator (from CI or local session)
        booted_udid = os.environ.get("AUTOSCREEN_UDID")
        booted_device_name = None
        if not booted_udid:
            try:
                res = subprocess.run(
                    ["xcrun", "simctl", "list", "devices"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                for line in res.stdout.splitlines():
                    if "(Booted)" in line and "(" in line and ")" in line:
                        booted_device_name = line.split("(")[0].strip()
                        booted_udid = line.split("(")[1].split(")")[0].strip()
                        break
            except Exception:
                pass

        if booted_udid:
            print(f"📱 Connected to booted simulator: {booted_device_name or 'Simulator'} ({booted_udid})")
            options.udid = booted_udid
            options.device_name = booted_device_name or "iPhone 16 Pro"
        else:
            options.device_name = self.app_cfg.get("device_name", "iPhone 16 Pro")

        app_target = os.environ.get("AUTOSCREEN_APP_PATH") or self.app_cfg.get("app_path", "")
        if app_target and not Path(app_target).exists():
            ci_fallback = ROOT_DIR / "build" / "ios" / "iphonesimulator" / "Runner.app"
            if ci_fallback.exists():
                app_target = str(ci_fallback)
        options.app = app_target
        options.automation_name = "XCUITest"
        options.auto_accept_alerts = True
        options.show_keychain_automatically = False
        # Set long command timeout (1 hour) to prevent Appium killing session during long VLM calls
        options.new_command_timeout = 3600
        options.set_capability("newCommandTimeout", 3600)
        # Disable Expo DevLauncher onboarding popup from source via processArguments
        options.process_arguments = {
            "args": ["-EXPO_DEV_LAUNCHER_IS_ONBOARDED", "1", "-EXPO_DEV_MENU_IS_DISABLED", "1"],
            "env": {
                "EXPO_NO_DEV_LAUNCHER": "1",
                "EXPO_PUBLIC_DEV_LAUNCHER": "0"
            }
        }
        
        appium_url = self.app_cfg.get("appium_url", "http://localhost:4723")
        print(f"🔌 Connecting to Appium server: {appium_url}")
        try:
            self.driver = webdriver.Remote(appium_url, options=options)
        except Exception as conn_err:
            err_str = str(conn_err)
            if "Connection refused" in err_str or "MaxRetryError" in err_str or "NewConnectionError" in err_str:
                print(f"\n❌ [Appium Connection Failed] Cannot connect to Appium server: {appium_url}")
                print("💡 Please make sure the Appium server is running in a terminal:")
                print("   $ appium\n")
                sys.exit(1)
            raise
        time.sleep(3)
        # Pre-cache window size on startup
        try:
            self._cached_window_size = self.driver.get_window_size()
        except Exception:
            pass
        return self.driver

    def reset_app(self, wait_after=4):
        """Forcefully terminates and reactivates app state (lightweight process restart preserving app data/auth)."""
        bundle_id = self.app_cfg.get("bundle_id", "")
        print(f"\n🔄 [Lightweight Reset] Resetting App state ({bundle_id})...")
        if not self.driver:
            self.init_driver()
        try:
            self.driver.terminate_app(bundle_id)
            time.sleep(2)
            self.driver.activate_app(bundle_id)
            time.sleep(wait_after)
            self.auto_dismiss_expo_popup()
        except Exception as e:
            print(f"⚠️ Reset App failed: {e}. Restarting Appium session...")
            try:
                self.driver.quit()
            except Exception:
                pass
            self.init_driver()
            try:
                self.driver.terminate_app(bundle_id)
                time.sleep(2)
                self.driver.activate_app(bundle_id)
                time.sleep(wait_after)
                self.auto_dismiss_expo_popup()
            except Exception as inner_e:
                print(f"⚠️ Relaunch fallback warning: {inner_e}")

    def get_native_interactive_bounds(self):
        """Extracts native interactive UI element bounding boxes (buttons, text fields, links, accessible elements) from Appium page_source XML."""
        bounds = []
        try:
            import xml.etree.ElementTree as ET
            xml_src = self.driver.page_source
            if not xml_src:
                return bounds
            root = ET.fromstring(xml_src.encode("utf-8"))
            
            interactive_types = {
                "XCUIElementTypeButton",
                "XCUIElementTypeTextField",
                "XCUIElementTypeSecureTextField",
                "XCUIElementTypeLink",
                "XCUIElementTypeSwitch",
                "XCUIElementTypeSearchField",
                "XCUIElementTypeStaticText",
                "XCUIElementTypeImage",
                "XCUIElementTypeOther",
            }
            
            window_size = self.get_window_size()
            win_w = window_size.get("width", 390)
            win_h = window_size.get("height", 844)

            for elem in root.iter():
                elem_type = elem.tag
                if elem_type not in interactive_types:
                    continue
                    
                x_str = elem.attrib.get("x")
                y_str = elem.attrib.get("y")
                w_str = elem.attrib.get("width")
                h_str = elem.attrib.get("height")
                if not (x_str and y_str and w_str and h_str):
                    continue

                fx, fy, fw, fh = float(x_str), float(y_str), float(w_str), float(h_str)

                # Filter out status bar, bottom indicator, fully off-screen, and zero-size elements
                if fx < 0 or fy < 28 or fy >= win_h - 10 or fx >= win_w:
                    continue
                if fw < 12 or fh < 12:
                    continue
                # Skip large card container wrappers (e.g. post cards) to expose inner interactive elements (Reply, Repost, Like, Share, Tabs)
                if fw > win_w * 0.85 and fh > win_h * 0.22:
                    continue

                label = elem.attrib.get("label") or elem.attrib.get("name") or elem.attrib.get("value")
                if label:
                    label = label.strip()

                is_button_or_input = elem_type in {"XCUIElementTypeButton", "XCUIElementTypeTextField", "XCUIElementTypeSecureTextField", "XCUIElementTypeLink", "XCUIElementTypeSwitch", "XCUIElementTypeSearchField"}
                is_icon_or_image = (elem_type in {"XCUIElementTypeImage", "XCUIElementTypeOther"}) and fw <= 65 and fh <= 65
                is_accessible_node = (elem.attrib.get("accessible") == "true" or elem.attrib.get("isAccessibilityElement") == "true") and label and len(label) > 0 and (fw / fh <= 6.0)
                is_text_label = elem_type == "XCUIElementTypeStaticText" and label and len(label) > 0 and len(label) < 50 and (fw / fh <= 6.0)

                if is_button_or_input or is_icon_or_image or is_accessible_node or is_text_label:
                    bounds.append((fx, fy, fw, fh))
        except Exception:
            pass
        return bounds

    def auto_dismiss_expo_popup(self):
        """Auto-detect and dismiss Expo DevLauncher popup using AI vision recognition."""
        try:
            xml_src = self.driver.page_source
            if not xml_src:
                return
            xml_lower = xml_src.lower()
            expo_keywords = ("developer menu", "development builds", "runtime version", "devlauncher", "expo-dev-menu", "expo dev-menu")
            if any(k in xml_lower for k in expo_keywords):
                print("ℹ️ Expo DevLauncher popup detected. Dismissing popup...")
                
                # Fast path: Try tapping native iOS close button if available
                try:
                    for sel in ("Close", "close", "Cancel", "cancel", "close-button"):
                        btns = self.driver.find_elements("accessibility id", sel)
                        if btns:
                            btns[0].click()
                            time.sleep(1.5)
                            print("✅ Expo popup dismissed via native accessibility button.")
                            return
                except Exception:
                    pass

                window_size = self.get_window_size()
                w_pts = window_size.get("width", 390)
                h_pts = window_size.get("height", 844)

                # 1. Capture current screenshot with popup
                popup_shot = self.diff_dir / "popup_dismiss_temp.png"
                self.driver.save_screenshot(str(popup_shot))

                # 2. Extract bounds & annotate SoM boxes
                native_bounds = []
                logical_w = window_size.get("width", 390)
                import cv2
                img_temp = cv2.imread(str(popup_shot))
                physical_w = img_temp.shape[1] if img_temp is not None else 1170
                scale_factor = physical_w / logical_w

                elements_map, annotated_path, elements = self.agent.annotate_ui_elements(
                    popup_shot,
                    native_bounds=native_bounds,
                    scale_factor=scale_factor
                )

                # 3. Query AI model to identify the popup close/dismiss button
                instruction = "Click the close button '[X]' at top right of the popup modal."
                decision = self.agent.ask_ai_for_element_id(annotated_path, instruction, elements)
                target_id = decision.get("target_id")
                is_fallback = decision.get("status") == "analyzed_fallback"

                click_pt = None
                if "center_x" in decision and "center_y" in decision and decision["center_x"] is not None:
                    cx = decision["center_x"]
                    cy = decision["center_y"]
                    click_pt = (int(cx / scale_factor), int(cy / scale_factor))
                elif not is_fallback and target_id is not None and (target_id in elements_map or str(target_id) in elements_map):
                    elem_info = elements_map.get(target_id) or elements_map.get(str(target_id))
                    if isinstance(elem_info, dict):
                        pt = elem_info.get("center_pt") or (elem_info.get("x", 0), elem_info.get("y", 0))
                        click_pt = (int(pt[0] / scale_factor), int(pt[1] / scale_factor))
                    elif isinstance(elem_info, (tuple, list)):
                        click_pt = (int(elem_info[0] / scale_factor), int(elem_info[1] / scale_factor))

                if not click_pt:
                    # Check if model described exact coordinates in rationale (e.g. around x=1065, y=1235)
                    rationale = str(decision.get("rationale", ""))
                    import re
                    match = re.search(r'(?:around\s*|at\s*|near\s*|\()?x?=?(\d{3,4})\s*,\s*y?=?(\d{3,4})\)?', rationale)
                    if match:
                        rx, ry = int(match.group(1)), int(match.group(2))
                        if rx > 500 and ry > 500:  # Physical pixels
                            click_pt = (int(rx / scale_factor), int(ry / scale_factor))
                        else:
                            click_pt = (rx, ry)
                        print(f"🤖 Extracted popup close coords from AI rationale: {click_pt}")

                if not click_pt:
                    # Fallback to center-card top-right [X] of Expo DevLauncher modal
                    click_pt = (int(w_pts * 0.88), int(h_pts * 0.47))
                    print(f"✅ Fallback tap for popup close button at ({click_pt[0]}, {click_pt[1]})")
                else:
                    print(f"🤖 AI recognized popup close button at logical ({click_pt[0]}, {click_pt[1]})")

                # Perform tap using W3C coordinates
                self.driver.tap([click_pt])
                time.sleep(2.0)

                for poll_i in range(5):
                    check_src = self.driver.page_source.lower()
                    if not any(k in check_src for k in expo_keywords):
                        print("✅ Expo DevLauncher popup successfully cleared by AI!")
                        break
                    print(f"⏳ Waiting for popup to clear ({poll_i+1}/5)...")
                    self.driver.tap([click_pt])
                    time.sleep(1.0)
        except Exception as dismiss_err:
            print(f"⚠️ AI popup auto-dismiss notice: {dismiss_err}")

    def execute_step(self, step_info, language_info, is_setup_step=False, step_index=None, loop_index=0):
        """Generic visual-agent step execution engine."""
        step_name = step_info["step_name"]
        instruction = step_info["instruction"]
        action = step_info.get("action", "click")
        page_name = step_info.get("page_name", step_name)
        fallback_id = step_info.get("fallback_id", -1)
        text_to_type = step_info.get("text", "")
        if text_to_type:
            clean_key = text_to_type.strip("[]$")
            if clean_key in os.environ:
                text_to_type = os.environ[clean_key]
        lang_folder = language_info.get("folder", "default")
        lang_name = language_info.get("language_name", "English")

        folder_dir = self.diff_dir / "states" / lang_folder
        folder_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        screenshot_path = folder_dir / f"{step_name}_{ts}.png"

        # 0. Wait before executing step if requested (e.g. waiting for bundle download)
        wait_before = step_info.get("wait_before", 0)
        if wait_before > 0:
            print(f"⏳ Waiting {wait_before}s for page/download to complete before step '{step_name}'...")
            time.sleep(wait_before)

        # Smart wait for Expo Metro bundle download (ONLY during initial setup steps)
        if is_setup_step:
            for check_idx in range(10):
                try:
                    src = self.driver.page_source.lower()
                    if "downloading" in src or "bundling" in src or "building" in src:
                        print(f"⏳ Expo JS bundle is still downloading/bundling ({check_idx+1}/10). Waiting 3s...")
                        time.sleep(3.0)
                    else:
                        break
                except Exception:
                    break

        # Auto-detect and dismiss Expo DevLauncher popup if blocking the screen (runs whenever popup appears)
        self.auto_dismiss_expo_popup()

        # Check and auto-dismiss/accept any blocking native alert (e.g. Save Password)
        try:
            alert = self.driver.switch_to.alert
            if alert:
                alert_text = alert.text
                print(f"ℹ️ Native alert detected ('{alert_text}'). Dismissing...")
                try:
                    alert.dismiss()
                except Exception:
                    alert.accept()
                time.sleep(1.0)
        except Exception:
            pass

        # 1. Capture current simulator screen (unmarked raw screenshot)
        self.driver.save_screenshot(str(screenshot_path))

        # 2. Save unmarked baseline screenshot to model screenshots directory (on first loop)
        if loop_index == 0:
            page_dir = self.screenshot_root / lang_folder
            page_dir.mkdir(parents=True, exist_ok=True)
            clean_sname = step_name.replace(" ", "_")
            
            # Save sequential flow screenshot for complete process recording
            if step_index is not None:
                prefix = f"setup_{step_index:02d}_" if is_setup_step else f"{step_index:02d}_"
                seq_img_path = page_dir / f"{prefix}{clean_sname}.png"
                shutil.copyfile(str(screenshot_path), str(seq_img_path))
                print(f"✅ Full flow screenshot saved: {seq_img_path.name}")
            
            # Save standard step name screenshot
            step_img_path = page_dir / f"{clean_sname}.png"
            shutil.copyfile(str(screenshot_path), str(step_img_path))

            # Save baseline by page_name for regression comparison (if not setup step and page_name defined)
            if page_name and not is_setup_step:
                page_img_path = page_dir / f"{page_name}.png"
                shutil.copyfile(str(screenshot_path), str(page_img_path))
                print(f"✅ Baseline screenshot saved: {page_img_path.name}")

        # Native bounds bypassed for major speedup (pure visual agent detection is faster and reliable)
        native_bounds = []
        window_size = self.get_window_size()
        logical_w = window_size.get("width", 390)
        import cv2
        img_temp = cv2.imread(str(screenshot_path))
        physical_w = img_temp.shape[1] if img_temp is not None else 1170
        scale_factor = physical_w / logical_w

        # 3. Hybrid SoM annotation (Native Bounds + Visual Enhancement)
        is_kb_shown = None
        try:
            is_kb_shown = self.driver.is_keyboard_shown()
        except Exception:
            pass

        elements_map, annotated_img_path, elements = self.agent.annotate_ui_elements(
            screenshot_path,
            is_language_choice=(step_name == "language_choice"),
            native_bounds=native_bounds,
            scale_factor=scale_factor,
            keyboard_shown=is_kb_shown
        )

        # Handle scroll/swipe gestures directly without VLM inference (generalized for any mobile app)
        action_lower = action.lower()
        instruction_lower = instruction.lower()
        if any(k in action_lower for k in ("scroll", "swipe")):
            window_size = self.get_window_size()
            width = window_size.get("width", 390)
            height = window_size.get("height", 844)

            # 1. Determine direction (down / up / right / left)
            direction = "down"
            if any(k in action_lower or k in instruction_lower for k in ("up", "top", "hoch", "nach oben", "anfang")):
                direction = "up"
            elif any(k in action_lower or k in instruction_lower for k in ("right", "rechts", "next")):
                direction = "right"
            elif any(k in action_lower or k in instruction_lower for k in ("left", "links", "prev")):
                direction = "left"

            # 2. Calculate coordinate vectors based on direction
            if direction == "down":
                start_x, end_x = width // 2, width // 2
                start_y, end_y = int(height * 0.75), int(height * 0.25)
            elif direction == "up":
                start_x, end_x = width // 2, width // 2
                start_y, end_y = int(height * 0.25), int(height * 0.75)
            elif direction == "right":
                start_x, end_x = int(width * 0.8), int(width * 0.2)
                start_y, end_y = height // 2, height // 2
            elif direction == "left":
                start_x, end_x = int(width * 0.2), int(width * 0.8)
                start_y, end_y = height // 2, height // 2

            # 3. Determine repeat count (from step_info config, semantic keywords, or regex numbers)
            repeat_count = step_info.get("repeat") or step_info.get("count") or step_info.get("times")
            if repeat_count is None:
                # Check for explicit number in action (e.g. scroll_down_3)
                num_match = re.search(r'(?:scroll|swipe).*?(\d+)', action_lower)
                if num_match:
                    repeat_count = int(num_match.group(1))
                elif any(k in action_lower or k in instruction_lower for k in ("bottom", "end", "ende", "unt", "top", "anfang")):
                    repeat_count = 5
                else:
                    repeat_count = 1
            else:
                repeat_count = int(repeat_count)

            print(f"[{lang_name}] Performing {action} ({direction}, {repeat_count}x swipe) on screen...")

            # 4. Cross-platform execution (iOS drag / standard W3C swipe)
            for _ in range(repeat_count):
                try:
                    self.driver.execute_script("mobile: dragFromToForDuration", {
                        "duration": 0.4,
                        "fromX": start_x,
                        "fromY": start_y,
                        "toX": end_x,
                        "toY": end_y
                    })
                except Exception:
                    try:
                        self.driver.swipe(start_x, start_y, end_x, end_y, 400)
                    except Exception as swipe_err:
                        print(f"⚠️ Scroll gesture error: {swipe_err}")
                time.sleep(0.4)

            time.sleep(1.5)
            return True

        # 4. Query VLM Agent
        decision = self.agent.ask_ai_for_element_id(
            annotated_img_path=annotated_img_path,
            instruction=instruction,
            elements=elements,
            language=lang_name
        )

        target_id = decision.get("target_id")
        confidence = decision.get("confidence", 1.0)
        rationale = str(decision.get("rationale", "")).lower()

        # -------------------------------------------------------------
        # Assertion Gate 1: Confidence & Element Existence Circuit Breaker
        # -------------------------------------------------------------
        CONFIDENCE_THRESHOLD = step_info.get("min_confidence", 0.35)
        NOT_FOUND_KEYWORDS = (
            "not found", "no matching", "cannot find", "not visible", 
            "does not exist", "none of the", "not present", "none", "null"
        )
        is_low_confidence = (confidence is not None and confidence < CONFIDENCE_THRESHOLD)
        is_semantic_not_found = (target_id is None) or any(k in rationale for k in NOT_FOUND_KEYWORDS)

        if is_low_confidence or is_semantic_not_found:
            if step_info.get("optional", False):
                print(f"ℹ️ [Optional Step] '{step_name}' ({instruction}) skipped - confidence {confidence:.2f} below threshold or element not found.")
                return True

            reason_desc = f"Model confidence too low ({confidence:.2f} < {CONFIDENCE_THRESHOLD})" if is_low_confidence else "Model determined target element is not present on screen"
            print(f"🛑 [Circuit Breaker] Step '{step_name}' failed: {reason_desc}!")
            decision["status"] = "step_failed_low_confidence" if is_low_confidence else "step_failed_target_not_found"
            decision["circuit_broken"] = True

            self.agent.log_interaction(
                language=lang_name,
                step_name=step_name,
                instruction=instruction,
                action=action,
                target_id=-1,
                center_x=None,
                center_y=None,
                screenshot_path=screenshot_path,
                annotated_image_path=annotated_img_path,
                decision=decision
            )
            return False

        used_fallback = False
        if "center_x" in decision and "center_y" in decision and decision["center_x"] is not None and decision["center_y"] is not None:
            center_x, center_y = decision["center_x"], decision["center_y"]
        elif str(target_id) in elements_map:
            center_x, center_y = elements_map[str(target_id)]
        else:
            # Dynamic heuristic fallback based on spatial coordinates & instruction
            target_id = self.agent.heuristic_fallback(instruction, elements)
            used_fallback = True
            if target_id is not None and str(target_id) in elements_map:
                print(f"⚠️ VLM ID not in element map. Heuristic coordinate fallback: target_id={target_id}")
                center_x, center_y = elements_map[str(target_id)]
            elif "center_x" in decision and "center_y" in decision and decision["center_x"] is not None:
                center_x, center_y = decision["center_x"], decision["center_y"]
            else:
                print(f"❌ Heuristic fallback also failed to resolve target ID for step: '{step_name}' ({instruction})")
                decision["status"] = "step_failed_unresolved_id"
                decision["circuit_broken"] = True
                self.agent.log_interaction(
                    language=lang_name,
                    step_name=step_name,
                    instruction=instruction,
                    action=action,
                    target_id=-1,
                    center_x=None,
                    center_y=None,
                    screenshot_path=screenshot_path,
                    annotated_image_path=annotated_img_path,
                    decision=decision
                )
                return False

        # 5. Perform physical-to-logical scaling & tap
        window_size = self.get_window_size()
        logical_w = window_size.get("width", 390)
        img = cv2.imread(str(screenshot_path))
        physical_w = img.shape[1] if img is not None else 1170
        scale_factor = physical_w / logical_w

        logical_x = center_x / scale_factor
        logical_y = center_y / scale_factor

        print(f"[{lang_name}] Executing {action} on {step_name} (id={target_id}) at physical ({center_x}, {center_y}) -> logical ({logical_x:.2f}, {logical_y:.2f})")

        # 6. Execute tap
        is_click_or_input = ("click" in action.lower()) or ("tap" in action.lower()) or (action == "type_text")
        if is_click_or_input:
            try:
                self.driver.tap([(logical_x, logical_y)])
            except Exception as tap_err:
                print(f"❌ Failed to tap target element at logical ({logical_x:.2f}, {logical_y:.2f}): {tap_err}")
                if "session" in str(tap_err).lower() or "terminated" in str(tap_err).lower():
                    print("🔄 Session dropped. Attempting to re-initialize Appium driver session...")
                    try:
                        self.init_driver()
                        self.driver.tap([(logical_x, logical_y)])
                        print("✅ Re-initialized session and successfully performed tap!")
                    except Exception as re_err:
                        print(f"❌ Re-initialize session failed: {re_err}")

        # -------------------------------------------------------------
        # Assertion Gate 2: Post-Click UI Change Assertion & Stagnation Circuit Breaker
        # -------------------------------------------------------------
        wait_after = step_info.get("wait_after", 1.5)
        time.sleep(wait_after)
        post_screenshot_path = folder_dir / f"{step_name}_{ts}_post.png"
        try:
            self.driver.save_screenshot(str(post_screenshot_path))
        except Exception:
            pass

        skip_layout_check = (
            is_setup_step
            or (action == "type_text")
            or ("click" not in action.lower() and "tap" not in action.lower())
            or step_info.get("skip_layout_check", False)
        )
        if not skip_layout_check and not self.agent.is_page_changed(screenshot_path, post_screenshot_path, threshold=0.005):
            print(f"⚠️ [Assertion Warning] Click on '{step_name}' (id={target_id}) triggered no UI response or transition. Attempting self-healing retry...")
            fb_id = self.agent.heuristic_fallback(instruction, elements)
            healed = False
            if fb_id is not None and fb_id != target_id and str(fb_id) in elements_map:
                fb_x, fb_y = elements_map[str(fb_id)]
                fb_lx, fb_ly = fb_x / scale_factor, fb_y / scale_factor
                print(f"🔄 [Self-Healing] Retrying click on candidate coordinate (id={fb_id}) at logical ({fb_lx:.2f}, {fb_ly:.2f})...")
                try:
                    self.driver.tap([(fb_lx, fb_ly)])
                    time.sleep(1.5)
                    self.driver.save_screenshot(str(post_screenshot_path))
                    if self.agent.is_page_changed(screenshot_path, post_screenshot_path, threshold=0.005):
                        healed = True
                        decision["self_healed"] = True
                        decision["healed_target_id"] = fb_id
                        target_id = fb_id
                        logical_x, logical_y = fb_lx, fb_ly
                        center_x, center_y = fb_x, fb_y
                        print("✅ [Self-Healing] Candidate coordinate retry succeeded, UI responded!")
                except Exception as tap_err:
                    print(f"⚠️ Self-healing retry failed: {tap_err}")

            if not healed:
                print(f"🛑 [Circuit Breaker] Click on '{step_name}' produced no UI change consecutively (possible tap on dead zone or stuck state). Step marked as failed!")
                decision["status"] = "step_failed_no_page_change"
                decision["circuit_broken"] = True
                self.agent.log_interaction(
                    language=lang_name,
                    step_name=step_name,
                    instruction=instruction,
                    action=action,
                    target_id=target_id,
                    center_x=center_x,
                    center_y=center_y,
                    screenshot_path=screenshot_path,
                    annotated_image_path=annotated_img_path,
                    decision=decision,
                    tap_x=logical_x,
                    tap_y=logical_y
                )
                return False


        # 7. Perform typing if action is type_text
        # 7. Perform typing if action is type_text (generalized for React Native, Flutter, native iOS & Android)
        if action == "type_text" and text_to_type:
            time.sleep(0.4)
            typed_success = False
            should_clear = step_info.get("clear", True)
            is_password_step = step_info.get("is_password", False) or any(
                k in step_name.lower() or k in instruction.lower() for k in ("password", "passwort", "pwd", "pin", "secret")
            )

            # Pre-sync to clipboard for long tokens/text so clipboard is ready
            try:
                self.driver.set_clipboard_text(text_to_type)
            except Exception:
                pass

            try:
                active_elem = self.driver.switch_to.active_element
                if active_elem:
                    if should_clear:
                        try:
                            active_elem.clear()
                            time.sleep(0.2)
                        except Exception:
                            pass

                    # ⚡ Fast Batch Typing: Send full string in 1 single W3C command (0.1s instead of 50s!)
                    try:
                        if len(text_to_type) > 40 and hasattr(active_elem, "set_value"):
                            try:
                                active_elem.set_value(text_to_type)
                                typed_success = True
                            except Exception:
                                active_elem.send_keys(text_to_type)
                                typed_success = True
                        else:
                            active_elem.send_keys(text_to_type)
                            typed_success = True

                        display_txt = (text_to_type[:15] + "..." + text_to_type[-10:]) if len(text_to_type) > 30 else text_to_type
                        print(f"[{lang_name}] ⚡ Fast-typed text into active element in 1 call: {display_txt}")
                    except Exception as type_err:
                        print(f"⚠️ Batch send_keys failed ({type_err}), attempting chunked input...")

                    # Verification: Check if text field matches expected text (guard against dropped characters)
                    try:
                        time.sleep(0.2)
                        val = active_elem.get_attribute("value") or active_elem.text or ""
                        # Handle password bullet masking (e.g. '••••••••••' or '**********') across platforms
                        is_bullet_masked = (
                            is_password_step
                            or (bool(val) and (set(val) <= {'•', '\u2022', '●', '*', '·', '∗'}) and len(val) == len(text_to_type))
                        )
                        if val and val != text_to_type and not is_bullet_masked:
                            display_val = (val[:15] + "...") if len(val) > 20 else val
                            display_exp = (text_to_type[:15] + "...") if len(text_to_type) > 20 else text_to_type
                            print(f"⚠️ [Typing Mismatch] Got '{display_val}', expected '{display_exp}'. Retrying with chunked send_keys...")
                            active_elem.clear()
                            time.sleep(0.2)
                            # Chunked typing: send in chunks of 16 characters
                            chunk_size = 16
                            for i in range(0, len(text_to_type), chunk_size):
                                active_elem.send_keys(text_to_type[i:i+chunk_size])
                                time.sleep(0.04)
                            print(f"[{lang_name}] Re-typed text into active element")
                    except Exception:
                        pass
            except Exception as elem_err:
                print(f"⚠️ Active element typing attempt failed: {elem_err}")

            # Approach B: ActionChains fallback for webviews or non-standard inputs
            if not typed_success:
                try:
                    from selenium.webdriver.common.action_chains import ActionChains
                    ActionChains(self.driver).send_keys(text_to_type).perform()
                    typed_success = True
                    print(f"[{lang_name}] Typed text via ActionChains fallback: {text_to_type}")
                except Exception as type_err:
                    print(f"⚠️ Type text warning: {type_err}")

            time.sleep(0.3)

            # Optional hide keyboard if specified in step_info
            if step_info.get("hide_keyboard", False):
                try:
                    self.driver.hide_keyboard()
                except Exception:
                    pass

            # Update post screenshot after typing so it captures the actual typed input
            try:
                self.driver.save_screenshot(str(post_screenshot_path))
            except Exception:
                pass


        # 8. Log interaction
        self.agent.log_interaction(
            language=lang_name,
            step_name=step_name,
            instruction=instruction,
            action=action,
            target_id=target_id,
            center_x=center_x,
            center_y=center_y,
            screenshot_path=screenshot_path,
            annotated_image_path=annotated_img_path,
            decision=decision,
            tap_x=logical_x,
            tap_y=logical_y
        )
        return True

    def run(self):
        """Runs the entire multi-loop, multi-language test flow engine."""
        start_time = time.time()
        self.init_driver()

        try:
            test_flows = self.config.get("test_flows", [])
            initial_setup = self.config.get("initial_setup_steps", [])

            # 1. Execute initial setup steps only once at the beginning (first run only)
            if initial_setup:
                print("\n================ 🚀 Initial Setup Steps (First Run Only) ================")
                self.reset_app()
                setup_lang = {
                    "folder": test_flows[0].get("folder", "default") if test_flows else "default",
                    "language_name": test_flows[0].get("language_name", "English") if test_flows else "English"
                }
                for s_i, setup_step in enumerate(initial_setup, 1):
                    ok = self.execute_step(setup_step, setup_lang, is_setup_step=True, step_index=s_i, loop_index=0)
                    if not ok:
                        print(f"🛑 [Setup Failed] Setup step '{setup_step['step_name']}' failed and triggered circuit break! Cannot proceed with test flows, aborting.")
                        return
            elif not test_flows:
                self.reset_app()

            # 2. Execute test flows for the specified number of loops (loop only applies to test_flows)
            if test_flows:
                for loop in range(self.loops):
                    print(f"\n================ 🌀 Generic GUI Agent Engine - Test Flow Loop {loop + 1}/{self.loops} ================")
                    
                    # Lightweight reset between loops to clean UI state, dismiss modals/keyboards, and restore root screen
                    if self.reset_between_loops and (loop > 0 or not initial_setup):
                        bundle_id = self.app_cfg.get("bundle_id", "")
                        print(f"🔄 [Lightweight Reset] Restarting app ({bundle_id}) for clean state before Loop {loop + 1}/{self.loops}...")
                        self.reset_app(wait_after=3)

                    for flow in test_flows:
                        lang_info = {
                            "folder": flow.get("folder", "default"),
                            "language_name": flow.get("language_name", "English")
                        }
                        print(f"\n🚀 Running test flow for language: {lang_info['language_name']} (Loop {loop + 1}/{self.loops})")
                        flow_broken = False
                        for s_i, step in enumerate(flow.get("steps", []), 1):
                            ok = self.execute_step(step, lang_info, is_setup_step=False, step_index=s_i, loop_index=loop)
                            if not ok:
                                print(f"🛑 [Flow Broken] Step '{step['step_name']}' triggered circuit breaker! Aborting current flow to prevent invalid API calls.")
                                flow_broken = True
                                break

                        if flow_broken:
                            # Trigger lightweight reset to clear stuck state before starting next flow or loop
                            self.reset_app(wait_after=3)
            elif not initial_setup:
                print("⚠️ No initial_setup_steps or test_flows defined in configuration.")

        finally:
            if self.driver:
                try:
                    self.driver.quit()
                except Exception:
                    pass
            elapsed = time.time() - start_time
            print(f"\n🎉 Test Engine Completed! Total Runtime: {int(elapsed//60)}m {int(elapsed%60)}s")


def main():
    parser = argparse.ArgumentParser(description="Generic Configuration-Driven GUI Test Agent Engine")
    parser.add_argument("--config", "-c", default=None, help="Path or name of test configuration JSON (e.g. 'bluesky', 'katapp', or 'configs/app.json')")
    parser.add_argument("--model-dir", "-m", default=".", help="Directory containing visual_agent.py (e.g. Qwen_trained, claude, openAI_5.6_Luna)")
    parser.add_argument("--agent-file", "-a", default=None, help="Agent script filename (e.g. visual_agent.py or visual_agent_DG.py)")
    parser.add_argument("--loops", "-l", type=int, default=int(os.environ.get("FLYWHEEL_LOOPS", 1)), help="Number of flywheel test loops")
    parser.add_argument("--no-loop-reset", action="store_true", help="Disable lightweight app reset between test loops")
    args = parser.parse_args()

    engine = GenericGUITestEngine(
        config_path=args.config,
        model_dir=args.model_dir,
        loops=args.loops,
        agent_filename=args.agent_file,
        reset_between_loops=not args.no_loop_reset
    )
    engine.run()


if __name__ == "__main__":
    main()
