#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "======================================================="
echo "🚀 [CI/CD Pipeline] Starting Multilingual Visual Regression Test"
echo "======================================================="

# 1. Initialize output directories
mkdir -p diff_results screenshots

# 2. Check and start Appium server (Port 4723)
if ! nc -z localhost 4723; then
    echo "🔌 [Step 1/5] Launching Appium Server in background..."
    appium --port 4723 > diff_results/appium.log 2>&1 &
    APPIUM_PID=$!
    sleep 4
    echo "✅ Appium started (PID: $APPIUM_PID)"
else
    echo "✅ [Step 1/5] Appium is already running on port 4723"
fi

# 3. Boot iOS Simulator (macOS only)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "📱 [Step 2/5] Checking and booting iOS Simulator..."
    # Prefer already booted simulator
    UDID=$(xcrun simctl list devices 2>/dev/null | grep "(Booted)" | head -1 | awk -F '[()]' '{print $2}')
    if [ -z "$UDID" ]; then
        UDID=$(xcrun simctl list devices available 2>/dev/null | grep -E "iPhone 18 Pro|iPhone 16 Pro|iPhone 16" | head -1 | awk -F '[()]' '{print $2}')
    fi
    if [ -z "$UDID" ]; then
        UDID=$(xcrun simctl list devices available 2>/dev/null | grep "iPhone 15" | head -1 | awk -F '[()]' '{print $2}')
    fi
    if [ -z "$UDID" ]; then
        UDID=$(xcrun simctl list devices available 2>/dev/null | grep "iPhone" | head -1 | awk -F '[()]' '{print $2}')
    fi

    if [ -n "$UDID" ]; then
        echo "Booting simulator: $UDID"
        xcrun simctl boot "$UDID" 2>/dev/null || true
        open -a Simulator 2>/dev/null || true
        echo "⏳ Waiting for simulator to finish booting..."
        xcrun simctl bootstatus "$UDID" -b
        echo "✅ Simulator is fully booted and ready."
        export AUTOSCREEN_UDID="$UDID"
    else
        echo "⚠️ No available iPhone simulator found, skipping auto-boot"
    fi
else
    echo "ℹ️ [Step 2/5] Non-macOS environment, skipping local simulator boot"
fi

# 4. Execute visual agent test flows and capture screenshots
echo "🤖 [Step 3/5] Running visual agent navigation flows..."
export AGENT_MODE="${AGENT_MODE:-dg}"
export LOCAL_MODEL_PATH="${LOCAL_MODEL_PATH:-$PROJECT_ROOT/Qwen_trained/Qwen3-VL-4B-V2}"
APP_NAME="${APP_NAME:-katapp}"
MODEL_NAME="${MODEL_NAME:-Qwen_trained}"

if [ -d "$PROJECT_ROOT/build/ios/iphonesimulator/Runner.app" ]; then
    export AUTOSCREEN_APP_PATH="$PROJECT_ROOT/build/ios/iphonesimulator/Runner.app"
    echo "📦 Using target application bundle: $AUTOSCREEN_APP_PATH"
fi

if [ -f "screenshot.py" ]; then
    python3 screenshot.py --mode "$AGENT_MODE" --loops 1
else
    # Use unified project CLI entrypoint
    python3 auto_screen.py test --app "$APP_NAME" --model "$MODEL_NAME" --mode "$AGENT_MODE" --loops 1
fi

# 5. Cross-language visual comparison and report generation
echo "🔍 [Step 4/5] Running pixel & OpenCV visual regression diff..."
# Report 1: language comparison (reference language against the other languages, same build).
# The CI quality gate reads this report.
# --model points explicitly to <model>/screenshots instead of relying on the fallback from ./screenshots.
python3 compare_images.py --batch --app "$APP_NAME" --model "$MODEL_NAME"

# Optional (not evaluated): BASELINE_MODE=approved compares with the baseline approved by
# tools/approve_baseline.py (baselines/<app>/ or APPROVED_BASELINE_DIR) instead of the previous build.
# An explicitly set PREVIOUS_SCREENSHOTS (e.g. by CI) takes precedence.
if [ "${BASELINE_MODE:-previous}" = "approved" ] && [ -z "${PREVIOUS_SCREENSHOTS:-}" ]; then
    APPROVED_DIR="${APPROVED_BASELINE_DIR:-$PROJECT_ROOT/baselines/$APP_NAME}"
    if [ -d "$APPROVED_DIR/screenshots" ]; then
        export PREVIOUS_SCREENSHOTS="$APPROVED_DIR/screenshots"
        PREVIOUS_APP_ID="${PREVIOUS_APP_ID:-approved-$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('label', ''))" "$APPROVED_DIR/baseline.json" 2>/dev/null)}"
        echo "✅ Using approved baseline $PREVIOUS_APP_ID from $APPROVED_DIR"
    else
        echo "⚠️ BASELINE_MODE=approved, but no approved baseline in $APPROVED_DIR. Version comparison skipped."
    fi
fi

# Report 2 (optional): version comparison against the screenshots of the previous release,
# each language folder against the folder of the same name, e.g.:
#   PREVIOUS_SCREENSHOTS=<prev_release>/screenshots ./run_pipeline_local.sh
if [ -n "${PREVIOUS_SCREENSHOTS:-}" ]; then
    echo "🔍 [Step 4b/5] Running version comparison against $PREVIOUS_SCREENSHOTS..."
    # Optional identities of the two app builds (set by CI), shown in the report
    LABEL_ARGS=()
    if [ -n "${PREVIOUS_APP_ID:-}" ]; then
        LABEL_ARGS+=(--baseline-label "$PREVIOUS_APP_ID")
    fi
    if [ -n "${CURRENT_APP_ID:-}" ]; then
        LABEL_ARGS+=(--actual-label "$CURRENT_APP_ID")
    fi
    # Optional (not evaluated): LOCAL_TRIAGE=1 rates the difference boxes with a local VLM
    if [ "${LOCAL_TRIAGE:-0}" = "1" ]; then
        LABEL_ARGS+=(--ai-local)
    fi
    python3 compare_images.py --batch --app "$APP_NAME" --model "$MODEL_NAME" \
        --version-baseline-root "$PREVIOUS_SCREENSHOTS" \
        --output-root diff_results/version_comparison \
        ${LABEL_ARGS[@]+"${LABEL_ARGS[@]}"}
fi

# 6. CI Quality Gate assertion
echo "⚖️ [Step 5/5] Checking pipeline quality gate..."
python3 assert_pipeline_result.py

echo "======================================================="
echo "🎉 Pipeline finished successfully! HTML report: diff_results/summary_report.html"
echo "======================================================="

# Automatically open HTML report on macOS (desktop environments only)
if [[ "$OSTYPE" == "darwin"* ]] && [ -z "$CI" ]; then
    open diff_results/summary_report.html 2>/dev/null || true
fi
