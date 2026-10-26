#!/usr/bin/env bash
# Installs a stored KatApp build at the path that run_pipeline_local.sh and the CI fallback use.
# Usage: ff3/switch_app.sh A | B5 | status
#   A  = Build A (fdb6f79, without seeded defects)
#   B5 = Build B (af44aec, seeded defects active in German)
set -euo pipefail

BUILDS="$HOME/Masterarbeit/katapp-builds"
APP="$HOME/Masterarbeit/katapp-app/katapp-app/build/ios/iphonesimulator/Runner.app"

status() {
    for b in A B5; do
        if cmp -s "$APP/Runner" "$BUILDS/$b/Runner"; then
            echo "Installed: Build $b"
            return
        fi
    done
    echo "Installed: unknown build (neither A nor B5)"
}

case "${1:-}" in
    A|B5)
        [ -f "$BUILDS/$1/Runner" ] || { echo "Build $1 not found in $BUILDS"; exit 1; }
        mkdir -p "$APP"
        rsync -a --delete "$BUILDS/$1/" "$APP/"
        status
        ;;
    status)
        status
        ;;
    *)
        echo "Usage: $0 A | B5 | status"
        exit 1
        ;;
esac
