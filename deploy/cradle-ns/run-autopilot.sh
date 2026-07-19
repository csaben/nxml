#!/bin/sh
set -eu

token_file=${NXML_AUTOPILOT_TOKEN_FILE:-/home/arelius/.config/nxml/autopilot.token}
if [ ! -r "$token_file" ]; then
    echo "autopilot token is missing; start nxml-edge once to provision it" >&2
    exit 1
fi
AUTOPILOT_WEB_TOKEN=$(tr -d '\r\n' < "$token_file")
export AUTOPILOT_WEB_TOKEN

exec uv run nxml-autopilot \
    --game "$NXML_GAME" \
    --policy "$NXML_POLICY" \
    --controller "$NXML_ORCHESTRATOR_URL" \
    --camera "$NXML_CAMERA_ID" \
    --input-source web \
    --web-host "$NXML_WEB_HOST" \
    --web-port "$NXML_WEB_PORT" \
    --mode "$NXML_INITIAL_MODE" \
    --record "$NXML_CAPTURE_DIR"
