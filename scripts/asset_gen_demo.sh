#!/bin/bash
# AI Asset Generation Demo
# Usage: ./scripts/asset_gen_demo.sh [port]
# Requires: TRIPO_API_KEY or MESHY_API_KEY set

PORT="${1:-3917}"
SERVER="http://127.0.0.1:${PORT}"
E="target/debug/euca -s ${SERVER}"

echo "=== AI Asset Generation Demo ==="

# Check available providers
echo "Checking providers..."
$E asset providers

# Generate a 3D model (uses first available provider)
echo ""
echo "Generating 3D model..."
PROVIDER="${AI_PROVIDER:-tripo}"  # override with AI_PROVIDER env var
$E asset generate --prompt "low poly medieval sword" --provider $PROVIDER --quality medium

echo ""
echo "Check assets/generated/ for the output GLB file."
echo "Use: euca entity create --mesh assets/generated/<file>.glb --position 0,1,0"
