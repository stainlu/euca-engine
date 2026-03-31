#!/bin/bash
# Serve the poker web client and start the poker server.
#
# Usage: ./serve.sh
#
# Opens:
#   - Poker server on ws://localhost:8080/ws
#   - Web client on http://localhost:8082/index.html

cd "$(dirname "$0")"

echo "Starting poker server on :8080..."
cargo run -p euca-poker-server &
SERVER_PID=$!

echo "Serving web client on :8082..."
python3 -m http.server 8082 &
WEB_PID=$!

echo ""
echo "  Poker server: http://localhost:8080"
echo "  Web client:   http://localhost:8082"
echo ""
echo "Press Ctrl+C to stop."

trap "kill $SERVER_PID $WEB_PID 2>/dev/null; exit" INT
wait
