#!/bin/bash
cd "$(dirname "$0")"
./target/debug/examples/dota_client 2>&1 | tee /tmp/dota_debug.log
