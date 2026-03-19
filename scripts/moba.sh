#!/bin/bash
# EUCA ARENA — 1-Lane MOBA Demo
# Run with: ./scripts/moba.sh (editor must be running)
set -e

EUCA="cargo run -p euca-cli --"

echo "=== EUCA ARENA — Setting up MOBA ==="

# Verify engine is running
$EUCA status

# Create match
$EUCA game create --mode deathmatch --score-limit 1

# ── BASES ──
echo "=== Creating bases ==="
# Blue base (entity ~2)
$EUCA entity create --mesh cube --position=-25,1,0 --scale 2.5,2.5,2.5 --color blue --health 2000 --team 1 --physics Static --collider aabb:1.25,1.25,1.25
# Red base (entity ~3)
$EUCA entity create --mesh cube --position=25,1,0 --scale 2.5,2.5,2.5 --color red --health 2000 --team 2 --physics Static --collider aabb:1.25,1.25,1.25

# ── TOWERS ──
echo "=== Creating towers ==="
# Blue T1 (outer)
$EUCA entity create --mesh cube --position=-15,1.5,0 --scale 0.5,3,0.5 --color cyan --health 800 --team 1 --physics Static --collider aabb:0.25,1.5,0.25 --combat --combat-damage 40 --combat-range 8 --combat-cooldown 1.5 --combat-style stationary
# Blue T2 (inner)
$EUCA entity create --mesh cube --position=-8,1.5,0 --scale 0.5,3,0.5 --color cyan --health 800 --team 1 --physics Static --collider aabb:0.25,1.5,0.25 --combat --combat-damage 40 --combat-range 8 --combat-cooldown 1.5 --combat-style stationary
# Red T1 (outer)
$EUCA entity create --mesh cube --position=15,1.5,0 --scale 0.5,3,0.5 --color orange --health 800 --team 2 --physics Static --collider aabb:0.25,1.5,0.25 --combat --combat-damage 40 --combat-range 8 --combat-cooldown 1.5 --combat-style stationary
# Red T2 (inner)
$EUCA entity create --mesh cube --position=8,1.5,0 --scale 0.5,3,0.5 --color orange --health 800 --team 2 --physics Static --collider aabb:0.25,1.5,0.25 --combat --combat-damage 40 --combat-range 8 --combat-cooldown 1.5 --combat-style stationary

# ── HEROES ──
echo "=== Creating heroes ==="
# Blue hero
$EUCA entity create --mesh sphere --position=-20,1,0 --scale 1.3,1.3,1.3 --color cyan --health 500 --team 1 --physics Dynamic --collider sphere:0.65 --combat --combat-damage 30 --combat-range 2 --combat-speed 5 --combat-cooldown 0.8
# Red hero
$EUCA entity create --mesh sphere --position=20,1,0 --scale 1.3,1.3,1.3 --color orange --health 500 --team 2 --physics Dynamic --collider sphere:0.65 --combat --combat-damage 30 --combat-range 2 --combat-speed 5 --combat-cooldown 0.8

# ── MINION WAVE RULES ──
echo "=== Creating minion wave rules ==="
# Blue melee minions (3 per wave, patrol from blue base → red base)
$EUCA rule create --when timer:30 --do-action "spawn cube -22,1,0 blue 80 1 true -22,0,0:0,0,0:22,0,0 3"
$EUCA rule create --when timer:30 --do-action "spawn cube -22,1,1.5 blue 80 1 true -22,0,1.5:0,0,1.5:22,0,1.5 3"
$EUCA rule create --when timer:30 --do-action "spawn cube -22,1,-1.5 blue 80 1 true -22,0,-1.5:0,0,-1.5:22,0,-1.5 3"

# Red melee minions (3 per wave, patrol from red base → blue base)
$EUCA rule create --when timer:30 --do-action "spawn cube 22,1,0 red 80 2 true 22,0,0:0,0,0:-22,0,0 3"
$EUCA rule create --when timer:30 --do-action "spawn cube 22,1,1.5 red 80 2 true 22,0,1.5:0,0,1.5:-22,0,1.5 3"
$EUCA rule create --when timer:30 --do-action "spawn cube 22,1,-1.5 red 80 2 true 22,0,-1.5:0,0,-1.5:-22,0,-1.5 3"

# ── SCORING ──
$EUCA rule create --when death --filter team:2 --do-action "score source +1"
$EUCA rule create --when death --filter team:1 --do-action "score source +1"

# ── HUD ──
echo "=== Setting up HUD ==="
$EUCA ui text "EUCA ARENA" --x 0.5 --y 0.01 --size 32 --color yellow
$EUCA ui bar --x 0.02 --y 0.95 --width 0.2 --height 0.03 --fill 1.0 --color blue
$EUCA ui bar --x 0.78 --y 0.95 --width 0.2 --height 0.03 --fill 1.0 --color red

# ── CAMERA ──
$EUCA camera set --eye 0,30,25 --target 0,0,0

# ── SCREENSHOT SETUP ──
echo "=== Taking setup screenshot ==="
$EUCA screenshot --output /tmp/moba_setup.png

echo ""
echo "=== MOBA setup complete! ==="
echo "  Bases: 2 (blue @ -25, red @ 25)"
echo "  Towers: 4 (2 per team)"
echo "  Heroes: 2 (blue @ -20, red @ 20)"
echo "  Minion waves: every 30s, 3 per team"
echo ""
echo "Starting simulation..."
$EUCA sim play

echo "=== Waiting 35s for first minion wave ==="
sleep 35
$EUCA sim pause
$EUCA screenshot --output /tmp/moba_wave1.png
echo "Wave 1 spawned. Resuming..."

$EUCA sim play
echo "=== Waiting 30s for combat ==="
sleep 30
$EUCA sim pause
$EUCA screenshot --output /tmp/moba_combat.png
$EUCA game state
echo "=== Checking entity health ==="
$EUCA entity list

$EUCA sim play
echo "=== Waiting 60s more ==="
sleep 60
$EUCA sim pause
$EUCA screenshot --output /tmp/moba_late.png
$EUCA game state
$EUCA entity list

echo "=== MOBA test complete ==="
echo "Screenshots saved to /tmp/moba_*.png"
