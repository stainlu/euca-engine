#!/bin/bash
# EUCA ARENA v2 — Full MOBA with Economy + Levels
# Run with: ./scripts/moba.sh (editor must be running)
set -e
E=target/debug/euca

echo "╔══════════════════════════════════╗"
echo "║      EUCA ARENA — MOBA v2       ║"
echo "╚══════════════════════════════════╝"

$E status

# === MATCH ===
$E game create --mode deathmatch --score-limit 1

# === BASES (high HP, no combat, structure role) ===
$E entity create --mesh cube "--position=-8,1,0" --scale 1.5,1.5,1.5 --color blue --health 2000 --team 1 --physics Static --collider aabb:0.75,0.75,0.75 --role structure
$E entity create --mesh cube --position=8,1,0 --scale 1.5,1.5,1.5 --color red --health 2000 --team 2 --physics Static --collider aabb:0.75,0.75,0.75 --role structure

# === TOWERS (stationary combat, gold/xp bounty) ===
$E entity create --mesh cube "--position=-5,1.5,0" --scale 0.4,2.5,0.4 --color cyan --health 800 --team 1 --physics Static --collider aabb:0.2,1.25,0.2 --combat --combat-damage 40 --combat-range 5 --combat-cooldown 1.5 --combat-style stationary --role tower --gold-bounty 150 --xp-bounty 100
$E entity create --mesh cube "--position=-3,1.5,0" --scale 0.4,2.5,0.4 --color cyan --health 800 --team 1 --physics Static --collider aabb:0.2,1.25,0.2 --combat --combat-damage 40 --combat-range 5 --combat-cooldown 1.5 --combat-style stationary --role tower --gold-bounty 150 --xp-bounty 100
$E entity create --mesh cube --position=5,1.5,0 --scale 0.4,2.5,0.4 --color orange --health 800 --team 2 --physics Static --collider aabb:0.2,1.25,0.2 --combat --combat-damage 40 --combat-range 5 --combat-cooldown 1.5 --combat-style stationary --role tower --gold-bounty 150 --xp-bounty 100
$E entity create --mesh cube --position=3,1.5,0 --scale 0.4,2.5,0.4 --color orange --health 800 --team 2 --physics Static --collider aabb:0.2,1.25,0.2 --combat --combat-damage 40 --combat-range 5 --combat-cooldown 1.5 --combat-style stationary --role tower --gold-bounty 150 --xp-bounty 100

# === HEROES (gold + level + bounty) ===
$E entity create --mesh sphere "--position=-7,1,0" --scale 1.2,1.2,1.2 --color cyan --health 500 --team 1 --physics Kinematic --collider sphere:0.6 --combat --combat-damage 30 --combat-range 2 --combat-speed 4 --combat-cooldown 0.8 --role hero --gold 0 --gold-bounty 300 --xp-bounty 200
$E entity create --mesh sphere --position=7,1,0 --scale 1.2,1.2,1.2 --color orange --health 500 --team 2 --physics Kinematic --collider sphere:0.6 --combat --combat-damage 30 --combat-range 2 --combat-speed 4 --combat-cooldown 0.8 --role hero --gold 0 --gold-bounty 300 --xp-bounty 200

# === SPAWN POINTS (for hero respawn) ===
# These are invisible entities with SpawnPoint component — not yet via CLI
# Heroes will respawn at their initial positions via the respawn system

# === MINION WAVE RULES (every 15s, with gold/xp bounty + role) ===
# Format: spawn mesh pos color health team combat waypoints speed scale gold_bounty xp_bounty role
# Blue minions (team 1): patrol toward red base
$E rule create --when timer:15 --do-action "spawn cube -7,1,0 blue 80 1 true -7,0,0:0,0,0:7,0,0 3 0.5,0.5,0.5 20 30 minion"
$E rule create --when timer:15 --do-action "spawn cube -7,1,1 blue 80 1 true -7,0,1:0,0,1:7,0,1 3 0.5,0.5,0.5 20 30 minion"
# Red minions (team 2): patrol toward blue base
$E rule create --when timer:15 --do-action "spawn cube 7,1,0 red 80 2 true 7,0,0:0,0,0:-7,0,0 3 0.5,0.5,0.5 20 30 minion"
$E rule create --when timer:15 --do-action "spawn cube 7,1,1 red 80 2 true 7,0,1:0,0,1:-7,0,1 3 0.5,0.5,0.5 20 30 minion"

# === SCORING ===
$E rule create --when death --filter team:2 --do-action "score source +1"
$E rule create --when death --filter team:1 --do-action "score source +1"

# === HUD ===
$E ui text "EUCA ARENA" --x 0.5 --y 0.02 --size 28 --color yellow

# === CAMERA ===
$E camera set --eye 0,12,10 --target 0,1,0

echo ""
echo "=== SETUP COMPLETE ==="
$E status
echo ""

# === RUN SIMULATION ===
echo ">>> Starting simulation..."
$E sim play
sleep 10
$E sim pause

echo ""
echo "=== ROUND 1 (10s) ==="
echo "Blue Hero (8):"
$E ability list 8
echo "Red Hero (9):"
$E ability list 9
$E screenshot

# Play more
$E sim play
sleep 20
$E sim pause

echo ""
echo "=== ROUND 2 (30s) ==="
echo "Blue Hero (8):"
$E ability list 8
echo "Red Hero (9):"
$E ability list 9
echo "Tower health:"
$E entity get 4 2>/dev/null | grep -A1 "health"
$E entity get 7 2>/dev/null | grep -A1 "health"
$E game state
$E status
$E screenshot

# Play final round
$E sim play
sleep 30
$E sim pause

echo ""
echo "=== ROUND 3 (60s) ==="
echo "Blue Hero (8):"
$E ability list 8
echo "Red Hero (9):"
$E ability list 9
$E game state
$E status
$E screenshot

echo ""
echo "=== MOBA TEST COMPLETE ==="
