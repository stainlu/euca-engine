#!/bin/bash
# EUCA ARENA — MOBA Demo v4
set -e
E=target/debug/euca

echo "╔══════════════════════════════════╗"
echo "║      EUCA ARENA — MOBA v4       ║"
echo "╚══════════════════════════════════╝"

$E status

$E game create --mode deathmatch --score-limit 100

# Spawn Points (invisible, mark respawn locations)
$E entity create "--position=-7,0.5,0" --spawn-point 1
$E entity create --position=7,0.5,0 --spawn-point 2

# Bases
$E entity create --mesh cube "--position=-8,0.5,0" --scale 1.5,1.5,1.5 --color blue --health 2000 --team 1 --physics Static --collider aabb:0.75,0.75,0.75 --role structure
$E entity create --mesh cube --position=8,0.5,0 --scale 1.5,1.5,1.5 --color red --health 2000 --team 2 --physics Static --collider aabb:0.75,0.75,0.75 --role structure

# Towers
$E entity create --mesh cube "--position=-4,1,0" --scale 0.4,2.5,0.4 --color cyan --health 800 --team 1 --physics Static --combat --combat-damage 40 --combat-range 5 --combat-cooldown 1.5 --combat-style stationary --role tower
$E entity create --mesh cube --position=4,1,0 --scale 0.4,2.5,0.4 --color orange --health 800 --team 2 --physics Static --combat --combat-damage 40 --combat-range 5 --combat-cooldown 1.5 --combat-style stationary --role tower

# Heroes (start at their bases)
$E entity create --mesh sphere "--position=-7,0.5,0" --scale 1.2,1.2,1.2 --color cyan --health 500 --team 1 --physics Kinematic --collider sphere:0.6 --combat --combat-damage 30 --combat-range 2 --combat-speed 4 --combat-cooldown 0.8 --role hero --gold 0 --gold-bounty 300 --xp-bounty 200
$E entity create --mesh sphere --position=7,0.5,0 --scale 1.2,1.2,1.2 --color orange --health 500 --team 2 --physics Kinematic --collider sphere:0.6 --combat --combat-damage 30 --combat-range 2 --combat-speed 4 --combat-cooldown 0.8 --role hero --gold 0 --gold-bounty 300 --xp-bounty 200

# Minion waves (every 8s)
$E rule create --when timer:8 --do-action "spawn cube -7,0.5,0 blue 80 1 true -7,0,0:0,0,0:7,0,0 3 0.5,0.5,0.5 20 30 minion"
$E rule create --when timer:8 --do-action "spawn cube 7,0.5,0 red 80 2 true 7,0,0:0,0,0:-7,0,0 3 0.5,0.5,0.5 20 30 minion"

# Scoring
$E rule create --when death --filter team:2 --do-action "score source +1"
$E rule create --when death --filter team:1 --do-action "score source +1"

# HUD + Camera
$E ui text "EUCA ARENA" --x 0.5 --y 0.02 --size 28 --color yellow
$E camera set --eye 0,6,6 --target 0,0.5,0

echo "=== SETUP DONE ==="
$E status
$E sim play
echo ">>> SIMULATION RUNNING <<<"
