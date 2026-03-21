#!/bin/bash
# EUCA ARENA — MOBA Demo
# Usage: ./scripts/moba.sh  (editor must be running)
set -e
E=target/debug/euca

# Wait for editor (poll, no fixed sleep)
while ! $E status >/dev/null 2>&1; do sleep 0.3; done

# Setup
$E game create --mode deathmatch --score-limit 100 2>/dev/null
$E entity create "--position=-7,0.5,0" --spawn-point 1 2>/dev/null
$E entity create --position=7,0.5,0 --spawn-point 2 2>/dev/null
$E entity create --mesh cube "--position=-8,0.5,0" --scale 1.5,1.5,1.5 --color blue --health 2000 --team 1 --physics Static --role structure 2>/dev/null
$E entity create --mesh cube --position=8,0.5,0 --scale 1.5,1.5,1.5 --color red --health 2000 --team 2 --physics Static --role structure 2>/dev/null
$E entity create --mesh cube "--position=-4,1,0" --scale 0.4,2.5,0.4 --color cyan --health 800 --team 1 --physics Static --combat --combat-damage 40 --combat-range 5 --combat-cooldown 1.5 --combat-style stationary --role tower --gold-bounty 150 --xp-bounty 100 2>/dev/null
$E entity create --mesh cube --position=4,1,0 --scale 0.4,2.5,0.4 --color orange --health 800 --team 2 --physics Static --combat --combat-damage 40 --combat-range 5 --combat-cooldown 1.5 --combat-style stationary --role tower --gold-bounty 150 --xp-bounty 100 2>/dev/null
$E entity create --mesh sphere "--position=-7,0.5,0" --scale 1.2,1.2,1.2 --color cyan --health 500 --team 1 --physics Kinematic --collider sphere:0.6 --combat --combat-damage 30 --combat-range 2 --combat-speed 4 --combat-cooldown 0.8 --role hero --gold 0 --gold-bounty 300 --xp-bounty 200 2>/dev/null
$E entity create --mesh sphere --position=7,0.5,0 --scale 1.2,1.2,1.2 --color orange --health 500 --team 2 --physics Kinematic --collider sphere:0.6 --combat --combat-damage 30 --combat-range 2 --combat-speed 4 --combat-cooldown 0.8 --role hero --gold 0 --gold-bounty 300 --xp-bounty 200 2>/dev/null
$E rule create --when timer:20 --do-action "spawn cube -7,0.5,0 blue 80 1 true -7,0,0:0,0,0:7,0,0 3 0.5,0.5,0.5 20 30 minion 3" 2>/dev/null
$E rule create --when timer:20 --do-action "spawn cube 7,0.5,0 red 80 2 true 7,0,0:0,0,0:-7,0,0 3 0.5,0.5,0.5 20 30 minion 3" 2>/dev/null
$E rule create --when death --filter team:2 --do-action "score source +1" 2>/dev/null
$E rule create --when death --filter team:1 --do-action "score source +1" 2>/dev/null
$E ui text "EUCA ARENA" --x 0.5 --y 0.02 --size 28 --color yellow 2>/dev/null
$E camera set --eye 0,6,6 --target 0,0.5,0 2>/dev/null

echo "EUCA ARENA ready — starting"
$E sim play
