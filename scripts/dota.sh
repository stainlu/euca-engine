#!/bin/bash
# EUCA DOTA — DotA-style MOBA setup
# Usage: ./scripts/dota.sh [port]   (headless_server must be running)
set -e

PORT="${1:-3917}"
SERVER="http://127.0.0.1:${PORT}"
E="target/debug/euca -s ${SERVER}"

# ── Wait for server ──
echo "Waiting for Euca server on port ${PORT}..."
while ! $E status >/dev/null 2>&1; do sleep 0.3; done
echo "Server ready."

# ── 1. Load the DotA level ──
echo "Loading DotA level..."
curl -s -X POST "${SERVER}/level/load" \
  -H 'Content-Type: application/json' \
  -d '{"path":"levels/dota.json"}'
echo ""

# ── 2. Define items ──
echo "Defining items..."

$E item define --id 1 --name "Iron Branch" --prop cost:50 --prop health:15 --prop damage:1 2>/dev/null
$E item define --id 2 --name "Healing Salve" --prop cost:100 --prop heal:400 2>/dev/null
$E item define --id 3 --name "Boots of Speed" --prop cost:500 --prop speed:2 2>/dev/null
$E item define --id 4 --name "Broadsword" --prop cost:1000 --prop damage:18 2>/dev/null
$E item define --id 5 --name "Platemail" --prop cost:1400 --prop armor:10 2>/dev/null
$E item define --id 6 --name "Power Treads" --prop cost:1400 --prop speed:3 --prop damage:10 2>/dev/null
$E item define --id 7 --name "Black King Bar" --prop cost:4050 --prop health:200 --prop damage:24 2>/dev/null
$E item define --id 8 --name "Daedalus" --prop cost:5150 --prop damage:88 --prop crit_chance:30 2>/dev/null

echo "  8 items defined."

# ── 3. Register hero definitions ──
echo "Registering heroes..."

# Juggernaut — melee carry with Blade Fury and Omnislash
curl -s -X POST "${SERVER}/hero/define" \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "Juggernaut",
    "health": 620,
    "mana": 290,
    "gold": 625,
    "damage": 52,
    "range": 1.5,
    "base_stats": {"max_health": 620, "attack_damage": 52, "armor": 2, "move_speed": 5},
    "stat_growth": {"max_health": 85, "attack_damage": 3.0, "armor": 0.3, "move_speed": 0},
    "abilities": [
      {
        "slot": "Q",
        "name": "Blade Fury",
        "cooldown": 12.0,
        "mana_cost": 110,
        "effect": {"AreaDamage": {"radius": 3.0, "damage": 120}}
      },
      {
        "slot": "W",
        "name": "Healing Ward",
        "cooldown": 30.0,
        "mana_cost": 120,
        "effect": {"Heal": {"amount": 200}}
      },
      {
        "slot": "R",
        "name": "Omnislash",
        "cooldown": 80.0,
        "mana_cost": 200,
        "effect": {"Chain": [
          {"Dash": {"distance": 5.0}},
          {"AreaDamage": {"radius": 2.0, "damage": 250}}
        ]}
      }
    ]
  }'
echo ""

# Crystal Maiden — ranged support with crowd-control and aura
curl -s -X POST "${SERVER}/hero/define" \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "Crystal Maiden",
    "health": 480,
    "mana": 400,
    "gold": 625,
    "damage": 35,
    "range": 6.0,
    "base_stats": {"max_health": 480, "attack_damage": 35, "armor": 1, "move_speed": 4},
    "stat_growth": {"max_health": 60, "attack_damage": 1.5, "armor": 0.1, "move_speed": 0},
    "abilities": [
      {
        "slot": "Q",
        "name": "Crystal Nova",
        "cooldown": 10.0,
        "mana_cost": 130,
        "effect": {"AreaDamage": {"radius": 5.0, "damage": 100}}
      },
      {
        "slot": "W",
        "name": "Frostbite",
        "cooldown": 9.0,
        "mana_cost": 115,
        "effect": {"Damage": {"amount": 150, "category": "magical"}}
      },
      {
        "slot": "R",
        "name": "Freezing Field",
        "cooldown": 90.0,
        "mana_cost": 300,
        "effect": {"AreaDamage": {"radius": 8.0, "damage": 400}}
      }
    ]
  }'
echo ""

# Sven — melee strength carry with stun and cleave
curl -s -X POST "${SERVER}/hero/define" \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "Sven",
    "health": 700,
    "mana": 250,
    "gold": 625,
    "damage": 63,
    "range": 1.5,
    "base_stats": {"max_health": 700, "attack_damage": 63, "armor": 3, "move_speed": 5},
    "stat_growth": {"max_health": 95, "attack_damage": 3.5, "armor": 0.4, "move_speed": 0},
    "abilities": [
      {
        "slot": "Q",
        "name": "Storm Hammer",
        "cooldown": 13.0,
        "mana_cost": 140,
        "effect": {"SpawnProjectile": {"speed": 12.0, "range": 8.0, "width": 0.5, "damage": 100, "category": "magical"}}
      },
      {
        "slot": "W",
        "name": "Warcry",
        "cooldown": 20.0,
        "mana_cost": 60,
        "effect": {"ApplyEffect": {"tag": "warcry", "modifiers": [["armor", "add", 10.0], ["move_speed", "add", 2.0]], "duration": 8.0}}
      },
      {
        "slot": "R",
        "name": "Gods Strength",
        "cooldown": 80.0,
        "mana_cost": 100,
        "effect": {"ApplyEffect": {"tag": "gods_strength", "modifiers": [["attack_damage", "multiply", 2.0]], "duration": 25.0}}
      }
    ]
  }'
echo ""

echo "  3 heroes registered."

# ── 3b. Apply hero templates — find heroes dynamically by role, never hardcode IDs ──
echo "Selecting heroes..."
OBSERVE=$(curl -s "${SERVER}/observe")

# Find hero entity IDs by role + team (entities with EntityRole::Hero)
HERO_T1=$(echo "$OBSERVE" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for e in data:
    c = e.get('components', {})
    if c.get('EntityRole') == 'Hero' and c.get('Team', {}).get('0', 0) == 1:
        print(e['id']); break
" 2>/dev/null)

HERO_T2=$(echo "$OBSERVE" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for e in data:
    c = e.get('components', {})
    if c.get('EntityRole') == 'Hero' and c.get('Team', {}).get('0', 0) == 2:
        print(e['id']); break
" 2>/dev/null)

if [ -n "$HERO_T1" ]; then
  curl -s -X POST "${SERVER}/hero/select" \
    -H 'Content-Type: application/json' \
    -d "{\"entity_id\": $HERO_T1, \"hero_name\": \"Juggernaut\"}"
  echo ""
fi
if [ -n "$HERO_T2" ]; then
  curl -s -X POST "${SERVER}/hero/select" \
    -H 'Content-Type: application/json' \
    -d "{\"entity_id\": $HERO_T2, \"hero_name\": \"Sven\"}"
  echo ""
fi
echo "  Heroes: Juggernaut (id=$HERO_T1) vs Sven (id=$HERO_T2)"

# ── 3c. Hero attribute definitions (STR/AGI/INT base + growth) ──
echo "Setting hero attributes..."
# Juggernaut: Agility hero
if [ -n "$HERO_T1" ]; then
  curl -s -X POST "${SERVER}/entity/${HERO_T1}/component" \
    -H 'Content-Type: application/json' \
    -d '{"HeroAttributes": {
      "primary": "Agility",
      "base": {"strength": 20.0, "agility": 34.0, "intelligence": 14.0},
      "growth": {"strength": 2.2, "agility": 2.8, "intelligence": 1.4},
      "timings": {"base_attack_time": 1.4, "attack_point": 0.33, "turn_rate": 0.6, "projectile_speed": 0}
    }}'
  echo ""
fi
# Sven: Strength hero
if [ -n "$HERO_T2" ]; then
  curl -s -X POST "${SERVER}/entity/${HERO_T2}/component" \
    -H 'Content-Type: application/json' \
    -d '{"HeroAttributes": {
      "primary": "Strength",
      "base": {"strength": 22.0, "agility": 21.0, "intelligence": 16.0},
      "growth": {"strength": 3.2, "agility": 2.0, "intelligence": 1.3},
      "timings": {"base_attack_time": 1.8, "attack_point": 0.4, "turn_rate": 0.6, "projectile_speed": 0}
    }}'
  echo ""
fi
echo "  Hero attributes set."

# ── 3d. Initialize match state (Roshan, day/night, wards) ──
echo "Initializing MOBA subsystems..."

# Spawn Roshan at pit location (near center, slightly dire-side)
$E entity create --mesh assets/generated/roshan.glb --position=-5,0,8 --scale 1.5,1.5,1.5 --color orange --health 6000 --physics Kinematic --combat --combat-damage 75 --combat-range 3 --combat-speed 2 --combat-cooldown 2.0 --gold-bounty 225 --xp-bounty 400 2>/dev/null
echo "  Roshan spawned at pit."

# Day/night cycle and ward stock are initialized in-engine by DotaMobaState;
# the shell script sets up the entity world, the client code handles the rest.
echo "  Day/night cycle: 5m day / 5m night (managed by engine)"
echo "  Ward stock: 2 observers, 3 sentries per team (managed by engine)"

# ── 4. Spawn neutral jungle camps ──
echo "Spawning neutral camps..."

# Radiant-side camps (between lanes, left half of map)
$E entity create --mesh assets/generated/neutral_wolf.glb --position=-18,0.5,10 --scale 0.6,0.6,0.6 --color green --health 500 --physics Kinematic --combat --combat-damage 25 --combat-range 2 --combat-speed 3 --combat-cooldown 1.2 --gold-bounty 60 --xp-bounty 80 2>/dev/null
$E entity create --mesh assets/generated/neutral_wolf.glb --position=-18,0.5,-10 --scale 0.6,0.6,0.6 --color green --health 500 --physics Kinematic --combat --combat-damage 25 --combat-range 2 --combat-speed 3 --combat-cooldown 1.2 --gold-bounty 60 --xp-bounty 80 2>/dev/null
$E entity create --mesh assets/generated/neutral_troll.glb --position=-24,0.5,10 --scale 0.8,0.8,0.8 --color green --health 800 --physics Kinematic --combat --combat-damage 35 --combat-range 2 --combat-speed 2 --combat-cooldown 1.5 --gold-bounty 100 --xp-bounty 120 2>/dev/null

# Dire-side camps (between lanes, right half of map)
$E entity create --mesh assets/generated/neutral_wolf.glb --position=18,0.5,10 --scale 0.6,0.6,0.6 --color green --health 500 --physics Kinematic --combat --combat-damage 25 --combat-range 2 --combat-speed 3 --combat-cooldown 1.2 --gold-bounty 60 --xp-bounty 80 2>/dev/null
$E entity create --mesh assets/generated/neutral_wolf.glb --position=18,0.5,-10 --scale 0.6,0.6,0.6 --color green --health 500 --physics Kinematic --combat --combat-damage 25 --combat-range 2 --combat-speed 3 --combat-cooldown 1.2 --gold-bounty 60 --xp-bounty 80 2>/dev/null
$E entity create --mesh assets/generated/neutral_troll.glb --position=24,0.5,-10 --scale 0.8,0.8,0.8 --color green --health 800 --physics Kinematic --combat --combat-damage 35 --combat-range 2 --combat-speed 2 --combat-cooldown 1.5 --gold-bounty 100 --xp-bounty 120 2>/dev/null

echo "  6 neutral camps spawned."

# ── 5. Add win condition rules on Ancients — find dynamically ──
echo "Setting up win conditions..."

# Find Ancient entity IDs by role=structure + team
ANCIENT_T1=$(echo "$OBSERVE" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for e in data:
    c = e.get('components', {})
    if c.get('EntityRole') == 'Structure' and c.get('Team', {}).get('0', 0) == 1:
        print(e['id']); break
" 2>/dev/null)

ANCIENT_T2=$(echo "$OBSERVE" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for e in data:
    c = e.get('components', {})
    if c.get('EntityRole') == 'Structure' and c.get('Team', {}).get('0', 0) == 2:
        print(e['id']); break
" 2>/dev/null)

if [ -n "$ANCIENT_T1" ]; then
  $E rule create --when death --filter "entity:$ANCIENT_T1" --do-action "endgame 2" 2>/dev/null
fi
if [ -n "$ANCIENT_T2" ]; then
  $E rule create --when death --filter "entity:$ANCIENT_T2" --do-action "endgame 1" 2>/dev/null
fi
echo "  Ancients: T1=$ANCIENT_T1, T2=$ANCIENT_T2"

echo "  Ancient death → game over."

# ── 6. Start the match ──
echo "Starting match..."
$E sim play

echo ""
echo "=== EUCA DOTA ready ==="
echo "  Team 1 (Radiant) — cyan, base at (-25, -25)"
echo "  Team 2 (Dire)    — red,  base at (+25, +25)"
echo "  3 L-shaped lanes (top: left+top, mid: diagonal, bot: bottom+right)"
echo "  24 towers, 6 neutral camps, minion waves every 30s"
echo "  Roshan at pit (-5, 0, 8) — drops Aegis, Cheese, Refresher Shard"
echo "  Day/night cycle: 5m day / 5m night (affects vision range)"
echo "  Wards: observer (6m, 1600 vision) + sentry (4m, 900 true sight)"
echo "  Hero attributes: STR/AGI/INT with per-level growth"
echo "  Destroy the enemy Ancient to win!"
