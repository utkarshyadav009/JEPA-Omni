#!/usr/bin/env bash
# Single startup command: cold to running perception demo, with the live web view and a
# public tunnel. Kills any previous instance first.
#
#   bash /home/bmo/start_bmo_demo.sh [seconds] [port]
#
# NOTE: the pkill patterns below must never appear in the *calling* command line -- Tailscale
# SSH puts the full remote command into the session's cmdline, so `pkill -f <pattern>` run
# directly over ssh matches and kills the session itself (observed: exit 255). Keeping the
# launch inside this file is what makes it safe.
set -u
SECS="${1:-1200}"
PORT="${2:-8899}"
LOG=/home/bmo/demo_live.log
TLOG=/home/bmo/tunnel.log

pkill -f "^python3 /home/bmo/bmo_perception_demo\.py" 2>/dev/null
pkill -f "^cloudflared tunnel" 2>/dev/null
sleep 1

T0=$(date +%s.%N)
setsid nohup python3 /home/bmo/bmo_perception_demo.py \
    --seconds "$SECS" --serve "$PORT" \
    --provenance /home/bmo/PROVENANCE_perception_demo.json \
    > "$LOG" 2>&1 < /dev/null &

# wait for the first rendered round rather than a fixed sleep
for i in $(seq 1 120); do
    if curl -s -o /dev/null -m 2 "http://127.0.0.1:$PORT/" 2>/dev/null; then
        if grep -q "ROUND\|where\|wearing" <(curl -s -m 2 "http://127.0.0.1:$PORT/") 2>/dev/null; then
            break
        fi
    fi
    sleep 1
done
T1=$(date +%s.%N)
echo "STARTUP: $(echo "$T1 - $T0" | bc) s  (cold -> first rendered round)"

setsid nohup cloudflared tunnel --url "http://127.0.0.1:$PORT" > "$TLOG" 2>&1 < /dev/null &
for i in $(seq 1 30); do
    U=$(grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" "$TLOG" 2>/dev/null | head -1)
    [ -n "$U" ] && break
    sleep 1
done
echo "TUNNEL:  ${U:-FAILED (see $TLOG)}"
echo "LOCAL:   http://127.0.0.1:$PORT"
