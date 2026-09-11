#!/usr/bin/env bash
# Safe to run over Tailscale SSH: the kill patterns live in this file, not the ssh cmdline.
set -u
pkill -f "cloudflared tunnel" 2>/dev/null
sleep 1
setsid nohup cloudflared tunnel --url http://127.0.0.1:8899 > /home/bmo/tunnel2.log 2>&1 < /dev/null &
for i in $(seq 1 30); do
  U=$(grep -oE "https://[a-z0-9]+(-[a-z0-9]+)+\.trycloudflare\.com" /home/bmo/tunnel2.log 2>/dev/null | head -1)
  [ -n "$U" ] && break
  sleep 1
done
echo "TUNNEL: ${U:-FAILED}"
[ -z "${U:-}" ] && grep -iE "error|failed|deadline" /home/bmo/tunnel2.log | tail -2
