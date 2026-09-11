#!/usr/bin/env bash
# Kill patterns are ANCHORED to the start of the command line (^python3 /^cloudflared).
#
# WHY THIS MATTERS: Tailscale SSH puts the entire remote command into the session's own
# cmdline, so an unanchored `pkill -f perception_demo.py` matches the ssh session whenever
# that string appears anywhere in the command being run -- killing the connection (exit 255,
# hit three times). The ssh session's cmdline begins with "bash -c" or the tailscaled
# be-child, never with "python3 " or "cloudflared ", so anchoring makes it unmatchable.
pkill -f "^python3 /home/bmo/bmo_perception_demo\.py" 2>/dev/null
pkill -f "^cloudflared tunnel" 2>/dev/null
sleep 1
LEFT=$(pgrep -f "^python3 /home/bmo/bmo_perception_demo\.py" | wc -l)
echo "stopped (remaining demo procs: $LEFT)"
