#!/bin/bash
# use_drm_face.sh -- switch the face engine from the Xorg build to the DRM build.
# RUN THIS ONLY WITH A SCREEN PLUGGED INTO DISPLAYPORT.
#
# Why: bmo_face_engine.service points at face_engine/BMO Face Engine/build/BMO_Engine,
# which was built PLATFORM=Desktop + GRAPHICS_API_OPENGL_ES2 -- a desktop GL context that
# then fails to compile every OpenGL ES shader. The DRM build
# (face_engine_drm/BMO Face Engine/build_drm, PLATFORM=DRM + ES2) is internally consistent
# and needs no X server at all, which also frees Xorg's memory.
#
# Verified 2026-08-23: with NO screen attached, BOTH builds fail identically at
#   "DISPLAY: No suitable DRM connector found" -> no GL context -> every shader fails.
#   /sys/class/drm/card1-DP-1/status was 'disconnected'. That was the real cause of
#   16,489 crash-restarts, not the shaders.
set -euo pipefail

DRM_DIR="/home/bmo/bmo_production/face_engine_drm/BMO Face Engine/build_drm"
UNIT=/etc/systemd/system/bmo_face_engine.service

echo "== DRM connector status =="
FOUND=0
for c in /sys/class/drm/card*-*/status; do
    s=$(cat "$c" 2>/dev/null || true)
    printf "  %-34s %s\n" "$(basename "$(dirname "$c")")" "$s"
    [ "$s" = "connected" ] && FOUND=1
done
if [ "$FOUND" -eq 0 ]; then
    echo "!! No connected display. Plug a screen into DisplayPort first -- the DRM"
    echo "   engine cannot create a surface and will crash-loop exactly as before."
    exit 1
fi

echo "== stopping the X server (DRM needs to be DRM master) =="
sudo systemctl stop bmo_face_engine || true
sudo systemctl disable --now bmo_xorg

echo "== repointing bmo_face_engine.service at the DRM build =="
sudo tee "$UNIT" >/dev/null <<UNITEOF
[Unit]
Description=BMO Face Engine (DRM/KMS, no X server)
After=systemd-user-sessions.service

[Service]
User=bmo
SupplementaryGroups=video render
WorkingDirectory=$DRM_DIR
ExecStart=$DRM_DIR/BMO_Engine
Restart=always
RestartSec=2
# Do not let a crash loop burn a core if the panel is unplugged again.
StartLimitIntervalSec=60
StartLimitBurst=5

[Install]
WantedBy=multi-user.target
UNITEOF

sudo systemctl daemon-reload
sudo systemctl restart bmo_face_engine
sleep 6
echo "== result =="
systemctl is-active bmo_face_engine
if pgrep -x BMO_Engine >/dev/null; then
    P=$(pgrep -x BMO_Engine | head -1)
    echo "ALIVE pid=$P rss=$(awk '/VmRSS/{print int($2/1024)}' /proc/$P/status) MiB"
    echo "restarts=$(systemctl show bmo_face_engine -p NRestarts --value)"
else
    echo "DEAD -- check: journalctl -u bmo_face_engine -n 40 --no-pager"
fi
echo
echo "To revert to the Xorg build:"
echo "  sudo systemctl enable --now bmo_xorg"
echo "  (and restore the old ExecStart pointing at face_engine/BMO Face Engine/start_engine.sh)"
