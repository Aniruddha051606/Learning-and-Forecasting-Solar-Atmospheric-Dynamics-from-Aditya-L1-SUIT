#!/usr/bin/env bash
# Install the SUIT-DYN buzzer service on the Ubuntu data machine:   sudo bash install.sh [port]
# Re-running it updates the script and restarts the service. Remove with:   sudo bash install.sh --uninstall
set -euo pipefail
[ "$(id -u)" -eq 0 ] || { echo "run it with sudo:  sudo bash $0 $*"; exit 1; }

if [ "${1:-}" = "--uninstall" ]; then
  systemctl disable --now suitdyn-buzzer 2>/dev/null || true
  rm -f /etc/systemd/system/suitdyn-buzzer.service /etc/udev/rules.d/70-suitdyn-pcspkr.rules \
        /etc/modules-load.d/suitdyn-pcspkr.conf
  rm -rf /opt/suitdyn-buzzer
  systemctl daemon-reload
  echo "removed (the group 'beep' and any firewall rule are left in place)"
  exit 0
fi

PORT="${1:-8765}"
HERE="$(cd "$(dirname "$0")" && pwd)"
install -d /opt/suitdyn-buzzer
install -m 0755 "$HERE/suitdyn_buzzer.py" /opt/suitdyn-buzzer/suitdyn_buzzer.py

# 1. access: the service runs as an unprivileged user in group 'beep', allowed to write the PC-speaker device only
getent group beep >/dev/null || groupadd --system beep
cat > /etc/udev/rules.d/70-suitdyn-pcspkr.rules <<'EOF'
ACTION=="add", SUBSYSTEM=="input", ATTRS{name}=="PC Speaker", ENV{DEVNAME}!="", GROUP="beep", MODE="0620"
EOF
udevadm control --reload-rules

# 2. driver: Ubuntu blacklists pcspkr for automatic loading; load it now and at every boot
echo pcspkr > /etc/modules-load.d/suitdyn-pcspkr.conf
modprobe pcspkr || echo "WARNING: could not load the pcspkr module"
udevadm settle || true
DEV=""
for d in /sys/class/input/event*; do
  if [ "$(cat "$d/device/name" 2>/dev/null)" = "PC Speaker" ]; then DEV="/dev/input/$(basename "$d")"; fi
done
if [ -n "$DEV" ]; then
  chgrp beep "$DEV" && chmod 0620 "$DEV"
  echo "PC speaker device: $DEV"
else
  echo "WARNING: no 'PC Speaker' input device. The board may have no buzzer driver; the service will run but stay silent."
fi

# 3. service
cat > /etc/systemd/system/suitdyn-buzzer.service <<EOF
[Unit]
Description=SUIT-DYN buzzer (PC-speaker alerts from the training laptop)
After=network-online.target
Wants=network-online.target

[Service]
ExecStart=/usr/bin/python3 /opt/suitdyn-buzzer/suitdyn_buzzer.py serve --port ${PORT}
DynamicUser=yes
SupplementaryGroups=beep
NoNewPrivileges=yes
ProtectHome=yes
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable suitdyn-buzzer >/dev/null
systemctl restart suitdyn-buzzer

# 4. firewall: only if ufw is active; LAN and Tailscale ranges only (the service also refuses other clients)
if command -v ufw >/dev/null && ufw status | grep -q "Status: active"; then
  for net in 192.168.0.0/16 10.0.0.0/8 172.16.0.0/12 100.64.0.0/10; do
    ufw allow from "$net" to any port "$PORT" proto tcp >/dev/null
  done
  echo "ufw: port $PORT opened for LAN and Tailscale addresses"
fi

# 5. check: the service answers and the buzzer sounds (two short beeps)
sleep 2
python3 - "$PORT" <<'EOF'
import json, sys, urllib.request
base = f"http://127.0.0.1:{sys.argv[1]}"
urllib.request.urlopen(base + "/event?kind=test&text=installed", timeout=5).read()
s = json.loads(urllib.request.urlopen(base + "/", timeout=5).read())
print("service answers; speaker device:", s["speaker"])
EOF
echo "installed. You should have heard two short beeps. Log: journalctl -u suitdyn-buzzer -f"
