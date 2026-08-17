#!/usr/bin/env bash
# Cold-start the lunarsim GUI for remote (Tailscale) viewing.
#
# Run this ON THE PC after any reboot or Tailscale restart. It re-applies the
# two non-persistent settings that the remote path needs and relaunches the
# server detached, so recovery mid-workshop is one command instead of the
# multi-step dance. Full background: memory remote-pc-gui-access.
#
#   ./scripts/start_gui.sh
#
# Then from the laptop: Chrome -> http://<pc-tailscale-ip>:8050
set -uo pipefail

PORT="${LUNARSIM_PORT:-8050}"
IFACE="${TS_IFACE:-tailscale0}"
MTU="${TS_MTU:-800}"
VENV_PY="${VENV_PY:-$HOME/topas/.venv/bin/python}"
LOG="${LUNARSIM_LOG:-$HOME/lunarsim_gui.log}"
export TOPAS_G4_DATA_DIR="${TOPAS_G4_DATA_DIR:-$HOME/G4Data}"

echo "[start_gui] $IFACE MTU -> $MTU (needs sudo; the Tailscale underlay here"
echo "            can't carry the default 1280 -- large packets get dropped and"
echo "            the page loads blank without this)"
sudo ip link set "$IFACE" mtu "$MTU" \
  || echo "[start_gui] WARN: MTU set failed (interface up? name right?) -- page may load blank"

# Stop any stale server holding the port (old localhost-bound or crashed run).
if ss -ltn "( sport = :$PORT )" 2>/dev/null | grep -q ":$PORT"; then
  echo "[start_gui] stopping existing server on :$PORT"
  pkill -f "port=$PORT" 2>/dev/null || pkill -f lunarsim.gui 2>/dev/null || true
  sleep 1
fi

echo "[start_gui] launching detached on 0.0.0.0:$PORT (log: $LOG)"
# The -c one-liner overrides gui.py's hardcoded 127.0.0.1 bind WITHOUT editing
# the file, so the server is reachable over Tailscale.
nohup "$VENV_PY" -c "from lunarsim.gui import app; app.run(host='0.0.0.0', port=$PORT)" \
  > "$LOG" 2>&1 &
sleep 2

echo "[start_gui] listen check:"
if ! ss -ltnp 2>/dev/null | grep ":$PORT"; then
  echo "[start_gui] ERROR: nothing listening on :$PORT -- last log lines:"
  tail -20 "$LOG"
  exit 1
fi

TS_IP="$(tailscale ip -4 2>/dev/null | head -1 || true)"
echo "[start_gui] READY -> http://${TS_IP:-<pc-tailscale-ip>}:$PORT"
