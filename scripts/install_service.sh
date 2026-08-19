#!/usr/bin/env bash
# Install lunarsim as a systemd SYSTEM service so the GUI self-heals: it starts
# on boot and restarts within seconds if it crashes -- no human SSHing in to
# rerun start_gui.sh mid-workshop.
#
# Run this ONCE, on the PC, as root:
#
#   sudo ./scripts/install_service.sh
#
# It resolves the target user, home, venv and `ip` path from the current
# environment, writes /etc/systemd/system/lunarsim-gui.service, then enables and
# starts it. Re-running is safe (it overwrites and restarts).
#
# After install, the deploy step becomes:   git pull && sudo systemctl restart lunarsim-gui
# Watch it live:                            journalctl -u lunarsim-gui -f
# Stop / disable:                           sudo systemctl disable --now lunarsim-gui
#
# The service supersedes the manual scripts/start_gui.sh: don't run both, they
# fight over the port. start_gui.sh stays useful for a quick ad-hoc launch on a
# machine where you don't want a permanent service.
set -euo pipefail

SERVICE="lunarsim-gui"
UNIT="/etc/systemd/system/${SERVICE}.service"
PORT="${LUNARSIM_PORT:-8050}"
IFACE="${TS_IFACE:-tailscale0}"
MTU="${TS_MTU:-800}"
THREADS="${LUNARSIM_THREADS:--2}"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "[install_service] must run as root -- use: sudo $0" >&2
  exit 1
fi

# Whom to run the server as (never root): the human who invoked sudo, falling
# back to the login name, then to the owner of the topas checkout.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_USER="${SUDO_USER:-$(logname 2>/dev/null || true)}"
if [[ -z "${TARGET_USER}" || "${TARGET_USER}" == "root" ]]; then
  TARGET_USER="$(stat -c '%U' "${REPO_ROOT}")"
fi
TARGET_HOME="$(getent passwd "${TARGET_USER}" | cut -d: -f6)"
if [[ -z "${TARGET_HOME}" ]]; then
  echo "[install_service] could not resolve home for user '${TARGET_USER}'" >&2
  exit 1
fi

VENV_PY="${VENV_PY:-${TARGET_HOME}/topas/.venv/bin/python}"
G4DATA="${TOPAS_G4_DATA_DIR:-${TARGET_HOME}/G4Data}"
IP_BIN="$(command -v ip || echo /usr/sbin/ip)"

if [[ ! -x "${VENV_PY}" ]]; then
  echo "[install_service] venv python not found/executable: ${VENV_PY}" >&2
  echo "                  set VENV_PY=/path/to/python and re-run." >&2
  exit 1
fi

echo "[install_service] user=${TARGET_USER} home=${TARGET_HOME}"
echo "[install_service] python=${VENV_PY}"
echo "[install_service] port=${PORT} iface=${IFACE} mtu=${MTU} threads=${THREADS}"

cat > "${UNIT}" <<UNITEOF
[Unit]
Description=lunarsim radiation GUI (Dash, Tailscale-facing)
After=network-online.target tailscaled.service
Wants=network-online.target
# A crash-loop shouldn't hammer forever: back off after 5 failures in 60s. Clear
# it manually with: systemctl reset-failed lunarsim-gui. (These are [Unit] keys
# in systemd >=230 -- they are silently ignored if placed under [Service].)
StartLimitIntervalSec=60
StartLimitBurst=5

[Service]
Type=simple
User=${TARGET_USER}
WorkingDirectory=${TARGET_HOME}/topas
Environment=TOPAS_G4_DATA_DIR=${G4DATA}
Environment=LUNARSIM_THREADS=${THREADS}
Environment=LUNARSIM_PORT=${PORT}
# The Tailscale underlay here can't carry the default 1280 MTU -- large packets
# get dropped and the page loads blank. Shrink it before launch. Runs as root
# (system service), so no sudo; leading '-' tolerates the iface not being up yet.
ExecStartPre=-${IP_BIN} link set ${IFACE} mtu ${MTU}
ExecStart=${VENV_PY} -c "from lunarsim.gui import app; app.run(host='0.0.0.0', port=${PORT})"
Restart=always
RestartSec=5
# On stop/restart, reap any TOPAS children the run spawned, then hard-kill after
# a grace period so a stuck run can't wedge the restart.
KillMode=mixed
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
UNITEOF

echo "[install_service] wrote ${UNIT}"
systemctl daemon-reload
systemctl enable --now "${SERVICE}"
sleep 2

echo "[install_service] status:"
systemctl --no-pager --lines=0 status "${SERVICE}" || true

if ss -ltn "( sport = :${PORT} )" 2>/dev/null | grep -q ":${PORT}"; then
  TS_IP="$(sudo -u "${TARGET_USER}" tailscale ip -4 2>/dev/null | head -1 || true)"
  echo "[install_service] READY -> http://${TS_IP:-<pc-tailscale-ip>}:${PORT}"
  echo "[install_service] logs:  journalctl -u ${SERVICE} -f"
else
  echo "[install_service] WARN: nothing listening on :${PORT} yet -- check: journalctl -u ${SERVICE} -e" >&2
fi
