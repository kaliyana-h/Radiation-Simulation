#!/usr/bin/env bash
# Morning-of smoke test: prove the full TOPAS path works end-to-end before the
# workshop opens. Runs the default dome at quick-look statistics through the real
# bridge (spawns TOPAS, reads G4Data, parses the CSV) and exits 0 only if TOPAS
# completed and returned a real dose.
#
# Run ON the machine that has the TOPAS binary + G4Data (the PC):
#
#   ./scripts/smoke.sh && echo "safe to open the workshop"
set -uo pipefail

export TOPAS_G4_DATA_DIR="${TOPAS_G4_DATA_DIR:-$HOME/G4Data}"
# Exercise the multicore path the workshop will actually run under, so this also
# proves the MT-merged dose scorers work end-to-end (not just single-threaded).
export LUNARSIM_THREADS="${LUNARSIM_THREADS:--2}"
VENV_PY="${VENV_PY:-$HOME/topas/.venv/bin/python}"
cd "$(dirname "$0")/.."   # repo root, so `-m lunarsim.bridge` resolves

echo "[smoke] TOPAS_G4_DATA_DIR=$TOPAS_G4_DATA_DIR"
echo "[smoke] LUNARSIM_THREADS=$LUNARSIM_THREADS"
echo "[smoke] running default dome (quick-look) end-to-end ..."
OUT="$("$VENV_PY" -m lunarsim.bridge 2>&1)"
echo "$OUT"

if echo "$OUT" | grep -q "rc=0" && ! echo "$OUT" | grep -q "dose=None"; then
  echo "[smoke] PASS -- TOPAS completed and returned a dose."
  exit 0
fi
echo "[smoke] FAIL -- TOPAS did not complete cleanly (see output above)."
echo "[smoke]        common causes: missing G4NDL neutron data, or a scorer not"
echo "[smoke]        compiled into the binary. See memory remote-pc-gui-access."
exit 1
