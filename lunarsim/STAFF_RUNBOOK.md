# Habitat Dose — Staff Runbook

Operational reference for running the Habitat Dose GUI during the workshop.
Keep this open on the admin laptop. Everything here is PC-side unless noted.

- **PC:** TopasSimulationPC, user `simulationuser`, Tailscale IP `100.126.83.20`
- **GUI URL (any device):** `http://100.126.83.20:8050`
- **Service:** `lunarsim-gui` (systemd, auto-start + auto-restart)

---

## 1. Who connects how

| Role | Needs | Notes |
|---|---|---|
| **Teams / viewers** | Tailscale + a browser | Open the GUI URL. No SSH, no key, nothing to install. Unlimited devices, simultaneously. |
| **Admin (deploy/restart)** | SSH to the PC **from Windows PowerShell** | The WSL→Windows hop drops packets, so SSH must originate from PowerShell, not WSL. |

The GUI is a **web server on the PC**. Every device is just a browser client of the
one backend — there is a single shared job queue, not private per-user sessions.
Two teams + staff is well within capacity. (See §6 for the multi-device details.)

---

## 2. Deploy an update

Code-only change (GUI / Python):
```bash
cd ~/topas && git pull
sudo systemctl restart lunarsim-gui
```

Change that touches C++ extensions or the TOPAS binary (**PC rebuilds** — the repo
does not ship a prebuilt binary for the PC):
```bash
cd ~/topas && git pull
cmake .              # only if new extension files were added
make                 # rebuild bin/topas with the extensions
sudo systemctl restart lunarsim-gui
```

After any restart, hard-refresh the browser (**Ctrl-Shift-R**) and confirm the header
reads **"Habitat Dose"** (not "Radiation Sim" — a stale process once caused exactly
that; see §4).

---

## 3. Health checks

```bash
systemctl status lunarsim-gui          # active (running)?
systemctl is-enabled lunarsim-gui      # enabled  -> survives a reboot
journalctl -u lunarsim-gui -n 50       # recent logs / errors
ss -ltnp | grep 8050                   # exactly ONE listener, owned by the service
```

Confirm the service environment has the Geant4 data path, or every job fails to find
neutron data:
```bash
systemctl show lunarsim-gui -p Environment   # must include TOPAS_G4_DATA_DIR=.../G4Data
```
(If a job is run by hand instead of through the service: `export TOPAS_G4_DATA_DIR=~/G4Data` first.)

---

## 4. Troubleshooting

**GUI shows old content / crash-loops on "Address already in use" / "Port 8050 is in use."**
A stale manual launch is squatting the port and blocking the service. Never run a manual
`nohup python ... gui.py` alongside the service — that is what causes this.
```bash
ss -ltnp | grep 8050          # find the non-systemd python PID
kill <that-pid>
sudo systemctl reset-failed lunarsim-gui
sudo systemctl restart lunarsim-gui
```

**A device loads a blank / stalled page over Tailscale (body never fills in).**
Known fix: lower that device's Tailscale interface MTU to **800**. (This was needed on the
WSL laptop; details in the `remote-pc-gui-access` notes.) A harmless
`RTNETLINK ... Operation not permitted` line on service start is expected and can be ignored.

**Job never finishes / queue looks stuck.** Check `journalctl -u lunarsim-gui -f` for the
running TOPAS process. Heavy-ion runs through thick shielding are genuinely slow — confirm
it is progressing, not hung, before restarting (a restart drops the queue).

---

## 5. Do NOT (during the workshop)

- Do **not** launch a manual `nohup`/`python gui.py` — fights the service for port 8050.
- Do **not** SSH from inside WSL — use Windows PowerShell.
- Do **not** change code once the dry run passes. Freeze on the known-good commit.

---

## 6. Adding a second device

**As a viewer (host the same GUI on another laptop):** nothing to add. Put the laptop on
the tailnet and open `http://100.126.83.20:8050`. Any number of devices can use the GUI at
once — they all drive the same backend and the same queue. The compute always runs on the
PC; laptops are only clients, so there is only ever one host to manage.

**As a second admin (deploy/restart from the new laptop too):** it needs its own SSH access.
- Generate a **fresh** keypair on the new laptop (`ssh-keygen -t ed25519`). Do **not** copy the
  existing key between machines.
- Add the new laptop's **public** key to `~simulationuser/.ssh/authorized_keys` on the PC.
- SSH from that laptop's **Windows PowerShell**, same as the primary.

Viewing needs none of this — only the admin role does.
