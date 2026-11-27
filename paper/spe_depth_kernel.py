#!/usr/bin/env python3
"""SPE proton response kernel on a depth grid (candidate; replaces one-wall kernel).

Why. The shipped SPE kernel (lunarsim/data/spe_proton_kernel.json) was transported
through ONE wall, the 147.5 g/cm2 calibration dome, and its nodes start at 430 MeV,
the just-penetrating energy for that wall. Any design outside 110-184 g/cm2 is
therefore "NOT ASSESSED": a thinner wall lets through protons the kernel never saw,
and a thicker one is folded as if it were 147.5.

What this builds. The GCR thin-wall kernel method (lunarsim.kernel_gen), applied to
SPE protons:
  * concentric Al SHELL wall around the 15 cm water organ-shell phantom (production
    kernel_gen geometry; the per-element GCR kernel also transports through Al);
  * a DEPTH GRID of walls (default 0-200 g/cm2) instead of one wall;
  * one common dense proton node grid from 10 MeV to 20 GeV at every depth, so
    sub-penetrating protons and their secondaries are always sampled
    (no penetration gating; stopped protons are cheap);
  * three scorers per organ shell: DoseToMedium (D), DoseEquivalent_ICRP (I) and
    DoseEquivalent_NASA (N, same transport, Cucinotta Q);
  * beam spot scaled per depth to cover the shell's outer radius (the spot45
    lesson: an under-lit rim loses lateral in-scatter), with histories scaled
    ~(spot/22)^2 so the per-run fluence stays near the production value;
  * R = D / phi_ff, phi_ff = the exact free-field scalar fluence of the run.

NORMALISATION. No CAL. R is per unit free-field scalar fluence, the same convention
as the GCR thin-wall kernel. The event fluence it is folded against is the event's
>30 MeV anchor (make_source.SPE_EVENTS) times SKY_FRACTION:
  1.0 (default) = the shipped convention: the event arrives as a beam/cone from a
                  Sun above the horizon, so the whole anchor reaches the habitat;
  0.5           = isotropic free-space event with the Moon blocking 2 pi
                  (the HZETRN/OLTARIS lunar-surface convention).
The report prints both. Which one the GUI uses is a decision for later.

KNOWN LIMITS (also stamped into the JSON meta):
  * spherical shell, vacuum world: no lunar floor, so no ground albedo neutrons
    (same as the GCR kernel);
  * Al stands in for every wall material (per-element GCR convention);
  * King-1974 exp-in-rigidity spectra are used below 30 MeV (10-30 MeV bin),
    which matters only for the skin behind walls under ~1 g/cm2.

TARGET-BLIND: no external dose enters. The acceptance reads are internal:
per-node dose fractions, a node-thinning check, and a report-only comparison with
the shipped one-wall kernel at 147.5 g/cm2.

The output is a CANDIDATE file (lunarsim/data/spe_proton_kernel_depth.json). Wiring
it into dosimetry.fold_spe is a separate step and the user's decision.

Usage (PC, from ~/topas):
  .venv/bin/python paper/spe_depth_kernel.py emit --out runs/spe_depth
  export TOPAS_G4_DATA_DIR=~/G4Data
  bash runs/spe_depth/run_all.sh 2>&1 | tee runs/spe_depth.log
  .venv/bin/python paper/spe_depth_kernel.py collect --out runs/spe_depth
  .venv/bin/python paper/spe_depth_kernel.py report | tee paper/spe_depth_kernel_report.txt
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lunarsim.kernel_gen import config, generate, templates  # noqa: E402

MATERIAL = "aluminium"
DEPTHS_GCM2 = [0.0, 2.0, 5.0, 10.0, 20.0, 35.0, 50.0, 75.0, 100.0, 150.0, 200.0]
NODES_MEV = [10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 70.0, 85.0, 100.0,
             120.0, 140.0, 170.0, 200.0, 240.0, 290.0, 350.0, 420.0, 500.0, 600.0,
             720.0, 870.0, 1050.0, 1300.0, 1600.0, 2100.0, 3000.0, 5000.0, 9000.0,
             20000.0]
SEEDS = 8
HIST0 = 8                 # histories per beam at the 22 cm production spot
SPOT_MARGIN = 1.35        # spot >= 1.35 x shell outer radius (spot45: 45/33.5)
SPOT_STEP_CM = 5.0
QUANTS = ("D", "I", "N")
SCORERS = {"D": "DoseToMedium", "I": "DoseEquivalent_ICRP", "N": "DoseEquivalent_NASA"}
SKY_FRACTION = 1.0        # shipped convention (directional event); 0.5 = isotropic 2 pi
DATA = REPO_ROOT / "lunarsim" / "data"
DEST = DATA / "spe_proton_kernel_depth.json"
SHIPPED = DATA / "spe_proton_kernel.json"
WORLD_HALF_CM = config.BEAM_RADIUS_CM + 50.0


# --------------------------------------------------------------------------
# geometry / illumination
# --------------------------------------------------------------------------
def shell_outer_cm(wall_gcm2: float, material: str = MATERIAL) -> float:
    return config.WALL_RMIN_CM + wall_gcm2 / config.MATERIALS[material]["density"]


def spot_cm(wall_gcm2: float, material: str = MATERIAL) -> float:
    """Disc radius covering the shell rim with margin, never below production 22."""
    need = SPOT_MARGIN * shell_outer_cm(wall_gcm2, material)
    return max(float(config.BEAM_SPOT_CM), SPOT_STEP_CM * math.ceil(need / SPOT_STEP_CM))


def histories(wall_gcm2: float, hist0: int = HIST0, material: str = MATERIAL) -> int:
    """Per-beam histories, scaled with spot area so per-run fluence ~ constant."""
    s = spot_cm(wall_gcm2, material)
    return max(hist0, int(round(hist0 * (s / config.BEAM_SPOT_CM) ** 2)))


def phi_ff(wall_gcm2: float, hist0: int = HIST0, material: str = MATERIAL) -> float:
    return templates.phi_ff(spot_cm(wall_gcm2, material), histories(wall_gcm2, hist0, material))


def csv_name(organ: str, q: str) -> str:
    return f"{organ}_{q}"


def run_name(wall_gcm2: float, node: int, seed: int) -> str:
    return f"spe_{MATERIAL}_w{wall_gcm2:g}_n{node}_s{seed}"


def build_param_file(wall_gcm2: float, energy_mev: float, seed: int, threads: int = 1,
                     hist0: int = HIST0, material: str = MATERIAL) -> str:
    """One mono-energetic proton run: kernel_gen shell geometry + D/I/N scorers."""
    mat = config.MATERIALS[material]
    spot = spot_cm(wall_gcm2, material)
    hist = histories(wall_gcm2, hist0, material)
    r_out = shell_outer_cm(wall_gcm2, material)
    if r_out >= config.BEAM_RADIUS_CM:
        raise ValueError(f"wall {wall_gcm2:g} g/cm2 reaches the source sphere")
    if math.hypot(config.BEAM_RADIUS_CM, spot) >= WORLD_HALF_CM:
        raise ValueError(f"{spot:g} cm spot leaves the world")
    L = [
        "# ============================================================",
        "# SPE proton response-kernel run (paper/spe_depth_kernel.py)",
        f"#   wall={wall_gcm2:g} g/cm^2 {material} shell  R_out={r_out:.3f} cm",
        f"#   proton E={energy_mev:g} MeV  seed={seed}",
        f"#   rings{config.RINGS} x az{config.AZIMUTH} x hist{hist}  beam_spot={spot:g} cm",
        f"#   phi_ff={templates.phi_ff(spot, hist):.6g} /cm^2",
        "# ============================================================",
        "",
        f"i:Ts/Seed            = {seed}",
        f"i:Ts/NumberOfThreads = {threads}",
        'b:Ts/PauseBeforeQuit = "False"',
        "i:Ts/ShowHistoryCountAtInterval = 0",
        f's:Ph/Default/Type = "{config.PHYSICS}"',
        "",
        's:Ge/World/Type      = "TsBox"',
        's:Ge/World/Material  = "Vacuum"',
        f"d:Ge/World/HLX       = {WORLD_HALF_CM:.1f} cm",
        f"d:Ge/World/HLY       = {WORLD_HALF_CM:.1f} cm",
        f"d:Ge/World/HLZ       = {WORLD_HALF_CM:.1f} cm",
        'b:Ge/World/Invisible = "true"',
        "",
    ]
    if mat["defn"]:
        L.extend(mat["defn"] + [""])
    if wall_gcm2 > 0:
        L += ['s:Ge/Wall/Type     = "TsSphere"',
              's:Ge/Wall/Parent   = "World"',
              f's:Ge/Wall/Material = "{mat["topas"]}"',
              f"d:Ge/Wall/RMin     = {config.WALL_RMIN_CM:.4f} cm",
              f"d:Ge/Wall/RMax     = {r_out:.4f} cm",
              'b:Ge/Wall/Invisible = "true"', ""]
    for name, r_in, r_o in config.SHELLS:
        comp = "Shell_" + name
        L += [f's:Ge/{comp}/Type     = "TsSphere"',
              f's:Ge/{comp}/Parent   = "World"',
              f's:Ge/{comp}/Material = "G4_WATER"',
              f"d:Ge/{comp}/RMin     = {r_in:.4f} cm",
              f"d:Ge/{comp}/RMax     = {r_o:.4f} cm"]
        for q in QUANTS:
            sc = f"{comp}_{q}"
            L += [f's:Sc/{sc}/Quantity   = "{SCORERS[q]}"',
                  f's:Sc/{sc}/Component  = "{comp}"',
                  f's:Sc/{sc}/OutputFile = "{csv_name(name, q)}"',
                  f's:Sc/{sc}/OutputType = "csv"',
                  f's:Sc/{sc}/IfOutputFileAlreadyExists = "Overwrite"']
        L.append("")
    L += [f"d:Ge/BeamRadius = {config.BEAM_RADIUS_CM:.1f} cm",
          f"d:So/BeamSpot   = {spot:.1f} cm", ""]
    for idx, (phi, theta) in enumerate(templates._ring_directions()):
        grp, src = templates._ring_groups(idx, phi, theta)
        tag = f"K{idx}"
        L += grp + [
            f's:So/{tag}/Type      = "Beam"',
            f's:So/{tag}/Component = "{src}"',
            f's:So/{tag}/BeamParticle = "proton"',
            f"d:So/{tag}/BeamEnergy = {energy_mev:.6g} MeV",
            f"u:So/{tag}/BeamEnergySpread = 0.0",
            f's:So/{tag}/BeamPositionDistribution = "Flat"',
            f's:So/{tag}/BeamPositionCutoffShape  = "Ellipse"',
            f"d:So/{tag}/BeamPositionCutoffX      = So/BeamSpot cm",
            f"d:So/{tag}/BeamPositionCutoffY      = So/BeamSpot cm",
            f's:So/{tag}/BeamAngularDistribution  = "None"',
            f"i:So/{tag}/NumberOfHistoriesInRun   = {hist}",
            ""]
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------
# emit / collect
# --------------------------------------------------------------------------
def emit(out: Path, depths=DEPTHS_GCM2, nodes=NODES_MEV, seeds: int = SEEDS,
         threads: int = 1, hist0: int = HIST0) -> int:
    runs = out / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    man = {"material": MATERIAL, "depths_gcm2": [float(d) for d in depths],
           "nodes_mev": [float(e) for e in nodes], "seeds": seeds, "threads": threads,
           "hist0": hist0, "spot_margin": SPOT_MARGIN, "physics": config.PHYSICS,
           "points": [{"wall_gcm2": float(d), "beam_spot_cm": spot_cm(d),
                       "histories": histories(d, hist0), "phi_ff_cm2": phi_ff(d, hist0)}
                      for d in depths],
           "runs": []}
    for d in depths:
        for j, e in enumerate(nodes):
            for seed in range(1, seeds + 1):
                rn = run_name(d, j, seed)
                (runs / rn).mkdir(exist_ok=True)
                (runs / rn / "run.txt").write_text(
                    build_param_file(d, e, seed, threads, hist0))
                man["runs"].append({"name": rn, "wall_gcm2": float(d), "node": j,
                                    "seed": seed})
    (out / "manifest.json").write_text(json.dumps(man, indent=2))
    generate._write_runner(out)
    n = len(man["runs"])
    prim = sum(config.RINGS * config.AZIMUTH * p["histories"] for p in man["points"])
    print(f"{n} runs ({len(depths)} depths x {len(nodes)} nodes x {seeds} seeds) -> {out}")
    for p in man["points"]:
        print(f"  {p['wall_gcm2']:6g} g/cm2  spot {p['beam_spot_cm']:5g} cm  "
              f"hist {p['histories']:4d}/beam  phi_ff {p['phi_ff_cm2']:.4g} /cm2")
    print(f"  primaries per seed per node summed over depths: {prim}")
    return n


def _missing(out: Path, man: dict) -> list:
    organ0 = config.SHELLS[0][0]
    return [r["name"] for r in man["runs"]
            if not all((out / "runs" / r["name"] / f"{csv_name(organ0, q)}.csv").exists()
                       for q in QUANTS)]


def collect(out: Path, dest: Path = DEST) -> dict:
    man = json.loads((out / "manifest.json").read_text())
    if man.get("material") != MATERIAL:
        raise SystemExit(f"{out}: manifest material {man.get('material')!r} != {MATERIAL!r}")
    miss = _missing(out, man)
    if miss:
        raise SystemExit(f"{out}: {len(miss)} runs have no output, e.g. {miss[:3]}")
    organs = [s[0] for s in config.SHELLS]
    nodes, seeds = man["nodes_mev"], man["seeds"]
    points = []
    for p in man["points"]:
        d, phi = p["wall_gcm2"], p["phi_ff_cm2"]
        if abs(phi - templates.phi_ff(p["beam_spot_cm"], p["histories"])) > 1e-12 * phi:
            raise SystemExit(f"{out}: phi_ff at {d:g} inconsistent with spot/histories")
        R = {o: {q: [0.0] * len(nodes) for q in QUANTS} for o in organs}
        S = {o: {q: [0.0] * len(nodes) for q in QUANTS} for o in organs}
        for j in range(len(nodes)):
            for o in organs:
                for q in QUANTS:
                    vals = []
                    for seed in range(1, seeds + 1):
                        f = out / "runs" / run_name(d, j, seed) / f"{csv_name(o, q)}.csv"
                        v = generate._read_scalar_csv(f)
                        if v is None:
                            raise SystemExit(f"unreadable scorer output {f}")
                        vals.append(v / phi)
                    R[o][q][j] = statistics.fmean(vals)
                    S[o][q][j] = (statistics.stdev(vals) / math.sqrt(len(vals))
                                  if len(vals) > 1 else 0.0)
        points.append({**p, "seeds": seeds, "sem_estimator": "sample", "R": R, "Rsem": S})
    K = {"meta": {
            "species": "H", "material": MATERIAL, "wall_geometry": "shell",
            "physics": man["physics"], "nodes_mev": nodes,
            "depths_gcm2": man["depths_gcm2"], "hist0": man["hist0"],
            "spot_margin": man["spot_margin"],
            "organs": [[o, w] for o, w in config.organs_from_reference()],
            "units": "R [Gy cm^2] per unit free-field scalar fluence (D, I, N)",
            "normalisation": ("no CAL; fold against event fluence x sky fraction "
                              "(1.0 directional / 0.5 isotropic 2pi)"),
            "limits": ["spherical Al shell, vacuum world: no lunar floor albedo",
                       "Al stands in for every wall material",
                       "King-1974 spectrum extrapolated below 30 MeV (10-30 MeV bin)"],
            "provenance": generate._provenance(),
            "status": "CANDIDATE (paper/spe_depth_kernel.py); not wired into fold_spe"},
         "H": {"points": points}}
    dest.write_text(json.dumps(K, indent=2))
    print(f"collected {len(points)} depths x {len(nodes)} nodes -> {dest}")
    return K


# --------------------------------------------------------------------------
# fold (report-side; mirrors dosimetry.fold_spe minus CAL/gauge)
# --------------------------------------------------------------------------
def bin_edges(nodes: list) -> list:
    return ([nodes[0]] + [math.sqrt(nodes[j] * nodes[j + 1]) for j in range(len(nodes) - 1)]
            + [nodes[-1]])


def bin_fluence(event: str, nodes: list) -> list:
    """Event proton fluence [/cm^2] per node bin (King-1974, >30 MeV anchor)."""
    from lunarsim import dosimetry
    r0 = dosimetry._load_make_source().SPE_EVENTS[event][0]
    return dosimetry._spe_bin_fluence(event, r0, None, bin_edges(nodes))


def interp_point(K: dict, wall: float) -> dict:
    """R/Rsem at an arbitrary wall: log-linear in depth between bracketing points
    (linear where either end is zero). Outside the grid -> ValueError."""
    pts = sorted(K["H"]["points"], key=lambda p: p["wall_gcm2"])
    ws = [p["wall_gcm2"] for p in pts]
    if not ws[0] <= wall <= ws[-1]:
        raise ValueError(f"wall {wall:g} outside kernel grid {ws[0]:g}-{ws[-1]:g} g/cm2")
    for p in pts:
        if p["wall_gcm2"] == wall:
            return {"R": p["R"], "Rsem": p["Rsem"]}
    i = next(k for k in range(1, len(ws)) if ws[k] > wall)
    a, b = pts[i - 1], pts[i]
    t = (wall - ws[i - 1]) / (ws[i] - ws[i - 1])

    def mix(x, y):
        if x > 0 and y > 0:
            return math.exp((1 - t) * math.log(x) + t * math.log(y))
        return (1 - t) * x + t * y

    out = {"R": {}, "Rsem": {}}
    for o in a["R"]:
        out["R"][o] = {q: [mix(x, y) for x, y in zip(a["R"][o][q], b["R"][o][q])]
                       for q in a["R"][o]}
        out["Rsem"][o] = {q: [math.hypot((1 - t) * x, t * y) for x, y in
                              zip(a["Rsem"][o][q], b["Rsem"][o][q])] for q in a["Rsem"][o]}
    return out


def fold(point: dict, pb: list, sky: float = SKY_FRACTION) -> dict:
    """{organ: {q: (dose, sem, max_node_fraction)}} for one depth."""
    res = {}
    for o, Rq in point["R"].items():
        res[o] = {}
        for q, R in Rq.items():
            terms = [r * f * sky for r, f in zip(R, pb)]
            val = sum(terms)
            sem = math.sqrt(sum((s * f * sky) ** 2 for s, f in zip(point["Rsem"][o][q], pb)))
            res[o][q] = (val, sem, max(terms) / val if val > 0 else 0.0)
    return res


def _spectrum(event: str):
    """(dN/dE callable, scale) with the same >30 MeV normalisation as dosimetry."""
    from lunarsim import dosimetry
    ms = dosimetry._load_make_source()
    r0 = ms.SPE_EVENTS[event][0]
    lo, hi, n = 30.0, 25000.0, 40000
    g = [lo * (hi / lo) ** (i / (n - 1)) for i in range(n)]
    f = [dosimetry._spe_dNdE(e, r0, None) for e in g]
    tot = sum(0.5 * (f[i] + f[i + 1]) * (g[i + 1] - g[i]) for i in range(n - 1))
    return (lambda e: dosimetry._spe_dNdE(e, r0, None)), ms.spe_integral_fluence(event) / tot


def loglog_weights(event: str, nodes: list, m: int = 48) -> list:
    """Per node-interval quadrature [(t_k, w_k)]: fluence weights w_k [/cm^2] at
    fractional log-E positions t_k between nodes j and j+1."""
    dnde, scale = _spectrum(event)
    out = []
    for a, b in zip(nodes[:-1], nodes[1:]):
        g = [a * (b / a) ** (k / m) for k in range(m + 1)]
        pts = []
        for k in range(m):
            e = math.sqrt(g[k] * g[k + 1])
            pts.append((math.log(e / a) / math.log(b / a), scale * dnde(e) * (g[k + 1] - g[k])))
        out.append(pts)
    return out


def fold_loglog(point: dict, weights: list, sky: float = SKY_FRACTION) -> dict:
    """Fold with R interpolated log-log between nodes (linear where an end is 0).
    {organ: {q: dose}}. Removes the piecewise-constant bin error at range thresholds."""
    res = {}
    for o, Rq in point["R"].items():
        res[o] = {}
        for q, R in Rq.items():
            tot = 0.0
            for j, pts in enumerate(weights):
                x, y = R[j], R[j + 1]
                if x > 0 and y > 0:
                    lx, ly = math.log(x), math.log(y)
                    tot += sum(w * math.exp(lx + t * (ly - lx)) for t, w in pts)
                else:
                    tot += sum(w * (x + t * (y - x)) for t, w in pts)
            res[o][q] = sky * tot
    return res


def thinned(point: dict, nodes: list) -> tuple[dict, list]:
    """Every-other-node kernel (keeps both ends) for the discretisation check."""
    keep = sorted(set(range(0, len(nodes), 2)) | {len(nodes) - 1})
    sub = {k: {o: {q: [v[j] for j in keep] for q, v in Rq.items()}
               for o, Rq in point[k].items()} for k in ("R", "Rsem")}
    return sub, [nodes[j] for j in keep]


def old_kernel_fold(event: str) -> dict:
    """Shipped one-wall kernel at its calibration dome, exactly as the GUI folds it."""
    from lunarsim import dosimetry
    from lunarsim.bridge import SPEScenario
    from lunarsim.spec import HabitatSpec, WallLayer
    meta = dosimetry._load_spe_kernel()["meta"]
    spec = HabitatSpec(name="spe_cal", shape=meta["cal_shape"],
                       inner_radius_cm=meta["cal_inner_radius_cm"],
                       walls=[WallLayer(m, t) for m, t in meta["cal_wall"]])
    sc = SPEScenario(name=event, event=event, fluence_cm2=0.0)
    return dosimetry.fold_spe(sc, spec)["shells"]


def below_cut_share(point: dict, event: str, nodes: list, cut_mev: float) -> dict:
    """{organ: {q: (dose_above_cut, share_below_cut)}} with bin fluence clipped at cut."""
    from lunarsim import dosimetry
    r0 = dosimetry._load_make_source().SPE_EVENTS[event][0]
    edges = [max(e, cut_mev) for e in bin_edges(nodes)]
    pb_hi = [dosimetry._spe_bin_fluence(event, r0, None, [a, b])[0] if b > a else 0.0
             for a, b in zip(edges[:-1], edges[1:])]
    full, hi = fold(point, bin_fluence(event, nodes)), fold(point, pb_hi)
    return {o: {q: (hi[o][q][0], 1.0 - hi[o][q][0] / full[o][q][0] if full[o][q][0] else 0.0)
                for q in hi[o]} for o in hi}


def report(path: Path = DEST, events=None, walls=None) -> dict:
    from lunarsim import dosimetry
    K = json.loads(path.read_text())
    nodes = K["meta"]["nodes_mev"]
    ms = dosimetry._load_make_source()
    events = events or list(ms.SPE_EVENTS)
    grid = K["meta"]["depths_gcm2"]
    walls = walls or grid
    out = {}
    print(f"SPE depth kernel {path.name}: {len(grid)} depths, {len(nodes)} nodes "
          f"{nodes[0]:g}-{nodes[-1]:g} MeV, {K['H']['points'][0]['seeds']} seeds")
    print(f"provenance: {K['meta'].get('provenance')}")
    print(f"sky fraction {SKY_FRACTION:g} (directional, shipped convention); "
          "isotropic-2pi values are exactly half\n")
    for ev in events:
        pb = bin_fluence(ev, nodes)
        print(f"== {ev}: R0 {ms.SPE_EVENTS[ev][0]:g} MV, phi(>30) "
              f"{ms.spe_integral_fluence(ev):.3g} /cm2, binned {sum(pb):.3g} /cm2")
        print(f"{'wall':>7} {'skin D':>9} {'skin N':>9} {'skin I':>9} {'BFO D':>9} "
              f"{'BFO N':>9} {'Q_N':>5} {'rel':>6} {'maxnode':>7} {'thin%':>7} "
              f"{'ll/bin':>7} {'thinLL%':>7}")
        print(f"{'g/cm2':>7} {'mGy':>9} {'mSv':>9} {'mSv':>9} {'mGy':>9} {'mSv':>9}")
        rows = []
        ww, wt = loglog_weights(ev, nodes), None
        for w in walls:
            pt = interp_point(K, w)
            f = fold(pt, pb)
            sub, sn = thinned(pt, nodes)
            ft = fold(sub, bin_fluence(ev, sn))
            sd, sN, sI, bd, bN = (f["skin"]["D"][0], f["skin"]["N"][0], f["skin"]["I"][0],
                                  f["deep"]["D"][0], f["deep"]["N"][0])
            rel = f["deep"]["N"][1] / bN if bN else float("nan")
            thin = 100.0 * (ft["deep"]["N"][0] / bN - 1.0) if bN else float("nan")
            wt = wt or loglog_weights(ev, sn)
            ll = fold_loglog(pt, ww)["deep"]["N"]
            llt = fold_loglog(sub, wt)["deep"]["N"]
            thin_ll = 100.0 * (llt / ll - 1.0) if ll else float("nan")
            row = {"wall": w, "skin_D_mGy": 1e3 * sd, "skin_N_mSv": 1e3 * sN,
                   "skin_I_mSv": 1e3 * sI, "bfo_D_mGy": 1e3 * bd, "bfo_N_mSv": 1e3 * bN,
                   "bfo_rel": rel, "bfo_maxnode": f["deep"]["N"][2], "thin_pct": thin,
                   "bfo_N_ll_mSv": 1e3 * ll, "thin_ll_pct": thin_ll}
            rows.append(row)
            print(f"{w:7g} {1e3*sd:9.4g} {1e3*sN:9.4g} {1e3*sI:9.4g} {1e3*bd:9.4g} "
                  f"{1e3*bN:9.4g} {(bN/bd if bd else 0):5.2f} {rel:6.3f} "
                  f"{row['bfo_maxnode']:7.2f} {thin:+7.1f} "
                  f"{(ll / bN if bN else float('nan')):7.3f} {thin_ll:+7.1f}")
        old = old_kernel_fold(ev)
        if grid[0] <= 147.5 <= grid[-1]:
            new = fold(interp_point(K, 147.5), pb)
            print("  vs shipped one-wall kernel @147.5 g/cm2 (report only, not a gate):")
            for o, lab in (("skin", "skin"), ("deep", "BFO ")):
                for q in ("D", "N"):
                    a, b = new[o][q][0], old[o][q][0]
                    r = f"{a/b:.3f}" if b else "n/a"
                    print(f"    {lab} {q}: new {1e3*a:.4g}  shipped {1e3*b:.4g}  "
                          f"new/shipped {r}  (isotropic-2pi {0.5*a/b if b else float('nan'):.3f})")
            cut = dosimetry._load_spe_kernel()["meta"]["nodes_mev"][0]
            dec = below_cut_share(interp_point(K, 147.5), ev, nodes, cut)
            print(f"  decomposition: protons below the shipped kernel's first node ({cut:g} MeV)")
            for o, lab in (("skin", "skin"), ("deep", "BFO ")):
                for q in ("D", "N"):
                    hi, share = dec[o][q]
                    b = old[o][q][0]
                    print(f"    {lab} {q}: share from <{cut:g} MeV {share:6.1%}   "
                          f"new(>={cut:g})/shipped {hi / b if b else float('nan'):.3f}")
        flags = [r["wall"] for r in rows if r["bfo_maxnode"] > 0.5 or abs(r["thin_pct"]) > 5]
        if flags:
            print(f"  CHECK (bin fold): one node >50% of BFO N, or thinning >5%, at {flags} g/cm2")
        flags = [r["wall"] for r in rows if abs(r["thin_ll_pct"]) > 5]
        print(f"  CHECK (log-log fold): thinning >5% at {flags} g/cm2"
              + (" -> node grid too coarse there" if flags else " -> none, node grid adequate"))
        print()
        out[ev] = rows
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pe = sub.add_parser("emit")
    pe.add_argument("--out", type=Path, required=True)
    pe.add_argument("--depths", type=float, nargs="+", default=DEPTHS_GCM2)
    pe.add_argument("--seeds", type=int, default=SEEDS)
    pe.add_argument("--hist0", type=int, default=HIST0)
    pe.add_argument("--threads", type=int, default=1)
    pc = sub.add_parser("collect")
    pc.add_argument("--out", type=Path, required=True)
    pc.add_argument("--dest", type=Path, default=DEST)
    pr = sub.add_parser("report")
    pr.add_argument("--kernel", type=Path, default=DEST)
    pr.add_argument("--events", nargs="+")
    pr.add_argument("--walls", type=float, nargs="+")
    a = ap.parse_args(argv)
    if a.cmd == "emit":
        emit(a.out, sorted(a.depths), NODES_MEV, a.seeds, a.threads, a.hist0)
    elif a.cmd == "collect":
        collect(a.out, a.dest)
    else:
        report(a.kernel, a.events, a.walls)
    return 0


if __name__ == "__main__":
    sys.exit(main())
