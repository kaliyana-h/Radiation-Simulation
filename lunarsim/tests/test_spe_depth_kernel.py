"""paper/spe_depth_kernel.py: SPE proton depth-grid kernel harness (no TOPAS)."""
import contextlib
import importlib.util
import io
import json
import math
import re
import tempfile
import unittest
from pathlib import Path

from lunarsim.kernel_gen import config, templates

_REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "spe_depth_kernel", _REPO / "paper" / "spe_depth_kernel.py")
sdk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sdk)

DEPTHS = [0.0, 20.0, 150.0]
NODES = sdk.NODES_MEV


def _fake_value(wall, node, organ, q, seed):
    """Scorer output such that R = v/phi_ff falls exp(-wall/50), +-1% seed scatter."""
    base = {"D": 1e-9, "I": 2e-9, "N": 3e-9}[q] * (1 + node) * math.exp(-wall / 50.0)
    return base * (1.0 + (0.01 if seed % 2 else -0.01)) * sdk.phi_ff(wall)


def _fake_outputs(out, man):
    for r in man["runs"]:
        d = out / "runs" / r["name"]
        for o, *_ in config.SHELLS:
            for q in sdk.QUANTS:
                v = _fake_value(r["wall_gcm2"], r["node"], o, q, r["seed"])
                (d / f"{sdk.csv_name(o, q)}.csv").write_text(f"# x\n{v!r}\n")


class Illumination(unittest.TestCase):
    def test_spot_covers_rim_and_hist_scales(self):
        # bare phantom: 1.35 x 15 cm = 20.25 -> next 5 cm step = 25 (> production 22)
        self.assertEqual(sdk.spot_cm(0.0), 25.0)
        self.assertEqual(sdk.histories(0.0), round(sdk.HIST0 * (25 / 22) ** 2))
        for d in sdk.DEPTHS_GCM2:
            s = sdk.spot_cm(d)
            self.assertGreaterEqual(s, sdk.SPOT_MARGIN * sdk.shell_outer_cm(d) - 1e-9)
            self.assertLess(math.hypot(config.BEAM_RADIUS_CM, s), sdk.WORLD_HALF_CM)
            # per-run fluence stays within ~25% of the production value
            ratio = sdk.phi_ff(d) / templates.phi_ff(config.BEAM_SPOT_CM, sdk.HIST0)
            self.assertTrue(0.75 < ratio < 1.25, (d, ratio))
        # 200 g/cm2 Al: R_out = 15 + 74.07 = 89.07 cm -> 1.35x = 120.2 -> 125
        self.assertEqual(sdk.spot_cm(200.0), 125.0)

    def test_param_file(self):
        t = sdk.build_param_file(150.0, 240.0, 3)
        self.assertIn("i:Ts/Seed            = 3", t)
        self.assertIn('s:Ge/Wall/Material = "G4_Al"', t)
        r = float(re.search(r"d:Ge/Wall/RMax\s*=\s*([\d.]+) cm", t).group(1))
        self.assertAlmostEqual(r, 15.0 + 150.0 / 2.70, places=3)
        self.assertIn(f"d:So/BeamSpot   = {sdk.spot_cm(150.0):.1f} cm", t)
        self.assertEqual(t.count("d:So/K0/BeamEnergy = 240 MeV"), 1)
        self.assertEqual(len(re.findall(r'BeamParticle = "proton"', t)),
                         config.RINGS * config.AZIMUTH)
        self.assertEqual(t.count(f"NumberOfHistoriesInRun   = {sdk.histories(150.0)}"),
                         config.RINGS * config.AZIMUTH)
        for o, *_ in config.SHELLS:
            for q, quant in sdk.SCORERS.items():
                self.assertIn(f's:Sc/Shell_{o}_{q}/Quantity   = "{quant}"', t)
                self.assertIn(f's:Sc/Shell_{o}_{q}/OutputFile = "{o}_{q}"', t)
        self.assertNotIn("Ge/Wall/", sdk.build_param_file(0.0, 10.0, 1))

    def test_ring_geometry_matches_production(self):
        """Same ring groups/directions as the GCR kernel for an identical wall."""
        ours = sdk.build_param_file(20.0, 100.0, 1)
        prod = templates.build_param_file("aluminium", 20.0, "H", 0, 1)
        grab = lambda s: re.findall(r"^[sd]:Ge/K(?:Azi|Aim|Src)\d+/.*$", s, re.M)
        self.assertEqual(grab(ours), grab(prod))
        wall = lambda s: re.findall(r"^[sd]:Ge/Wall/.*$", s, re.M)
        self.assertEqual(wall(ours), wall(prod))


class EmitCollect(unittest.TestCase):
    def test_emit_collect_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            out, dest = Path(d) / "k", Path(d) / "k.json"
            with contextlib.redirect_stdout(io.StringIO()):
                n = sdk.emit(out, DEPTHS, NODES, seeds=2)
            self.assertEqual(n, len(DEPTHS) * len(NODES) * 2)
            self.assertTrue((out / "run_all.sh").exists())
            man = json.loads((out / "manifest.json").read_text())
            with self.assertRaises(SystemExit):          # nothing has run yet
                sdk.collect(out, dest)
            _fake_outputs(out, man)
            with contextlib.redirect_stdout(io.StringIO()):
                K = sdk.collect(out, dest)
            self.assertTrue(dest.exists())
            self.assertEqual(K["meta"]["depths_gcm2"], DEPTHS)
            p = {pt["wall_gcm2"]: pt for pt in K["H"]["points"]}[20.0]
            want = 1e-9 * 4 * math.exp(-20.0 / 50.0)      # D, node 3, seed mean
            self.assertAlmostEqual(p["R"]["skin"]["D"][3] / want, 1.0, places=9)
            # two seeds at +-1%: sample SEM = 1% of mean
            self.assertAlmostEqual(p["Rsem"]["deep"]["N"][3] / p["R"]["deep"]["N"][3],
                                   0.01, places=9)
            # one missing scorer file -> refuse
            (out / "runs" / man["runs"][5]["name"] / "skin_N.csv").unlink()
            with self.assertRaises(SystemExit):
                sdk.collect(out, dest)


def _kernel(depths, Rfun):
    pts = []
    for w in depths:
        R = {o: {q: [Rfun(w, j) for j in range(len(NODES))] for q in sdk.QUANTS}
             for o, *_ in config.SHELLS}
        S = {o: {q: [0.01 * Rfun(w, j) for j in range(len(NODES))] for q in sdk.QUANTS}
             for o, *_ in config.SHELLS}
        pts.append({"wall_gcm2": w, "R": R, "Rsem": S, "seeds": 8})
    return {"meta": {"nodes_mev": NODES, "depths_gcm2": depths}, "H": {"points": pts}}


class Fold(unittest.TestCase):
    def test_constant_R_fold_and_thinning(self):
        K = _kernel([0.0, 10.0], lambda w, j: 2.0e-10)
        pb = sdk.bin_fluence("aug1972", NODES)
        self.assertEqual(len(pb), len(NODES))
        f = sdk.fold(sdk.interp_point(K, 0.0), pb, sky=0.5)
        self.assertAlmostEqual(f["skin"]["D"][0] / (0.5 * 2.0e-10 * sum(pb)), 1.0, places=12)
        sub, sn = sdk.thinned(sdk.interp_point(K, 0.0), NODES)
        self.assertEqual((sn[0], sn[-1]), (NODES[0], NODES[-1]))
        ft = sdk.fold(sub, sdk.bin_fluence("aug1972", sn), sky=0.5)
        # same total fluence over the same range -> thinning changes nothing
        self.assertAlmostEqual(ft["skin"]["D"][0] / f["skin"]["D"][0], 1.0, places=6)

    def test_binned_fluence_brackets_anchor(self):
        """Bins wholly above 30 MeV hold less than the >30 anchor; all bins
        (down to 10 MeV) hold more."""
        from lunarsim import dosimetry
        anchor = dosimetry._load_make_source().spe_integral_fluence("feb1956")
        pb = sdk.bin_fluence("feb1956", NODES)
        edges = sdk.bin_edges(NODES)
        above = sum(f for f, lo in zip(pb, edges) if lo >= 30.0)
        self.assertLess(above, anchor)
        self.assertGreater(above, 0.8 * anchor)
        self.assertGreater(sum(pb), anchor)

    def test_loglog_fold(self):
        pb = sdk.bin_fluence("oct1989", NODES)
        ww = sdk.loglog_weights("oct1989", NODES)
        # constant R: log-log and bin folds integrate the same 10 MeV-20 GeV fluence
        K = _kernel([0.0, 10.0], lambda w, j: 3.0e-10)
        pt = sdk.interp_point(K, 0.0)
        self.assertAlmostEqual(sdk.fold_loglog(pt, ww, sky=1.0)["deep"]["D"] /
                               sdk.fold(pt, pb, sky=1.0)["deep"]["D"][0], 1.0, places=4)
        # power-law R is exact under log-log: compare with a fine direct integral
        Kp = _kernel([0.0, 10.0], lambda w, j: 1e-12 * (NODES[j] / 100.0) ** 1.5)
        dnde, scale = sdk._spectrum("oct1989")
        n, a, b = 200000, NODES[0], NODES[-1]
        g = [a * (b / a) ** (i / n) for i in range(n + 1)]
        ref = sum(scale * dnde(math.sqrt(x * y)) * (y - x) * 1e-12 *
                  (math.sqrt(x * y) / 100.0) ** 1.5 for x, y in zip(g[:-1], g[1:]))
        got = sdk.fold_loglog(sdk.interp_point(Kp, 0.0), ww, sky=1.0)["skin"]["N"]
        self.assertAlmostEqual(got / ref, 1.0, places=3)

    def test_below_cut_share(self):
        K = _kernel([0.0, 10.0], lambda w, j: 1e-11)
        pt = sdk.interp_point(K, 0.0)
        d0 = sdk.below_cut_share(pt, "aug1972", NODES, NODES[0])
        self.assertAlmostEqual(d0["skin"]["D"][1], 0.0, places=9)
        d = sdk.below_cut_share(pt, "aug1972", NODES, 430.0)
        self.assertTrue(0.9 < d["skin"]["D"][1] < 1.0)    # soft event: most fluence < 430

    def test_interp_log_midpoint_and_bounds(self):
        K = _kernel([10.0, 30.0], lambda w, j: math.exp(-w / 10.0))
        mid = sdk.interp_point(K, 20.0)["R"]["deep"]["D"][0]
        self.assertAlmostEqual(mid, math.sqrt(math.exp(-1) * math.exp(-3)), places=12)
        K0 = _kernel([10.0, 30.0], lambda w, j: 0.0 if w > 20 else 1.0)
        self.assertAlmostEqual(sdk.interp_point(K0, 20.0)["R"]["skin"]["N"][0], 0.5)
        with self.assertRaises(ValueError):
            sdk.interp_point(K, 40.0)

    def test_report_runs_and_compares_to_shipped(self):
        K = _kernel([0.0, 100.0, 200.0], lambda w, j: 1e-11 * (1 + j) * math.exp(-w / 80))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "k.json"
            p.write_text(json.dumps(K))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rows = sdk.report(p, events=["aug1972"], walls=[0.0, 147.5])
            self.assertIn("vs shipped one-wall kernel", buf.getvalue())
            self.assertIn("decomposition", buf.getvalue())
            self.assertIn("CHECK (log-log fold)", buf.getvalue())
            self.assertEqual([r["wall"] for r in rows["aug1972"]], [0.0, 147.5])
            self.assertGreater(rows["aug1972"][0]["bfo_N_mSv"],
                               rows["aug1972"][1]["bfo_N_mSv"])


def _step_kernel(E_thr, sem_rel=0.01):
    """R(E) jumps x100 between two nodes at E_thr; power law elsewhere."""
    return _kernel([0.0, 10.0], lambda w, j: 1e-12 * (NODES[j] / 100.0) ** 0.5 *
                   (100.0 if NODES[j] > E_thr else 1.0))


class TopUp(unittest.TestCase):
    EVENTS = ["oct1989"]

    def test_split_interval(self):
        self.assertEqual(sdk.split_interval(100.0, 120.0, 4), [104.7, 109.5, 114.7])
        self.assertEqual(sdk.split_interval(10.0, 1000.0, 2), [100.0])

    def test_plan_flags_step_not_power_law(self):
        smooth = _kernel([0.0, 10.0], lambda w, j: 1e-12 * (NODES[j] / 100.0) ** 1.5)
        self.assertEqual(sdk.topup_plan(smooth, events=self.EVENTS), {})
        plan = sdk.topup_plan(_step_kernel(150.0), events=self.EVENTS, seeds=3)
        self.assertEqual(sorted(plan), [0.0, 10.0])
        new = [e for e, first, n in plan[0.0]]
        self.assertTrue(all((first, n) == (1, 3) for _, first, n in plan[0.0]))
        # the step sits in 140-170; its pair partner interval is refined too
        self.assertTrue(any(140 < e < 170 for e in new), new)
        self.assertTrue(all(100 <= e <= 240 for e in new), new)
        self.assertFalse(set(new) & set(NODES))
        only = sdk.topup_plan(_step_kernel(150.0), events=self.EVENTS, seeds=3,
                              depths=[10])
        self.assertEqual(only, {10.0: plan[10.0]})

    def test_noise_only_difference_is_not_refined(self):
        K = _step_kernel(150.0)
        for p in K["H"]["points"]:                     # 200% node noise
            for o in p["Rsem"]:
                for q in p["Rsem"][o]:
                    p["Rsem"][o][q] = [2.0 * r for r in p["R"][o][q]]
        self.assertEqual(sdk.topup_plan(K, events=self.EVENTS), {})

    def test_reseed_numbers_after_base(self):
        smooth = _kernel([0.0, 10.0], lambda w, j: 1e-12 * (NODES[j] / 100.0) ** 1.5)
        plan = sdk.topup_plan(smooth, events=self.EVENTS, reseed_depths=[10.0],
                              reseed_band=(100.0, 140.0), reseed_seeds=4)
        self.assertEqual(plan, {10.0: [(100.0, 9, 4), (120.0, 9, 4), (140.0, 9, 4)]})

    def _base(self, d):
        out, dest = Path(d) / "k", Path(d) / "k.json"
        with contextlib.redirect_stdout(io.StringIO()):
            sdk.emit(out, DEPTHS, NODES, seeds=2)
            man = json.loads((out / "manifest.json").read_text())
            _fake_outputs(out, man)
            sdk.collect(out, dest)
        return dest

    @staticmethod
    def _fake_topup(out, man, f):
        for r in man["runs"]:
            d = out / "runs" / r["name"]
            for o, *_ in config.SHELLS:
                for q in sdk.QUANTS:
                    v = f(r["wall_gcm2"], r["energy_mev"], q) * \
                        (1.0 + (0.02 if r["seed"] % 2 else -0.02)) * sdk.phi_ff(r["wall_gcm2"])
                    (d / f"{sdk.csv_name(o, q)}.csv").write_text(f"# x\n{v!r}\n")

    def test_emit_merge_pool_fill_and_report(self):
        import statistics
        f = lambda w, e, q: 5e-9 * math.exp(-w / 50.0)
        with tempfile.TemporaryDirectory() as d:
            base = self._base(d)
            # 20 g/cm2: one new node (130 MeV) + 2 extra seeds on node 3 (25 MeV)
            plan = {20.0: [(25.0, 3, 2), (130.0, 1, 2)]}
            tu, dest = Path(d) / "tu", Path(d) / "merged.json"
            n = sdk.topup_emit(tu, base, plan)
            self.assertEqual(n, 4)
            man = json.loads((tu / "manifest.json").read_text())
            self.assertEqual(man["kind"], "topup")
            self.assertTrue((tu / "runs" / "spe_aluminium_w20_e130_s2" / "run.txt").exists())
            self.assertIn("d:So/K0/BeamEnergy = 130 MeV",
                          (tu / "runs" / "spe_aluminium_w20_e130_s1" / "run.txt").read_text())
            with self.assertRaises(SystemExit):        # nothing has run yet
                sdk.topup_merge(base, tu, dest)
            self._fake_topup(tu, man, f)
            with contextlib.redirect_stdout(io.StringIO()):
                K = sdk.topup_merge(base, tu, dest)
            nodes = K["meta"]["nodes_mev"]
            self.assertEqual(len(nodes), len(NODES) + 1)
            i130, i25 = nodes.index(130.0), nodes.index(25.0)
            P = {p["wall_gcm2"]: p for p in K["H"]["points"]}
            # pooled 25 MeV cell = all four seed samples
            phi = sdk.phi_ff(20.0)
            vals = [_fake_value(20.0, 3, "deep", "N", s) / phi for s in (1, 2)] + \
                   [f(20.0, 25.0, "N") * (1 + x) for x in (0.02, -0.02)]
            self.assertAlmostEqual(P[20.0]["R"]["deep"]["N"][i25] / statistics.fmean(vals),
                                   1.0, places=9)
            self.assertAlmostEqual(P[20.0]["Rsem"]["deep"]["N"][i25] /
                                   (statistics.stdev(vals) / 2.0), 1.0, places=9)
            self.assertEqual(P[20.0]["node_seeds"][i25], 4)
            self.assertEqual(P[20.0]["seed_ranges"][i25], [[1, 2], [3, 4]])
            self.assertAlmostEqual(P[20.0]["R"]["skin"]["D"][i130] /
                                   f(20.0, 130.0, "D"), 1.0, places=9)
            self.assertEqual(P[20.0]["filled_mev"], [])
            # other depths: 130 MeV filled log-log between their 120 and 140 nodes
            self.assertEqual(P[0.0]["filled_mev"], [130.0])
            lo, hi = P[0.0]["R"]["mid"]["D"][i130 - 1], P[0.0]["R"]["mid"]["D"][i130 + 1]
            t = math.log(130 / 120) / math.log(140 / 120)
            self.assertAlmostEqual(P[0.0]["R"]["mid"]["D"][i130],
                                   lo ** (1 - t) * hi ** t, places=20)
            self.assertEqual(P[0.0]["node_seeds"][i130], 0)
            self.assertEqual(len(K["meta"]["topups"]), 1)
            # checks/plans use each depth's MEASURED nodes: 130 is filled at 0 g/cm2
            own, on = sdk.measured(P[0.0], nodes)
            self.assertNotIn(130.0, on)
            self.assertEqual(len(on), len(NODES))
            self.assertIn(130.0, sdk.measured(P[20.0], nodes)[1])
            self.assertEqual(sdk.interp_point(K, 10.0)["filled"], [130.0])
            plan2 = sdk.topup_plan(K, events=["feb1956"], min_share=10.0,
                                   reseed_depths=[0.0, 20.0], reseed_band=(120.0, 140.0),
                                   reseed_seeds=2)
            self.assertEqual(plan2[0.0], [(120.0, 3, 2), (140.0, 3, 2)])
            self.assertEqual(plan2[20.0], [(120.0, 3, 2), (130.0, 3, 2), (140.0, 3, 2)])
            # the same seeds again -> overlap refused
            with self.assertRaises(SystemExit):
                sdk.topup_merge(dest, tu, Path(d) / "again.json")
            # report runs on the merged grid
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                rows = sdk.report(dest, events=["feb1956"], walls=[20.0])
            self.assertIn("32 nodes", buf.getvalue())
            self.assertEqual([r["wall"] for r in rows["feb1956"]], [20.0])

    def test_merge_refuses_mismatched_batch(self):
        with tempfile.TemporaryDirectory() as d:
            base = self._base(d)
            tu = Path(d) / "tu"
            sdk.topup_emit(tu, base, {20.0: [(130.0, 1, 2)]})
            man = json.loads((tu / "manifest.json").read_text())
            man["hist0"] += 1
            (tu / "manifest.json").write_text(json.dumps(man))
            with self.assertRaises(SystemExit):
                sdk.topup_merge(base, tu, Path(d) / "m.json")

    def test_cli_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "k.json"
            p.write_text(json.dumps(_step_kernel(150.0)))
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                sdk.main(["topup", "--base", str(p), "--dry-run",
                          "--out", str(Path(d) / "tu")])
            self.assertIn("total", buf.getvalue())
            self.assertFalse((Path(d) / "tu").exists())


if __name__ == "__main__":
    unittest.main()
