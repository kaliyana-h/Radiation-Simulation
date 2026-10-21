"""Regression guard for the per-element high-energy TAIL harness
(kernel_gen/config.py + kernel_gen/tail.py; memory energy-ceiling-tail-measured).

The #3 energy-ceiling run extends the shipped 10-ion per-element kernel's
truncated 6000/5000 MeV/n tops to the free-field ceiling (GCR_EMAX_PER_NUC =
20000 MeV/n) via a diagnostic-only `perelement_tail` composition. Two classes of
thing must not silently regress:

  * CONFIG invariants -- species_perelement_tail must carry exactly the 10
    per-element ions, with append nodes that sit strictly ABOVE each ion's base
    top and land ON the ceiling, and metadata byte-identical to the base table
    (abundance/particle/group) so the merge keeps the base kernel's physics. The
    `perelement_tail` composition must be registered and dispatched.

  * MERGE guards -- kernel_gen.tail.merge appends a tail batch onto a base kernel.
    It guards three ways against a corrupted ship: the manifest must name a known
    tail composition, the --base kernel's composition must be the one that tail
    extends, and the base/tail species SETS must match. A tail node that overlaps
    (does not extend) the base grid is also rejected. These guards are exactly the
    footguns the harness added, so they get a pinned test.

All assertions are pure arithmetic / fabricated fixtures -- no TOPAS, no MC,
sub-second. The merge tests monkeypatch generate._collect_point so no run outputs
are needed on disk.

TARGET-BLIND: no dose value, limit, or external reference appears here.

    cd ~/topas && python3 -m unittest lunarsim.tests.test_tail_harness -v
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lunarsim.kernel_gen import config, generate, tail


# --------------------------------------------------------------------------
# CONFIG invariants
# --------------------------------------------------------------------------
class TestPerElementTailConfig(unittest.TestCase):

    def test_perelement_tail_registered_and_dispatched(self):
        self.assertIn("perelement_tail", config.COMPOSITIONS)
        self.assertIn("perelement_tail", config.TAIL_COMPOSITIONS)
        # perelement_tail may only be merged onto the perelement base kernel.
        self.assertEqual(config.TAIL_COMPOSITIONS["perelement_tail"], ("perelement",))
        # species_for dispatches it to the tail table (not a ValueError).
        self.assertEqual(
            set(config.species_for("perelement_tail")),
            set(config.species_perelement_tail()))

    def test_tail_compositions_map_shape(self):
        # Every tail composition's declared base(s) must be real compositions, and
        # a tail composition is never its own base.
        for comp, bases in config.TAIL_COMPOSITIONS.items():
            self.assertIn(comp, config.COMPOSITIONS)
            self.assertIsInstance(bases, tuple)
            for b in bases:
                self.assertIn(b, config.COMPOSITIONS)
                self.assertNotIn(b, config.TAIL_COMPOSITIONS,
                                 f"{comp} base {b} must be a non-tail base kernel")

    def test_species_perelement_tail_matches_base_table(self):
        base = config.species_perelement()
        tailt = config.species_perelement_tail()
        # Same 10 ions, same order.
        self.assertEqual(list(tailt), list(base))
        self.assertEqual(len(tailt), len(config.PERELEMENT_IONS))
        for name, ts in tailt.items():
            bs = base[name]
            # Identity metadata is carried through unchanged (merge keeps base
            # abundance; a drift here would mislabel the diagnostic batch).
            for key in ("z", "a", "abundance", "particle", "group"):
                self.assertEqual(ts[key], bs[key], f"{name}.{key} drifted from base")

    def test_tail_nodes_extend_and_pin_to_ceiling(self):
        base = config.species_perelement()
        ceil = config.TAIL_CEILING_PERNUC_MEV
        for name, ts in config.species_perelement_tail().items():
            tnodes = ts["nodes_pernuc_mev"]
            base_top = base[name]["nodes_pernuc_mev"][-1]
            self.assertTrue(tnodes, f"{name} has no tail nodes")
            # Strictly above the base top -> satisfies the merge overlap guard.
            self.assertGreater(tnodes[0], base_top,
                               f"{name} first tail node must extend past base top")
            # Monotone increasing.
            self.assertEqual(tnodes, sorted(tnodes), f"{name} tail nodes not sorted")
            # Top lands ON the free-field ceiling so the fold drops no residual flux.
            self.assertAlmostEqual(tnodes[-1], ceil, places=6,
                                   msg=f"{name} top node != ceiling")

    def test_ceiling_equals_make_source_free_field(self):
        # The whole harness is pointless if the tail ceiling has drifted from the
        # spectrum's emission ceiling.
        import make_source as ms
        self.assertAlmostEqual(
            config.TAIL_CEILING_PERNUC_MEV, float(ms.GCR_EMAX_PER_NUC), places=3)

    def test_check_ceiling_accepts_tail_compositions(self):
        # Both registered tail compositions pass the launch sanity check.
        for comp in config.TAIL_COMPOSITIONS:
            tail._check_ceiling(comp)  # must not raise

    def test_check_ceiling_rejects_unknown_composition(self):
        with self.assertRaises(SystemExit):
            tail._check_ceiling("perelement")       # a real comp, but not a tail one
        with self.assertRaises(SystemExit):
            tail._check_ceiling("not_a_composition")

    def test_run_counts_match_harness_plan(self):
        # Documented invariant (memory energy-ceiling-tail-measured): the
        # per-element tail is 10 ions x 2 nodes x 4 anchors x 4 seeds = 320 runs;
        # the default 5-ion tail is 160. Compute from the tables so a grid/seed
        # change updates the expectation, but pin the headline numbers too.
        def expected(comp, seeds):
            sp = config.species_for(comp)
            node_species = sum(len(s["nodes_pernuc_mev"]) for s in sp.values())
            grid = generate._grid_for("aluminium")
            return node_species * len(grid) * seeds

        pe = len(list(generate.enumerate_runs("aluminium", 4, "perelement_tail")))
        base5 = len(list(generate.enumerate_runs("aluminium", 4, "tail")))
        self.assertEqual(pe, expected("perelement_tail", 4))
        self.assertEqual(base5, expected("tail", 4))
        self.assertEqual(pe, 320)
        self.assertEqual(base5, 160)


# --------------------------------------------------------------------------
# MERGE guards (fabricated kernel + tail batch; no run outputs on disk)
# --------------------------------------------------------------------------
def _organs():
    return [name for name, _ in config.organs_from_reference()]


def _r_block(nnodes):
    return {o: {"D": [0.0] * nnodes, "I": [0.0] * nnodes} for o in _organs()}


def _species_block(nodes):
    nodes = list(nodes)
    return {"nodes_pernuc_mev": nodes,
            "R": _r_block(len(nodes)),
            "Rsem": _r_block(len(nodes))}


def _base_kernel(composition, species_nodes, walls):
    """A minimal but merge-faithful base kernel JSON dict."""
    return {
        "meta": {
            "composition": composition,
            "organs": [[n, w] for n, w in config.organs_from_reference()],
            "note": "fixture base kernel",
        },
        "points": [
            {"wall_gcm2": w,
             "species": {n: _species_block(nodes) for n, nodes in species_nodes.items()}}
            for w in walls
        ],
    }


def _tail_point(wall, species_tail_nodes):
    return {"wall_gcm2": wall,
            "species": {n: _species_block(nodes)
                        for n, nodes in species_tail_nodes.items()}}


class TestTailMerge(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="tailmerge_"))
        self.out = self.tmp / "out"
        (self.out / "runs").mkdir(parents=True)
        self.walls = [2.025, 10.0, 50.0, 0.0]
        # Faithful per-element base + tail node tables from config.
        self.base_nodes = {n: s["nodes_pernuc_mev"]
                           for n, s in config.species_perelement().items()}
        self.tail_nodes = {n: s["nodes_pernuc_mev"]
                           for n, s in config.species_perelement_tail().items()}

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_manifest(self, composition):
        (self.out / "manifest.json").write_text(json.dumps({
            "material": "aluminium", "seeds": 4,
            "grid_gcm2": self.walls, "composition": composition,
        }))

    def _write_base(self, composition, species_nodes):
        p = self.tmp / "base.json"
        p.write_text(json.dumps(_base_kernel(composition, species_nodes, self.walls)))
        return p

    def _patch_collect(self, tail_nodes):
        def fake(runs_dir, material, wall, seeds, composition):
            return _tail_point(wall, tail_nodes)
        return mock.patch.object(generate, "_collect_point", fake)

    def test_merge_appends_tail_nodes_well_formed(self):
        self._write_manifest("perelement_tail")
        base = self._write_base("perelement", self.base_nodes)
        dest = self.tmp / "dest.json"
        with self._patch_collect(self.tail_nodes):
            tail.merge(self.out, base, dest)
        self.assertTrue(dest.exists())
        K = json.loads(dest.read_text())
        # Every species' node grid grew by exactly its tail-node count, with the
        # tail nodes appended after the base ones.
        for p in K["points"]:
            for name, bs in p["species"].items():
                tnodes = self.tail_nodes[name]
                self.assertEqual(bs["nodes_pernuc_mev"][-len(tnodes):], tnodes)
                self.assertEqual(len(bs["nodes_pernuc_mev"]),
                                 len(self.base_nodes[name]) + len(tnodes))
                # R/Rsem arrays grew in lockstep for every organ.
                for o in _organs():
                    for q in ("D", "I"):
                        self.assertEqual(len(bs["R"][o][q]),
                                         len(bs["nodes_pernuc_mev"]))
        meta = K["meta"]["tail_extension"]
        self.assertEqual(meta["composition"], "perelement_tail")
        self.assertEqual(set(meta["nodes_pernuc_mev"]), set(self.base_nodes))
        self.assertEqual(meta["ceiling_pernuc_mev"], config.TAIL_CEILING_PERNUC_MEV)

    def test_merge_rejects_non_tail_manifest_composition(self):
        self._write_manifest("perelement")          # a base comp, not a tail comp
        base = self._write_base("perelement", self.base_nodes)
        with self._patch_collect(self.tail_nodes):
            with self.assertRaises(SystemExit):
                tail.merge(self.out, base, self.tmp / "dest.json")

    def test_merge_rejects_wrong_base_composition(self):
        # Manifest is perelement_tail, but the --base kernel is the 5-ion corrected
        # one -- merging would silently pair only the overlapping species.
        self._write_manifest("perelement_tail")
        base = self._write_base("corrected", self.base_nodes)
        with self._patch_collect(self.tail_nodes):
            with self.assertRaises(SystemExit):
                tail.merge(self.out, base, self.tmp / "dest.json")

    def test_merge_rejects_species_set_mismatch(self):
        self._write_manifest("perelement_tail")
        base = self._write_base("perelement", self.base_nodes)
        dropped = {n: v for n, v in self.tail_nodes.items() if n != "Fe"}
        with self._patch_collect(dropped):
            with self.assertRaises(SystemExit):
                tail.merge(self.out, base, self.tmp / "dest.json")

    def test_merge_rejects_overlapping_tail_node(self):
        # Species set matches, but one ion's first tail node sits at/below the base
        # top -> the fold grid would overlap instead of extend.
        self._write_manifest("perelement_tail")
        base = self._write_base("perelement", self.base_nodes)
        bad = dict(self.tail_nodes)
        bad["Fe"] = [100.0, config.TAIL_CEILING_PERNUC_MEV]   # 100 < Fe base top 5000
        with self._patch_collect(bad):
            with self.assertRaises(SystemExit):
                tail.merge(self.out, base, self.tmp / "dest.json")


# --------------------------------------------------------------------------
# Cross-check: the SHIPPED per-element kernel matches the config it was built from
# --------------------------------------------------------------------------
class TestShippedPerElementKernel(unittest.TestCase):

    def test_shipped_kernel_species_match_config(self):
        path = _REPO_ROOT / "lunarsim" / "data" / config.PERELEM_OUT
        if not path.exists():
            self.skipTest(f"{config.PERELEM_OUT} not present")
        K = json.loads(path.read_text())
        self.assertEqual(K["meta"].get("composition"), "perelement")
        self.assertEqual(set(K["points"][0]["species"]),
                         set(config.species_perelement()))


if __name__ == "__main__":
    unittest.main()
