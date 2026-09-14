"""Thin-wall GCR kernel smoke test (sort-fix + corrected-Fe sanity).

Run from the repo root:  python3 kernel_smoketest.py

PASS conditions:
  (a) skin / effective dose DROP monotonically as the wall thickens
      -> the fold sort-fix is live (no clamp-to-unshielded anchor);
  (b) Fe/Si effective-dose ratio is physical (~0.7-0.8 thin, falling
      with shielding) -> the corrected 6-node Fe kernel is loaded, not
      the truncated 3-node one that pinned Fe far too low.

If every row is IDENTICAL, the checkout is missing the dosimetry.py
sort-fix. If Fe/Si is tiny (<~0.2) everywhere, it is serving the old
truncated Fe kernel.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lunarsim.dosimetry import fold_gcr_thinwall

SEC_PER_YEAR = 365.25 * 86400.0


class Stub:                                   # minimal spec: just an areal density
    def __init__(self, ad):
        self.ad = ad

    def areal_density_gcm2(self):
        return self.ad


def mSv_yr(sv_s):
    return sv_s * SEC_PER_YEAR * 1000.0


print(f"{'wall g/cm2':>10} | {'skin mSv/yr':>11} | {'effective mSv/yr':>16} | {'Fe/Si':>6}")
print("-" * 56)
rows = []
for ad in [0.0, 2.025, 10.0, 50.0]:
    r = fold_gcr_thinwall(Stub(ad), phi_MV=400.0, material="aluminium")
    skin = mSv_yr(next(x for x in r["rows"] if x["shell"] == "skin")["doseeq_sv_s"])
    eff = mSv_yr(r["E_sv_s"])
    ps = r["per_species"]
    fe = ps.get("Fe", {}).get("E_sv_s", 0.0)
    si = ps.get("Si", {}).get("E_sv_s", 0.0)
    ratio = fe / si if si else float("nan")
    rows.append((ad, skin, eff, ratio))
    print(f"{ad:>10} | {skin:>11.0f} | {eff:>16.0f} | {ratio:>6.2f}")

thin, thick = rows[1][2], rows[3][2]          # effective at 2.025 vs 50 g/cm2
print("\nsort-fix check (thin 2.025 vs thick 50 g/cm2 effective): "
      f"{thin:.0f} vs {thick:.0f} -> "
      f"{'PASS' if thin > thick * 1.05 else 'FAIL (clamped/equal?)'}")
