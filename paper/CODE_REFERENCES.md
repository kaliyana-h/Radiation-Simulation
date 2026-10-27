# Literature-reference audit of the lunar radiation tool

Every hardcoded number, equation, and model choice in the code that comes from
(or should be attributed to) a literature source, with where it lives in the
code and the citation to put next to it in the thesis / paper.

**Status column:**
- **cited** — the reference is already written into the source-file comments.
- **add** — the value is correct but the code does not name its source; supply
  the citation below in the paper.
- **choice** — an engineering/compute decision, not a literature value (still
  worth stating, but it is *ours*, not a citation).

---

## 1. GCR energy spectrum (the primary source)

| Quantity | Value / form in code | Location | Source | Status |
|---|---|---|---|---|
| Proton local interstellar spectrum (LIS) | `J = 1.9e4·T^−2.78 / (1 + 0.4866·T^−2.51)` [/(m²·s·sr·GeV)] | `make_source.py:123` `lis_proton`; mirrored in `dosimetry.py` | **Usoskin et al. (2005)**, *J. Geophys. Res.* 110, A12108, doi:10.1029/2005JA011250 | cited |
| Force-field modulation | `Φ = (Z/A)·φ`; `J(T) = J_LIS(T+Φ)·T(T+2E₀) / [(T+Φ)(T+Φ+2E₀)]` | `make_source.py:151`, `gcr_spectrum` | **Gleeson & Axford (1968)**, *ApJ* 154, 1011 (force-field approximation); modulation-potential reconstruction from Usoskin et al. (2005) above | cited (form) / **add** (Gleeson & Axford) |
| Modulation potential φ = 400 MV | project worst-case (deep 2019/2020 solar minimum) | `make_source.py:88,428`; `dosimetry.py GCR_CALIB_PHI_MV` | φ time series: **Usoskin et al. (2005/2011)** reconstruction (deep min ≈ 380–400 MV) | cited |
| Rigidity-matched heavy-ion shape (`"rigidity"` option) | `R = (A/Z)·√(T(T+2E₀))`; universality in rigidity | `make_source.py:128` | **ISO 15390:2004**; **Matthiä et al. (2013)**, *Adv. Space Res.* 51, 329 | cited |
| Proton rest energy E₀ = 938.272 MeV | constant | `make_source.py:47`; `dosimetry.py E0_PROTON` | CODATA / PDG | add (standard constant) |
| GCR elemental composition & abundances | H 1.0, He 0.105, C 0.033, Si 0.012, Fe 0.0020 (`dosimetry.py`); H/He/C/O/Fe in the kernel JSON | `dosimetry.py GCR_COMPOSITION`; `gcr_thinwall_kernel.json` | **Simpson (1983)**, *Ann. Rev. Nucl. Part. Sci.* 33, 323; or **George et al. (2009)**, *ApJ* 698, 1666 (ACE/CRIS solar-min abundances) | **add** ⚠ see note |
| Per-nucleon energy ceiling 20 GeV/nuc | `GCR_EMAX_PER_NUC = 2.0e4` | `make_source.py:88` | compute compromise (documented ~5.6% dose under-report vs 100 GeV/nuc) | **choice** |
| Universal-in-energy/nucleon ansatz (`"pernuc"` default) | heavy ions borrow the proton LIS shape | `make_source.py:117` | standard first-order GCR approximation; the `"rigidity"` alternative (above) is the physically-preferred refinement | choice / add |

⚠ **Composition note for the meeting:** two things to flag honestly. (i) The
abundance ratios are the one set of GCR numbers with no citation in the code —
pin them to Simpson (1983) or George et al. (2009). (ii) `dosimetry.py` uses
**H/He/C/Si/Fe** while the shipped kernel JSON uses **H/He/C/O/Fe** (O replaces
Si). Both are defensible 5-species reductions, but they differ; decide on one
and state it, or note the two engines use different lumping and why.

---

## 2. Solar particle event (SPE) source

| Quantity | Value / form in code | Location | Source | Status |
|---|---|---|---|---|
| SPE spectral shape | exponential in magnetic rigidity, `dN/dR ∝ exp(−R/R₀)`, converted to `dN/dE ∝ exp(−R/R₀)·(E+mₚ)/R` | `make_source.py:170–234`, `spe_spectrum` | **King (1974)**, *J. Spacecraft & Rockets* 11, 401 | cited |
| Aug 1972 design event | R₀ = 100 MV, φ(>30 MeV) = 5.0×10⁹ cm⁻² | `make_source.py:195` `SPE_EVENTS` | King (1974); canonical crewed worst case | cited |
| Oct 1989 (GLE series) | R₀ = 150 MV, φ(>30 MeV) = 1.5×10⁹ cm⁻² | `make_source.py:197` | **Tylka & Dietrich (2009)** proton-spectra fits (31st ICRC); **Xapsos et al.** ESP model, *IEEE Trans. Nucl. Sci.* | cited (names) / **add** (full refs) |
| Feb 1956 (GLE 5) | R₀ = 220 MV, φ(>30 MeV) = 1.0×10⁹ cm⁻² | `make_source.py:199` | Tylka & Dietrich (2009); hardest modern event | cited / add |
| SPE integral-fluence anchor (>30 MeV) | measured φ(>30 MeV) per event | `make_source.py:239` | standard shielding-design integral anchor; per-event values from the fits above | cited |

**Honest caveat already in code** (`make_source.py:186`): a single
exponential-in-rigidity slightly *under*-predicts the hardest tail of real events
(which harden to a double power-law / Band function). The three events bracket
this. Good to say out loud — the supervisors will ask.

---

## 3. Quality factors / dose-equivalent conversion

| Quantity | Value / form in code | Location | Source | Status |
|---|---|---|---|---|
| ICRP Q(L) | `Q=1 (L<10); 0.32L−2.2 (10≤L≤100); 300/√L (L>100)`, L = unrestricted LET in keV/µm | `DoseEquivalentICRP.cc` | **ICRP Publication 60 (1991)**, *1990 Recommendations*, §Q(L) | cited |
| NASA solid-cancer QF | `Q=(1−P)+(X/L)·P`; `P=[1−e^(−(Z*²/β²)/κ)]^m·[1−e^(−E/0.2)]`; X=7000 keV/µm, κ=1000 (Z≤4)/500 (Z>4), m=3; Barkas `Z*=Z[1−e^(−125βZ^(−2/3))]` | `DoseEquivalentNASA.cc` | **NASA/TP-2013-217375**, Cucinotta, Kim & Chappell (2013), Eqs. 25–26 | cited |
| Leukaemia QF variant | X = 1750 keV/µm | `DoseEquivalentNASA.cc` | same, TP-2013-217375 | cited |
| Atomic mass unit u = 931.494 MeV | constant in NASA QF | `DoseEquivalentNASA.cc` | CODATA | add (standard constant) |
| Default scalar GCR quality factor 3.5 | `DEFAULT_QUALITY_FACTOR = 3.5` (stated as literature mean, range 2–6) | `dosimetry.py` | GCR mean-Q at solar min: e.g. **Badhwar & O'Neill**, or ISS/CRaTER measured ⟨Q⟩; NCRP-142 | **add** ⚠ |
| Default SPE quality factor 1.5 | `DEFAULT_SPE_QUALITY_FACTOR = 1.5` | `dosimetry.py` | proton ⟨Q⟩; **ICRP 60** radiation weighting for protons | **add** |

⚠ The scalar 3.5 is a *fallback* mean; the real numbers now come from the
per-step ICRP-60 and NASA scorers (transport-resolved LET → Q). Worth clarifying
in the meeting that the scalar is legacy/fallback, not the shipped physics.

---

## 4. Regolith composition & density (the shielding geometry)

| Quantity | Value in code | Location | Source | Status |
|---|---|---|---|---|
| Apollo-17 high-Ti mare oxide assay | SiO₂ 39.9, TiO₂ 9.4, Al₂O₃ 11.3, FeO 17.5, MgO 9.6, CaO 10.7, Na₂O 0.35 wt% | `lunar_environment.txt` | Taurus-Littrow soil: **Lunar Sourcebook** (Heiken, Vaniman & French 1991), Ch. 7; or the specific Apollo 17 soil analysis you took it from | **add** ⚠ pin the exact sample/table |
| 8-element mass fractions | O 0.4184 … Fe 0.1374 | `lunar_environment.txt` | derived from the oxide assay above (stoichiometry) | derived |
| Depth density profile | 4 REDMoon layers → 3 slabs, ρ = 1.76 / 2.11 / 1.78 g/cm³ | `lunar_environment.txt` | **Dobynde & Guo (2021)**, *JGR Planets*, doi:10.1029/2021JE006930 | cited |
| EVA-suit & LCVG compositions | 7-element ρ 0.252; 4-element ρ 0.5133 g/cm³ | `lunar_environment.txt`; `kernel_gen/config.py` | EVA-laminate / LCVG material spec (state the source datasheet/paper) | **add** |

⚠ The oxide assay is a real Apollo-17 measurement but the code doesn't name the
sample. Cite the exact soil (e.g. a specific 7xxxx sample or the mare-soil
average from the Lunar Sourcebook) so it's reproducible.

---

## 5. Physics list & transport

| Quantity | Value | Location | Source | Status |
|---|---|---|---|---|
| Reference physics list | FTFP_BERT_HP (Bertini <10 GeV, Fritiof above, HP neutrons <20 MeV) | habitat `.txt` files | **Geant4** reference physics lists; FTF: Fritiof model; BERT: **Bertini et al.** cascade | add (name the Geant4 version) |
| High-precision neutron data | G4NDL (<20 MeV) | physics list | **G4NDL / ENDF/B** evaluated neutron libraries | add |
| EM model | G4EmStandardPhysics, default 0.05 mm range cut | habitat `.txt` | Geant4 EM standard | add |

State the **Geant4 version** and the exact data-library versions (G4NDL x.y) —
reviewers of an NIMB paper will expect this in §2.

---

## 6. Calibration & validation anchors (external measurements)

| Quantity | Value | Location | Source | Status |
|---|---|---|---|---|
| Surface absorbed-dose anchor | 13.2 µGy/h at ~22 g/cm² → CAL = 1.14867 | `dosimetry.py`, calibration note | **Zhang et al. (2020)**, "First measurements of the radiation dose on the lunar surface", *Science Advances* 6, eaaz1334 (Chang'E-4 LND) | cited |
| Outer-gauge anchor | `OUTER_GAUGE_ANCHOR_CAL = 0.94932` (0.25620·(1.7933e-7/3.9997e-8)·(800/880)²) | `dosimetry.py` | internal geometric re-anchor (not literature) | choice |
| OLTARIS cross-check | 214.0 mSv/yr effective (phantom); 202.5 point-H | `paper/oltaris_points.csv` | **OLTARIS** (NASA LaRC), BON-2020 GCR = **Badhwar–O'Neill 2020** (Slaba et al.); ICRP-60 Q | cited |
| SHARC / prior validation | ~286 mSv/yr | validation notes | OLTARIS / SHARC comparison | add (name the run config) |

---

## 7. Dose limits (post-hoc comparison only — NOT physics inputs)

| Limit | Value | Location | Source | Status |
|---|---|---|---|---|
| NASA 30-day | 250 mSv | `dosimetry.py DOSE_LIMITS_MSV` | **NASA-STD-3001**, Vol. 1 | **add** |
| NASA annual | 500 mSv | same | NASA-STD-3001 | add |
| Career (3% REID) | 600 mSv | same | NASA / NCRP-132 (3% REID) | add |
| ESA career | 1000 mSv | same | ESA career limit | add |

**Target-blind confirmation for the meeting:** these limits appear *only* in
`DOSE_LIMITS_MSV` and are used solely for the post-hoc "how does our number
compare" banner. No limit feeds into the MC transport, the kernel, the source
spectrum, or any calibration — verified during this audit. That's the point to
make if a supervisor asks whether the tool was tuned to hit a limit.

---

## Summary — what needs a citation added before publication

The physics that's already properly attributed in the code: **Usoskin 2005**
(GCR LIS), **King 1974** (SPE), **ICRP 60** and **NASA TP-2013-217375** (quality
factors), **Dobynde & Guo 2021** (density), **Zhang 2020 / Chang'E-4** (surface
anchor), **ISO 15390 / Matthiä 2013** (rigidity model).

The gaps to close (all values are correct — they just lack a named source in the
code):

1. **GCR elemental abundances** → Simpson (1983) or George et al. (2009). Also
   reconcile the H/He/C/**Si**/Fe vs H/He/C/**O**/Fe difference between the two
   engines.
2. **Gleeson & Axford (1968)** for the force-field approximation itself.
3. **Apollo-17 oxide assay** → name the exact sample / Lunar Sourcebook table.
4. **Scalar mean quality factors** (3.5, 1.5) → NCRP-142 / measured ⟨Q⟩ (or
   just document them as fallbacks superseded by the per-step scorers).
5. **Geant4 + G4NDL versions** for the transport.
6. **NASA-STD-3001 / NCRP-132 / ESA** for the comparison limits.
7. **Standard constants** (E₀ = 938.272 MeV, u = 931.494 MeV) → CODATA/PDG.

---

## Citation verification pass (web-checked 2026-09-14)

Every reference-derived number was re-checked to confirm (a) the cited source is
real and exists, and (b) the number/form in the code actually matches what that
source says. Classification: **VERIFIED** = source confirmed real *and* value
matches; **MATCH-CAVEAT** = source real, but the code's use differs from the
canonical statement (needs a wording fix, not a new citation); **GROUP-VALUE** =
deliberate aggregate, citation legitimises the data source not the exact number;
**NEEDS-USER-CHECK** = real source but I could not confirm the exact figure from
open sources.

| Number / form | Cited source | Verdict | Note |
|---|---|---|---|
| GCR LIS coeffs 1.9e4, −2.78, 0.4866, −2.51; units /(m²·s·sr·GeV) | Usoskin et al. (2005), after Burger, Potgieter & Heber (2000) | **VERIFIED** (coeffs + units) / **MATCH-CAVEAT** (variable) | Source real; coefficients & units are the standard Burger/Usoskin parameterisation. **BUT the canonical form's variable is rigidity P in GV, not kinetic energy T in GeV.** `lis_proton(T_GeV)` and the docstring (make_source.py:25–26, 123–125) substitute T directly for P. For a proton P=√(T²+2·T·E₀) so P>T (P/T≈4 at 100 MeV, ≈1.7 at 1 GeV, →1 at high E); using T for P over-weights the sub-GeV flux and distorts the *shape* the calibration then re-normalises. See dedicated note below. |
| Force-field modulation J(T)=J_LIS(T+Φ)·[T(T+2E₀)]/[(T+Φ)(T+Φ+2E₀)] | Gleeson & Axford (1968) | **VERIFIED** | Standard force-field equation; folded into docx §2.3. |
| Solar-min modulation φ≈400 MV (calib) / 325 MV context | George et al. (2009), ApJ 698, 1666 | **VERIFIED** | Paper confirmed real (ACE/CRIS, Z 5–28, 50–500 MeV/nuc); it reports φ=325 MV at solar min, 900 MV at max — brackets our 400 MV calib choice. |
| GCR abundances H1/He0.105/C0.033/Si0.012/Fe0.0020 | Simpson (1983) / George et al. (2009) | **GROUP-VALUE** | Explicitly Z-group aggregates ("representative group values, order-of-magnitude"), not single-element source values. Citation legitimises the data source; the grouping is the tool's. Still reconcile Si (dosimetry) vs O (kernel JSON) 5th species. |
| SPE exp-in-rigidity dN/dR∝exp(−R/R₀); R₀=100/150/220 MV | King (1974), J. Spacecraft & Rockets 11(6), 401 | **VERIFIED** / **MATCH-CAVEAT** | Paper confirmed real (DOI 10.2514/3.62088, Aug-1972 "anomalously large"). Caveat: King (1974) is a *fluence-probability* model; the exp-in-**rigidity** spectral form itself traces to Freier & Webber (1963). Code already lists Tylka & Dietrich (2009) as cross-check — consider crediting the form to Freier & Webber and the event fluences to King. |
| Regolith 3-layer density 1.76/2.11/1.78 g/cm³ | Dobynde & Guo (2021), JGR Planets, e2021JE006930 | **VERIFIED** | DOI/title/authors/journal all confirmed ("Radiation Environment at the Surface and Subsurface of the Moon…", REDMoon). |
| Surface absorbed-dose anchor (Chang'E-4 LND) | Zhang et al. (2020), Sci. Adv. 6, eaaz1334 | **NEEDS-USER-CHECK** | Reference is real & correct paper; confirm the exact dose-rate figure used for CAL=1.14867 against the paper's stated LND value. |
| Career limit 600 mSv (flat, all ages/sexes) | was NCRP 132; **should be NASEM (2021) + NASA-STD-3001 (2022/23)** | **MATCH-CAVEAT** | Confirmed: NASEM (2021) recommended the universal ~600 mSv (3% REID of 35-yr female); NASA implemented it via NASA-STD-3001. NCRP 132 is the older LEO-guidance basis. Add the **NASEM 2021** citation for the flat-600 provenance. |
| ~2 protons/cm²/s solar-min lunar-surface flux anchor (GCR_PROTON_FLUX_REF_2PI) | *(was uncited)* | **VERIFIED-support-found** | Open sources confirm order-of-magnitude: ~2 p/cm²/s up to 1 GeV/nuc, and integral 4π GCR proton flux 1.5–5.0 /cm²/s over the solar cycle (solar-min high end). Add a citable anchor (e.g. CRaTER lunar-orbit integral-flux, Zeitlin et al. 2019, Space Weather; or Simpson 1983). Still the single most important number to pin with a named source. **2026-10-27: now used ONLY by the legacy 'none' model; matthia2013/perelement keep Matthiä's own absolute flux (2.49 /cm²/s 2π, 10 MeV–20 GeV), which matches PAMELA Dec-2009 protons to 1.00 (Adriani et al. 2013, ApJ 765, 91).** |
| E₀=938.272 MeV; u=931.494 MeV | PDG / CODATA | **VERIFIED** | Standard constants; PDG entry folded into docx §2.3. |
| DEFAULT_QUALITY_FACTOR=3.5; DEFAULT_SPE_QF=1.5 | — (fallback) | **INTERNAL** | Not source-derived — documented fallbacks superseded by the per-step ICRP-60 / NASA-TP-2013 scorers. Keep labelled as fallbacks; no citation required, but a ⟨Q⟩~2–6 literature-range note (NCRP 142) helps. |

### The one substantive physics flag: LIS variable (rigidity vs energy)

This is the only item where a real citation may not faithfully match the code.
The Burger/Usoskin LIS is canonically written in **rigidity P (GV)**:
`J = 1.9e4·P^−2.78 / (1 + 0.4866·P^−2.51)`. The code evaluates it at **kinetic
energy per nucleon T (GeV)** instead. Two honest options for the paper:

1. **Treat it as an approximation** and say so explicitly — the spectrum is
   calibrated to a fixed integral flux (Chang'E-4-anchored) and validated against
   OLTARIS, so the surviving effect is a modest sub-GeV shape distortion, not a
   normalisation error. Cheapest; defensible given the empirical validation.
2. **Fix it** — pass `P(T)=√(T²+2·T·E₀)` into `lis_proton` (a one-line change in
   `gcr_spectrum`), re-anchor CAL, and re-run the regolith/validation suite. This
   makes the code match the cited source exactly. Preferred if a reviewer is
   likely to check the LIS form against Usoskin (2005).

**Do not** cite Usoskin (2005) for an energy-variable LIS without either fixing
the code or adding an explicit "evaluated in kinetic-energy approximation" caveat
in §2.3 — that mismatch is exactly what this verification pass exists to catch.

### Bottom line

All cited sources are **real and correctly attributed**. Every reference-derived
number matches its source except two wording/rigor items: the **LIS variable**
(rigidity vs energy — the one to decide on) and the **career-limit provenance**
(add NASEM 2021 for the flat-600). Two figures still want a first-party
number-check against the paper: the **Chang'E-4 dose rate** behind CAL, and a
named **~2 p/cm²/s** flux citation.
