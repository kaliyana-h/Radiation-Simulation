// Scorer for DoseEquivalent_NASA
//
// Per-step dose-equivalent (Sv) for a mixed GCR field, weighted by the NASA /
// Cucinotta space-radiation quality factor rather than the ICRP-60 Q(L) used by
// the DoseEquivalent_ICRP scorer.
//
// WHY A SECOND SCORER. The two established quality-factor models disagree by a
// design-dependent factor (~1.6-2x for the lunar-surface GCR field) because they
// are non-linear functionals of DIFFERENT variables:
//   * ICRP-60 Q(L)      -- a function of unrestricted LET only, peaking ~30 at
//                          100 keV/um and turning DOWN above that (overkill).
//   * NASA-2013 Q       -- a function of track structure, Z*^2/beta^2 AND LET,
//                          fit to HZE cancer radiobiology; keeps heavy ions at
//                          high quality where Q(L) rolls off, so the effective
//                          field quality (and the annual effective dose) is
//                          higher for the same transport.
// Because that ratio is NOT constant across shielding/geometry, the NASA number
// cannot be recovered by scaling the ICRP number by a fixed constant -- it must
// be computed per design, from the same field. This scorer does exactly that.
//
// The transport, the sensitive volume, and the LET evaluation
// (G4EmCalculator::ComputeElectronicDEDX, cut = DBL_MAX) are IDENTICAL to
// DoseEquivalent_ICRP. The sole difference is the Q mapping below, so scoring
// both on the same component yields two dose-equivalents whose ratio is purely
// the quality-factor-model choice -- nothing else.
//
// Model: NASA/TP-2013-217375 (Cucinotta, Kim, Chappell), solid-cancer QF.
//   Q = (1 - P) + (X / L) * P
//   P = [1 - exp(-(Z*^2/beta^2) / kappa)]^m * [1 - exp(-E/0.2)]
//   X = 6.24 * (Sigma0/alpha_gamma) = 7000 keV/um   (solid cancer; leukaemia = 1750)
//   kappa = 1000 (Z <= 4), 500 (Z > 4);   m = 3
//   Z* = Z [1 - exp(-125 beta Z^(-2/3))]   (Barkas effective charge)
//   E  = kinetic energy per nucleon in MeV/u;   L = LET in keV/um
//
// Output units are set to "Gy"; the value is numerically Sv because Q is
// dimensionless. Score it on the same tissue-equivalent component as the plain
// DoseToMedium scorer so absorbed dose, ICRP dose-equivalent, and NASA
// dose-equivalent are all directly comparable.

#include "DoseEquivalentNASA.hh"

#include "G4Step.hh"
#include "G4ParticleDefinition.hh"
#include "G4SystemOfUnits.hh"

#include <cmath>

namespace {
	// NASA/Cucinotta solid-cancer quality factor (NASA/TP-2013-217375, Eqs 25-26).
	//   let  : unrestricted electronic LET in keV/micrometre
	//   z    : atomic number of the stepping ion
	//   a    : mass number (nucleon count) of the stepping ion
	//   kinE : kinetic energy of the ion in Geant4 internal units
	G4double QualityFactorNASA(G4double let, G4int z, G4int a, G4double kinE)
	{
		if (z < 1 || let <= 0. || a < 1) return 1.0;

		// beta from kinetic energy per nucleon (u = 931.494 MeV)
		const G4double amuMeV = 931.494;
		G4double ePerNuc = (kinE / MeV) / static_cast<G4double>(a);   // MeV/u
		G4double gamma   = 1.0 + ePerNuc / amuMeV;
		G4double beta2   = 1.0 - 1.0 / (gamma * gamma);
		if (beta2 <= 0.) return 1.0;
		G4double beta = std::sqrt(beta2);

		// Barkas effective charge  Z* = Z [1 - exp(-125 beta Z^(-2/3))]
		G4double zEff = z * (1.0 - std::exp(-125.0 * beta * std::pow(static_cast<G4double>(z), -2.0 / 3.0)));
		G4double zeb2 = (zEff * zEff) / beta2;                        // (Z*/beta)^2

		// track-structure probability P_{Z,E}
		const G4double kappa = (z <= 4) ? 1000.0 : 500.0;
		const G4double m     = 3.0;
		G4double P = std::pow(1.0 - std::exp(-zeb2 / kappa), m)
					 * (1.0 - std::exp(-ePerNuc / 0.2));

		// Q = (1 - P) + (X / L) P,  X = 7000 keV/um (solid cancer peak-height constant)
		const G4double X = 7000.0;
		return (1.0 - P) + (X / let) * P;
	}
}

DoseEquivalentNASA::DoseEquivalentNASA(TsParameterManager* pM, TsMaterialManager* mM, TsGeometryManager* gM,
									   TsScoringManager* scM, TsExtensionManager* eM,
									   G4String scorerName, G4String quantity, G4String outFileName, G4bool isSubScorer)
	: TsVBinnedScorer(pM, mM, gM, scM, eM, scorerName, quantity, outFileName, isSubScorer)
{
	SetUnit("Gy");   // numerically Sv -- the quality factor Q is dimensionless
}

DoseEquivalentNASA::~DoseEquivalentNASA() {}

G4bool DoseEquivalentNASA::ProcessHits(G4Step* aStep, G4TouchableHistory*)
{
	if (!fIsActive) {
		fSkippedWhileInactive++;
		return false;
	}

	G4double edep = aStep->GetTotalEnergyDeposit();
	if (edep <= 0.) return false;

	ResolveSolid(aStep);

	G4StepPoint* preStep = aStep->GetPreStepPoint();
	G4Material* material = preStep->GetMaterial();
	G4double mass = material->GetDensity() * GetCubicVolume(aStep);
	if (mass <= 0.) return false;

	// NASA quality factor from THIS step's particle. Neutral particles (neutrons,
	// gammas) deposit via charged secondaries that step separately, so for them
	// we fall back to Q = 1 -- identical treatment to DoseEquivalent_ICRP.
	G4double qualityFactor = 1.0;
	const G4ParticleDefinition* particle = aStep->GetTrack()->GetParticleDefinition();
	G4double charge = particle->GetPDGCharge();
	if (charge != 0.) {
		G4double kinE = preStep->GetKineticEnergy();
		G4double dedx = fEmCalculator.ComputeElectronicDEDX(kinE, particle, material);
		if (dedx > 0. && kinE > 0.) {
			G4double let = dedx / (keV / um);            // unrestricted LET in keV/micrometre
			G4int    z   = static_cast<G4int>(std::lround(charge / eplus));
			G4int    a   = particle->GetBaryonNumber();  // nucleon count (proton 1, alpha 4, Fe-56 56)
			qualityFactor = QualityFactorNASA(let, z, a, kinE);
		}
	}

	G4double doseEquivalent = (edep * qualityFactor / mass) * preStep->GetWeight();
	AccumulateHit(aStep, doseEquivalent);
	return true;
}
