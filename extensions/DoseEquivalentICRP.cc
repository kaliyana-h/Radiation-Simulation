// Scorer for DoseEquivalent_ICRP
//
// Per-step, LET-weighted dose-equivalent (Sv) for a mixed GCR field.
//
// The flat single-field quality factor (Q = 3.5) that TOPAS' built-in dose
// scorers leave to the analysis layer badly under-weights heavy ions: an Fe
// nucleus and a proton deposit energy with the same Q, so the high-LET HZE
// component -- which dominates *dose-equivalent* on the lunar surface -- is lost
// and the annual effective dose comes out several-fold too low.
//
// This scorer fixes that at the source. For every charged step inside the
// sensitive volume it computes the unrestricted electronic LET of the stepping
// particle in the local material (via G4EmCalculator, cut = DBL_MAX), maps it to
// the radiation quality factor Q(L) of ICRP Publication 60, and accumulates
//     edep * Q(L) / mass
// Summed over all primaries AND secondaries (each secondary steps with its own
// LET, so degraded-spectrum and albedo-neutron recoil contributions behind thick
// shielding are captured automatically) this yields the true dose-equivalent.
// The effective field Q then emerges per design as (dose-equivalent / absorbed
// dose) instead of being assumed.
//
// Output units are set to "Gy"; the value is numerically Sv because Q is
// dimensionless. Score it on the same tissue-equivalent component as the plain
// DoseToMedium skin scorer so the two are directly comparable.

#include "DoseEquivalentICRP.hh"

#include "G4Step.hh"
#include "G4ParticleDefinition.hh"
#include "G4SystemOfUnits.hh"

#include <cmath>

namespace {
	// ICRP Publication 60 quality factor Q(L); L is unrestricted LET in water in
	// keV/micrometre.
	G4double QualityFactor(G4double let)
	{
		if (let < 10.)   return 1.0;
		if (let <= 100.) return 0.32 * let - 2.2;
		return 300.0 / std::sqrt(let);
	}
}

DoseEquivalentICRP::DoseEquivalentICRP(TsParameterManager* pM, TsMaterialManager* mM, TsGeometryManager* gM,
									   TsScoringManager* scM, TsExtensionManager* eM,
									   G4String scorerName, G4String quantity, G4String outFileName, G4bool isSubScorer)
	: TsVBinnedScorer(pM, mM, gM, scM, eM, scorerName, quantity, outFileName, isSubScorer)
{
	SetUnit("Gy");   // numerically Sv -- the quality factor Q is dimensionless
}

DoseEquivalentICRP::~DoseEquivalentICRP() {}

G4bool DoseEquivalentICRP::ProcessHits(G4Step* aStep, G4TouchableHistory*)
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

	// ICRP-60 quality factor from the unrestricted electronic LET of THIS step's
	// particle. Neutral particles (neutrons, gammas) deposit via charged
	// secondaries that step separately, so for them we fall back to Q = 1.
	G4double qualityFactor = 1.0;
	const G4ParticleDefinition* particle = aStep->GetTrack()->GetParticleDefinition();
	if (particle->GetPDGCharge() != 0.) {
		G4double kinE = preStep->GetKineticEnergy();
		G4double dedx = fEmCalculator.ComputeElectronicDEDX(kinE, particle, material);
		if (dedx > 0.) {
			G4double let = dedx / (keV / um);   // unrestricted LET in keV/micrometre
			qualityFactor = QualityFactor(let);
		}
	}

	G4double doseEquivalent = (edep * qualityFactor / mass) * preStep->GetWeight();
	AccumulateHit(aStep, doseEquivalent);
	return true;
}
