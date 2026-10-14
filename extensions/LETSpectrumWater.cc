// Scorer for LETSpectrum_Water
//
// Per-step LET-in-water ntuple scorer for the thin-wall Q-bar cross-check.
//
// For every charged step inside the sensitive volume it records one ntuple row:
//   LETWater      -- the unrestricted electronic LET of the stepping particle in
//                    the local material, via G4EmCalculator (cut = DBL_MAX),
//                    expressed in keV/um. Score on a G4_WATER volume so this is
//                    LET-in-water -- the quantity CRaTER measures and the
//                    arXiv 2511.09040 Geant4/INCLXX reference tabulates.
//   EnergyDeposit -- energy deposited on this step (MeV).
//   StepLength    -- geometric step length (um); the track-length contribution to
//                    the fluence LET spectrum dPhi/dL.
//   Charge        -- particle charge in units of e, so the collect step can split
//                    the HZE (Z>=3) tail that drives the thin-wall over-read from
//                    the light (H/He) and secondary-electron background offline.
//
// This is the per-hit TWIN of DoseEquivalentICRP: that scorer maps the same
// per-step LET through ICRP-60 Q(L) and accumulates dose-equivalent; here we emit
// the raw (LET, edep, steplen) triples so the fluence- and dose-weighted LET
// spectra -- and the dose-averaged <Q> in a water POINT (not effective-dose,
// organ-weighted) -- are reconstructed offline exactly the way the reference does,
// removing the effective-dose weighting confound.
//
// Neutral particles (neutrons, gammas) carry no electronic LET; they deposit via
// charged secondaries that step separately and are recorded on their own steps,
// so they are skipped here (matching DoseEquivalentICRP's Q=1 fall-back logic).
//
// The kernel-generation ring source uses no biasing / variance reduction, so every
// primary and secondary carries weight 1; weight is therefore not recorded and
// every row counts once in the offline fold.

#include "LETSpectrumWater.hh"

#include "G4Step.hh"
#include "G4ParticleDefinition.hh"
#include "G4SystemOfUnits.hh"

#include <cmath>

LETSpectrumWater::LETSpectrumWater(TsParameterManager* pM, TsMaterialManager* mM, TsGeometryManager* gM,
								   TsScoringManager* scM, TsExtensionManager* eM,
								   G4String scorerName, G4String quantity, G4String outFileName, G4bool isSubScorer)
	: TsVNtupleScorer(pM, mM, gM, scM, eM, scorerName, quantity, outFileName, isSubScorer),
	  fLet(0.), fEdep(0.), fStepLen(0.), fCharge(0)
{
	fNtuple->RegisterColumnD(&fLet,     "LETWater",      "keV/um");
	fNtuple->RegisterColumnD(&fEdep,    "EnergyDeposit", "MeV");
	fNtuple->RegisterColumnD(&fStepLen, "StepLength",    "um");
	fNtuple->RegisterColumnI(&fCharge,  "Charge");
}

LETSpectrumWater::~LETSpectrumWater() {}

G4bool LETSpectrumWater::ProcessHits(G4Step* aStep, G4TouchableHistory*)
{
	if (!fIsActive) {
		fSkippedWhileInactive++;
		return false;
	}

	G4double edep = aStep->GetTotalEnergyDeposit();
	if (edep <= 0.) return false;

	const G4ParticleDefinition* particle = aStep->GetTrack()->GetParticleDefinition();
	if (particle->GetPDGCharge() == 0.) return false;   // neutral: scored via its charged secondaries

	G4StepPoint* preStep = aStep->GetPreStepPoint();
	G4Material* material = preStep->GetMaterial();
	G4double kinE = preStep->GetKineticEnergy();
	G4double dedx = fEmCalculator.ComputeElectronicDEDX(kinE, particle, material);
	if (dedx <= 0.) return false;

	// Store in G4 native units; the registered unit string converts on write.
	fLet     = dedx;                      // native MeV/mm == keV/um numerically
	fEdep    = edep;                      // native MeV
	fStepLen = aStep->GetStepLength();    // native mm -> written as um
	fCharge  = (G4int) std::lround(particle->GetPDGCharge() / eplus);

	fNtuple->Fill();
	return true;
}
