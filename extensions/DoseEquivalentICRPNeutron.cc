// Scorer for DoseEquivalent_ICRP_Neutron
//
// Neutron-lineage-filtered twin of DoseEquivalent_ICRP: the LET-weighted ICRP-60
// dose-equivalent (Sv), but summed ONLY over the energy deposited by charged
// particles that descend from a neutron. Scored on the same tissue-equivalent
// lining as DoseEquivalent_ICRP, the ratio
//     H_neutron / H_total
// is the fraction of the behind-shield crew dose-equivalent carried by wall-bred
// secondary (albedo) neutrons -- the last open workshop critique. Behind thick
// regolith the primary GCR/SPE field breeds neutrons in the shield; those neutrons
// deposit their dose indirectly, via recoil protons/ions (elastic + inelastic
// scattering) and capture/de-excitation gammas, so the signature is not "the
// particle is a neutron" (neutrons deposit almost nothing directly) but "this
// charged recoil has a neutron somewhere in its ancestry".
//
// The fraction is a ratio of two dose-equivalents scored in the SAME run, so the
// flux / gauge normalisation cancels: it needs no calibration and is reported as a
// pure dimensionless diagnostic alongside the effective dose.
//
// Lineage test (NeutronInLineage): rigorous when TsTrackInformation ancestry is
// available -- SetNeedsTrackingAction() in the constructor asks TOPAS to attach it
// -- in which case a hit counts iff a neutron appears anywhere in the depositing
// track's ancestry (or the track itself is a neutron). This cleanly separates a
// neutron-elastic recoil proton from a primary-proton-elastic recoil, which share
// the "hadElastic" process name and cannot be told apart from the creator process
// alone. If ancestry is somehow absent it falls back to the creator-process name
// (including hadElastic), which slightly over-counts but never silently returns
// zero.
//
// Output units are set to "Gy"; the value is numerically Sv because Q is
// dimensionless -- identical convention to DoseEquivalentICRP.

#include "DoseEquivalentICRPNeutron.hh"
#include "TsTrackInformation.hh"

#include "G4Step.hh"
#include "G4ParticleDefinition.hh"
#include "G4Neutron.hh"
#include "G4VProcess.hh"
#include "G4SystemOfUnits.hh"

#include <cmath>

namespace {
	// ICRP Publication 60 quality factor Q(L); L is unrestricted LET in water in
	// keV/micrometre. (Same mapping as DoseEquivalentICRP.)
	G4double QualityFactor(G4double let)
	{
		if (let < 10.)   return 1.0;
		if (let <= 100.) return 0.32 * let - 2.2;
		return 300.0 / std::sqrt(let);
	}
}

DoseEquivalentICRPNeutron::DoseEquivalentICRPNeutron(TsParameterManager* pM, TsMaterialManager* mM, TsGeometryManager* gM,
													 TsScoringManager* scM, TsExtensionManager* eM,
													 G4String scorerName, G4String quantity, G4String outFileName, G4bool isSubScorer)
	: TsVBinnedScorer(pM, mM, gM, scM, eM, scorerName, quantity, outFileName, isSubScorer)
{
	SetUnit("Gy");   // numerically Sv -- the quality factor Q is dimensionless
	// Ask TOPAS to attach TsTrackInformation so we can read the full track
	// ancestry (GetParticleDefs) in the neutron-lineage test.
	fPm->SetNeedsTrackingAction();
}

DoseEquivalentICRPNeutron::~DoseEquivalentICRPNeutron() {}

G4bool DoseEquivalentICRPNeutron::NeutronInLineage(const G4Step* aStep) const
{
	const G4Track* track = aStep->GetTrack();
	const G4ParticleDefinition* neutron = G4Neutron::Neutron();

	// Direct: the stepping particle is itself a neutron (rare edep, but count it).
	if (track->GetParticleDefinition() == neutron) return true;

	// Rigorous: full ancestry via TsTrackInformation (enabled by
	// SetNeedsTrackingAction). If present, its verdict is authoritative -- a
	// neutron anywhere in the lineage means neutron-induced, and its absence means
	// definitively not, so we do NOT fall through to the ambiguous process test.
	TsTrackInformation* ti = dynamic_cast<TsTrackInformation*>(track->GetUserInformation());
	if (ti) {
		for (G4ParticleDefinition* pdef : ti->GetParticleDefs())
			if (pdef == neutron) return true;
		return false;
	}

	// Fallback (ancestry unavailable): the creator process. Includes hadElastic,
	// which over-counts primary-elastic recoils, but never returns a silent zero.
	const G4VProcess* cp = track->GetCreatorProcess();
	if (cp) {
		const G4String& n = cp->GetProcessName();
		if (n == "neutronInelastic" || n == "nCapture" ||
			n == "nFission" || n == "hadElastic")
			return true;
	}
	return false;
}

G4bool DoseEquivalentICRPNeutron::ProcessHits(G4Step* aStep, G4TouchableHistory*)
{
	if (!fIsActive) {
		fSkippedWhileInactive++;
		return false;
	}

	G4double edep = aStep->GetTotalEnergyDeposit();
	if (edep <= 0.) return false;

	// Only neutron-descended deposits count towards this scorer.
	if (!NeutronInLineage(aStep)) return false;

	ResolveSolid(aStep);

	G4StepPoint* preStep = aStep->GetPreStepPoint();
	G4Material* material = preStep->GetMaterial();
	G4double mass = material->GetDensity() * GetCubicVolume(aStep);
	if (mass <= 0.) return false;

	// ICRP-60 quality factor from the unrestricted electronic LET of THIS step's
	// particle. Neutral particles fall back to Q = 1 (their charged secondaries
	// step separately and carry their own LET).
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
