// Scorer for DoseEquivalent_ICRP_ParentQ
//
// DIAGNOSTIC twin of DoseEquivalent_ICRP (code review 2026-11-15, "delta rays at
// Q = 1"). Transport is untouched -- this is one extra scorer in the same run -- so
// absorbed dose is identical by construction; only the Q given to delta electrons
// changes.
//
// DoseEquivalent_ICRP scores edep * Q(L) with L the unrestricted LET of the
// STEPPING particle. A delta electron produced above the production cut is its own
// track with LET < 10 keV/um, so its energy is scored at Q = 1. Under a
// primary-L_inf convention (CRaTER-style <Q>), that energy belongs to the ion and
// should carry the ion's Q. This twin applies that convention:
//
//   * every step scores edep * Q_own, exactly as DoseEquivalent_ICRP, EXCEPT
//   * an e- created by an ionisation process ("...Ioni": hIoni, ionIoni, muIoni,
//     eIoni) on a step whose scored Q was > 1 inherits that Q, and scores ALL of
//     its own energy deposits with it. Its own deltas inherit the same Q
//     (cascade), because the parent Q passed on is the Q this step was scored with.
//
// Convention caveat: ICRU's D_L over ALL charged particles arguably already scores
// deltas at their own L (that is what DoseEquivalent_ICRP does). The ratio
// P / I therefore measures how big the convention choice is; it is not by itself
// a bug fix. Adopting either convention is a user decision.
//
// Bookkeeping. Secondaries get track IDs only when stacked, so at birth they are
// keyed by G4Track pointer, with the parent's track ID and the secondary's
// kinetic energy stored as a check: the pointer persists until the secondary is
// tracked, and the check rejects a reused (pooled) pointer. On the secondary's
// first scored step the entry is matched, erased, and its Q cached for the rest of
// that track. The map is thread_local and SHARED across all instances of this
// scorer (one per organ shell), so a delta born in one shell that deposits in a
// neighbour still inherits. It is cleared whenever the event changes.
//
// Limitations (bias P/I LOW, i.e. the reported effect is a lower bound): deltas
// born OUTSIDE any volume scored by this quantity (in the wall, in air) keep Q = 1;
// energy carried away by bremsstrahlung / fluorescence photons is not inherited;
// sub-cut deltas are already part of the ion's continuous loss and scored at the
// ion's Q by both scorers, so the effect depends on the production cut (production
// default 0.05 mm, ~57 keV in water).
//
// Output units "Gy"; the value is numerically Sv (Q dimensionless), as the twin.

#include "DoseEquivalentICRPParentQ.hh"

#include "G4Step.hh"
#include "G4Track.hh"
#include "G4Event.hh"
#include "G4EventManager.hh"
#include "G4ParticleDefinition.hh"
#include "G4Electron.hh"
#include "G4VProcess.hh"
#include "G4SystemOfUnits.hh"

#include <cmath>
#include <unordered_map>

namespace {
	// ICRP Publication 60 quality factor Q(L); L is unrestricted LET in water in
	// keV/micrometre. (Same mapping as DoseEquivalentICRP.)
	G4double QualityFactor(G4double let)
	{
		if (let < 10.)   return 1.0;
		if (let <= 100.) return 0.32 * let - 2.2;
		return 300.0 / std::sqrt(let);
	}

	struct Birth {
		G4int parentID;
		G4double kineticEnergy;
		G4double q;
	};

	struct ParentQState {
		const G4Event* event = nullptr;
		G4int eventID = -1;
		std::unordered_map<const G4Track*, Birth> births;
		G4int curTrackID = -1;
		G4double curQ = 1.0;     // inherited Q of the current track (1 = none)
	};

	ParentQState& State()
	{
		static thread_local ParentQState s;
		return s;
	}

	void SyncEvent(ParentQState& s)
	{
		G4EventManager* em = G4EventManager::GetEventManager();
		const G4Event* e = em ? em->GetConstCurrentEvent() : nullptr;
		G4int id = e ? e->GetEventID() : -1;
		if (e != s.event || id != s.eventID) {
			s.births.clear();
			s.event = e;
			s.eventID = id;
			s.curTrackID = -1;
			s.curQ = 1.0;
		}
	}

	// Inherited Q of this track (1.0 if it is not a recorded delta electron).
	G4double InheritedQ(ParentQState& s, const G4Track* track)
	{
		G4int tid = track->GetTrackID();
		if (tid == s.curTrackID) return s.curQ;
		s.curTrackID = tid;
		s.curQ = 1.0;
		auto it = s.births.find(track);
		if (it != s.births.end()) {
			if (it->second.parentID == track->GetParentID() &&
				it->second.kineticEnergy == track->GetVertexKineticEnergy())
				s.curQ = it->second.q;
			s.births.erase(it);
		}
		return s.curQ;
	}

	G4bool IsIonisationName(const G4String& n)
	{
		return n.size() >= 4 && n.compare(n.size() - 4, 4, "Ioni") == 0;
	}
}

DoseEquivalentICRPParentQ::DoseEquivalentICRPParentQ(TsParameterManager* pM, TsMaterialManager* mM, TsGeometryManager* gM,
													 TsScoringManager* scM, TsExtensionManager* eM,
													 G4String scorerName, G4String quantity, G4String outFileName, G4bool isSubScorer)
	: TsVBinnedScorer(pM, mM, gM, scM, eM, scorerName, quantity, outFileName, isSubScorer)
{
	SetUnit("Gy");   // numerically Sv -- the quality factor Q is dimensionless
}

DoseEquivalentICRPParentQ::~DoseEquivalentICRPParentQ() {}

G4double DoseEquivalentICRPParentQ::OwnQ(const G4Step* aStep)
{
	// Exactly DoseEquivalentICRP: unrestricted electronic LET of THIS step's
	// particle; neutral particles fall back to Q = 1.
	const G4ParticleDefinition* particle = aStep->GetTrack()->GetParticleDefinition();
	if (particle->GetPDGCharge() == 0.) return 1.0;
	G4StepPoint* preStep = aStep->GetPreStepPoint();
	G4double dedx = fEmCalculator.ComputeElectronicDEDX(preStep->GetKineticEnergy(), particle,
														preStep->GetMaterial());
	if (dedx <= 0.) return 1.0;
	return QualityFactor(dedx / (keV / um));
}

G4bool DoseEquivalentICRPParentQ::ProcessHits(G4Step* aStep, G4TouchableHistory*)
{
	if (!fIsActive) {
		fSkippedWhileInactive++;
		return false;
	}

	ParentQState& s = State();
	SyncEvent(s);
	const G4Track* track = aStep->GetTrack();
	G4double inherited = InheritedQ(s, track);

	G4double edep = aStep->GetTotalEnergyDeposit();
	const std::vector<const G4Track*>* secondaries = aStep->GetSecondaryInCurrentStep();
	G4bool haveSecondaries = secondaries && !secondaries->empty();
	if (edep <= 0. && !haveSecondaries) return false;

	// Q this step is scored with: inherited (delta electron) or its own.
	G4bool isElectron = track->GetParticleDefinition() == G4Electron::Electron();
	G4double q = (isElectron && inherited > 1.) ? inherited : OwnQ(aStep);

	// Record delta electrons born on this step, BEFORE the edep early return: a
	// step can create a delta while depositing nothing locally.
	if (haveSecondaries && q > 1.) {
		const G4ParticleDefinition* electron = G4Electron::Electron();
		for (const G4Track* sec : *secondaries) {
			if (sec->GetParticleDefinition() != electron) continue;
			const G4VProcess* cp = sec->GetCreatorProcess();
			if (!cp || !IsIonisationName(cp->GetProcessName())) continue;
			s.births[sec] = Birth{track->GetTrackID(), sec->GetKineticEnergy(), q};
		}
	}

	if (edep <= 0.) return false;

	ResolveSolid(aStep);

	G4StepPoint* preStep = aStep->GetPreStepPoint();
	G4Material* material = preStep->GetMaterial();
	G4double mass = material->GetDensity() * GetCubicVolume(aStep);
	if (mass <= 0.) return false;

	G4double doseEquivalent = (edep * q / mass) * preStep->GetWeight();
	AccumulateHit(aStep, doseEquivalent);
	return true;
}
