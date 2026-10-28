// Scorer for EntryChannelDose
//
// Absorbed dose (Gy, same as DoseToMedium) restricted to one ENTRY CHANNEL, so
// the tool's total skin dose can be split the way Chang'E-4 LND reports it
// (Zhang et al. 2020: total 13.2, neutral 3.1, charged 10.2 uGy/h in Si). LND
// separates neutral from charged dose by anticoincidence: a deposit is "neutral"
// when no charged particle crossed into the detector to make it. The in-volume
// analogue scored here:
//
//   * a track seen for the first time ENTERING the scored volume is tagged by its
//     own type: charged, neutron, gamma or other neutral;
//   * a track BORN inside the volume inherits the tag of its parent (which must
//     already have stepped in the volume to create it there).
//
// So a neutron-elastic recoil proton made in the lining is "neutron", a Compton
// electron from an entering photon is "gamma", while a delta ray or a
// bremsstrahlung photon from an entering ion stays "charged" -- exactly the
// events LND's anticoincidence would put in each bin. Charged particles made
// OUTSIDE (e.g. in the wall, by a neutron) that then enter count as charged, as
// they would for LND.
//
// s:Sc/<name>/Channel = "Neutral" (neutron + gamma + other neutral; default),
//                       "Neutron", "Gamma" or "Charged".
// Scored on the same lining as DoseToMedium, channel/total is a ratio of two
// absorbed doses from one run, so the flux/gauge normalisation cancels.
//
// Tracks born inside the volume whose parent was never seen there (should not
// happen) default to charged, which can only UNDER-count the neutral share.

#include "EntryChannelDose.hh"

#include "G4Step.hh"
#include "G4Track.hh"
#include "G4ParticleDefinition.hh"
#include "G4Neutron.hh"
#include "G4Gamma.hh"
#include "G4EventManager.hh"
#include "G4Event.hh"
#include "G4LogicalVolume.hh"
#include "G4VPhysicalVolume.hh"

#include <algorithm>
#include <cctype>

EntryChannelDose::EntryChannelDose(TsParameterManager* pM, TsMaterialManager* mM, TsGeometryManager* gM,
								   TsScoringManager* scM, TsExtensionManager* eM,
								   G4String scorerName, G4String quantity, G4String outFileName, G4bool isSubScorer)
	: TsVBinnedScorer(pM, mM, gM, scM, eM, scorerName, quantity, outFileName, isSubScorer),
	  fChannelName("neutral"), fEventID(-1)
{
	SetUnit("Gy");
	if (fPm->ParameterExists(GetFullParmName("Channel"))) {
		fChannelName = fPm->GetStringParameter(GetFullParmName("Channel"));
		std::transform(fChannelName.begin(), fChannelName.end(), fChannelName.begin(),
					   [](unsigned char c) { return std::tolower(c); });
	}
	if (fChannelName != "neutral" && fChannelName != "neutron" &&
		fChannelName != "gamma" && fChannelName != "charged") {
		G4cerr << "Topas is exiting due to a serious error in scoring setup." << G4endl;
		G4cerr << GetFullParmName("Channel") << " = " << fChannelName
			   << " -- must be Neutral, Neutron, Gamma or Charged." << G4endl;
		fPm->AbortSession(1);
	}
}

EntryChannelDose::~EntryChannelDose() {}

G4int EntryChannelDose::ClassifyTrack(const G4Step* aStep)
{
	const G4Event* evt = G4EventManager::GetEventManager()->GetConstCurrentEvent();
	const G4int eventID = evt ? evt->GetEventID() : -1;
	if (eventID != fEventID) {
		fTrackChannel.clear();
		fEventID = eventID;
	}

	const G4Track* track = aStep->GetTrack();
	const G4int id = track->GetTrackID();
	auto it = fTrackChannel.find(id);
	if (it != fTrackChannel.end()) return it->second;

	G4int channel = kCharged;
	const G4LogicalVolume* here = aStep->GetPreStepPoint()->GetPhysicalVolume()->GetLogicalVolume();
	const G4bool bornHere = (track->GetParentID() > 0 && track->GetLogicalVolumeAtVertex() == here);
	if (bornHere) {
		auto parent = fTrackChannel.find(track->GetParentID());
		if (parent != fTrackChannel.end()) channel = parent->second;
	} else {
		const G4ParticleDefinition* pdef = track->GetParticleDefinition();
		if (pdef == G4Neutron::Neutron())   channel = kNeutron;
		else if (pdef == G4Gamma::Gamma())  channel = kGamma;
		else if (pdef->GetPDGCharge() == 0.) channel = kOtherNeutral;
	}
	fTrackChannel[id] = channel;
	return channel;
}

G4bool EntryChannelDose::Selected(G4int channel) const
{
	if (fChannelName == "neutral") return channel != kCharged;
	if (fChannelName == "neutron") return channel == kNeutron;
	if (fChannelName == "gamma")   return channel == kGamma;
	return channel == kCharged;
}

G4bool EntryChannelDose::ProcessHits(G4Step* aStep, G4TouchableHistory*)
{
	if (!fIsActive) {
		fSkippedWhileInactive++;
		return false;
	}

	// Classify EVERY step (including zero-deposit neutral steps), so a parent is
	// always tagged before any secondary it creates in the volume is tracked.
	const G4int channel = ClassifyTrack(aStep);

	G4double edep = aStep->GetTotalEnergyDeposit();
	if (edep <= 0.) return false;
	if (!Selected(channel)) return false;

	ResolveSolid(aStep);

	G4StepPoint* preStep = aStep->GetPreStepPoint();
	G4double mass = preStep->GetMaterial()->GetDensity() * GetCubicVolume(aStep);
	if (mass <= 0.) return false;

	AccumulateHit(aStep, (edep / mass) * preStep->GetWeight());
	return true;
}
