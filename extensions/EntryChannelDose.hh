#ifndef EntryChannelDose_hh
#define EntryChannelDose_hh

#include "TsVBinnedScorer.hh"

#include <unordered_map>

// Absorbed dose (Gy) split by ENTRY CHANNEL -- see EntryChannelDose.cc for the
// rationale. A deposit counts only when the particle that carried its lineage
// INTO the scored volume was of the selected kind (charged / neutron / gamma /
// any neutral), the in-volume analogue of Chang'E-4 LND's anticoincidence split
// of total vs neutral dose. Registered under the parameter-file quantity name
// "EntryChannelDose"; the channel is chosen with s:Sc/<name>/Channel.
class EntryChannelDose : public TsVBinnedScorer
{
public:
	EntryChannelDose(TsParameterManager* pM, TsMaterialManager* mM, TsGeometryManager* gM,
					 TsScoringManager* scM, TsExtensionManager* eM,
					 G4String scorerName, G4String quantity, G4String outFileName, G4bool isSubScorer);
	virtual ~EntryChannelDose();

	virtual G4bool ProcessHits(G4Step*, G4TouchableHistory*);

	enum Channel { kCharged = 0, kNeutron = 1, kGamma = 2, kOtherNeutral = 3 };

private:
	G4int ClassifyTrack(const G4Step* aStep);
	G4bool Selected(G4int channel) const;

	G4String fChannelName;
	// trackID -> entry channel, for the CURRENT event only (track IDs restart
	// every event). Reset when the event ID changes.
	std::unordered_map<G4int, G4int> fTrackChannel;
	G4int fEventID;
};

#endif
