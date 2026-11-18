#ifndef DoseEquivalentICRPParentQ_hh
#define DoseEquivalentICRPParentQ_hh

#include "TsVBinnedScorer.hh"

#include "G4EmCalculator.hh"

// Delta-ray parent-Q twin of DoseEquivalentICRP -- see DoseEquivalentICRPParentQ.cc.
// Identical transport and identical per-step LET-weighted ICRP-60 dose-equivalent,
// EXCEPT that a delta electron knocked out by an ion's ionisation scores its energy
// with the PARENT's Q instead of its own (~1). Scored on the same volume as
// DoseEquivalent_ICRP so the paired same-run ratio P/I is the size of the
// delta-ray Q convention. Registered as quantity "DoseEquivalent_ICRP_ParentQ".
class DoseEquivalentICRPParentQ : public TsVBinnedScorer
{
public:
	DoseEquivalentICRPParentQ(TsParameterManager* pM, TsMaterialManager* mM, TsGeometryManager* gM,
							  TsScoringManager* scM, TsExtensionManager* eM,
							  G4String scorerName, G4String quantity, G4String outFileName, G4bool isSubScorer);
	virtual ~DoseEquivalentICRPParentQ();

	virtual G4bool ProcessHits(G4Step*, G4TouchableHistory*);

private:
	G4double OwnQ(const G4Step* aStep);

	G4EmCalculator fEmCalculator;
};

#endif
