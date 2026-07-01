#ifndef DoseEquivalentICRP_hh
#define DoseEquivalentICRP_hh

#include "TsVBinnedScorer.hh"

#include "G4EmCalculator.hh"

// LET-weighted dose-equivalent (Sv) scorer -- see DoseEquivalentICRP.cc for the
// physics rationale. Registered under the parameter-file quantity name
// "DoseEquivalent_ICRP".
class DoseEquivalentICRP : public TsVBinnedScorer
{
public:
	DoseEquivalentICRP(TsParameterManager* pM, TsMaterialManager* mM, TsGeometryManager* gM,
					   TsScoringManager* scM, TsExtensionManager* eM,
					   G4String scorerName, G4String quantity, G4String outFileName, G4bool isSubScorer);
	virtual ~DoseEquivalentICRP();

	virtual G4bool ProcessHits(G4Step*, G4TouchableHistory*);

private:
	G4EmCalculator fEmCalculator;
};

#endif
