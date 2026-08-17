#ifndef DoseEquivalentICRPNeutron_hh
#define DoseEquivalentICRPNeutron_hh

#include "TsVBinnedScorer.hh"

#include "G4EmCalculator.hh"

// Neutron-lineage twin of DoseEquivalentICRP -- see DoseEquivalentICRPNeutron.cc
// for the rationale. Identical LET-weighted ICRP-60 dose-equivalent, but a hit is
// accumulated ONLY when the depositing charged particle descends from (or is) a
// neutron. Scored on the SAME volume as DoseEquivalent_ICRP so the ratio of the
// two is the wall-bred secondary-neutron dose fraction. Registered under the
// parameter-file quantity name "DoseEquivalent_ICRP_Neutron".
class DoseEquivalentICRPNeutron : public TsVBinnedScorer
{
public:
	DoseEquivalentICRPNeutron(TsParameterManager* pM, TsMaterialManager* mM, TsGeometryManager* gM,
							  TsScoringManager* scM, TsExtensionManager* eM,
							  G4String scorerName, G4String quantity, G4String outFileName, G4bool isSubScorer);
	virtual ~DoseEquivalentICRPNeutron();

	virtual G4bool ProcessHits(G4Step*, G4TouchableHistory*);

private:
	G4bool NeutronInLineage(const G4Step* aStep) const;

	G4EmCalculator fEmCalculator;
};

#endif
