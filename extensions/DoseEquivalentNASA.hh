#ifndef DoseEquivalentNASA_hh
#define DoseEquivalentNASA_hh

#include "TsVBinnedScorer.hh"

#include "G4EmCalculator.hh"

// Dose-equivalent (Sv) scorer using the NASA/Cucinotta space-radiation quality
// factor -- see DoseEquivalentNASA.cc for the physics rationale. Registered under
// the parameter-file quantity name "DoseEquivalent_NASA". Deliberately a drop-in
// twin of DoseEquivalent_ICRP: identical transport, identical LET calculation,
// the ONLY difference is Q(L) (ICRP-60) vs Q(Z*^2/beta^2, L) (NASA-2013), so the
// two scored side by side isolate the quality-factor model choice.
class DoseEquivalentNASA : public TsVBinnedScorer
{
public:
	DoseEquivalentNASA(TsParameterManager* pM, TsMaterialManager* mM, TsGeometryManager* gM,
					   TsScoringManager* scM, TsExtensionManager* eM,
					   G4String scorerName, G4String quantity, G4String outFileName, G4bool isSubScorer);
	virtual ~DoseEquivalentNASA();

	virtual G4bool ProcessHits(G4Step*, G4TouchableHistory*);

private:
	G4EmCalculator fEmCalculator;
};

#endif
