#ifndef LETSpectrumWater_hh
#define LETSpectrumWater_hh

#include "TsVNtupleScorer.hh"

#include "G4EmCalculator.hh"

// Per-step LET-in-water ntuple scorer -- see LETSpectrumWater.cc for the physics
// rationale. Registered under the parameter-file quantity name "LETSpectrum_Water".
// Score it on a G4_WATER detector volume so the recorded LET is LET-in-water,
// directly comparable to CRaTER / the arXiv 2511.09040 water-point reference.
class LETSpectrumWater : public TsVNtupleScorer
{
public:
	LETSpectrumWater(TsParameterManager* pM, TsMaterialManager* mM, TsGeometryManager* gM,
					 TsScoringManager* scM, TsExtensionManager* eM,
					 G4String scorerName, G4String quantity, G4String outFileName, G4bool isSubScorer);
	virtual ~LETSpectrumWater();

	virtual G4bool ProcessHits(G4Step*, G4TouchableHistory*);

private:
	G4EmCalculator fEmCalculator;

	// ntuple columns (bound by address in the constructor)
	G4double fLet;       // unrestricted electronic LET in water, keV/um
	G4double fEdep;      // energy deposited this step, MeV
	G4double fStepLen;   // geometric step length, um
	G4int    fCharge;    // particle charge in units of e (for HZE/light split)
};

#endif
