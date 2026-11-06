#ifndef WeightedDiscBeam_hh
#define WeightedDiscBeam_hh

#include "TsVGenerator.hh"

// Parallel-ray disc source (the lunarsim GCR "Beam" disc: uniform on a disc of
// radius DiscRadius in the component's XY plane, every ray along the component's
// local +Z) with optional, UNBIASED importance sampling of the start point toward
// a target (the crew phantom).
//
// A fraction BiasFraction of histories start uniformly inside the target's
// shadow circle (radius BiasRadius around the target's projection onto the disc
// along the ray direction); the rest start uniformly over the whole disc. Every
// primary carries the weight
//     w = p_true / q = 1 / ((1 - f) + f * (R/r_b)^2 * [inside shadow])
// so E_q[w * score] = E_true[score] exactly: dose, albedo and fluence keep the
// physical field, only the variance changes. With BiasFraction = 0 this is the
// plain uniform disc with weight 1 (identical to the TOPAS Beam/Flat/Ellipse
// source). Energy and particle come from the standard Beam parameters
// (BeamParticle, BeamEnergySpectrum*), resolved by TsVGenerator.
class WeightedDiscBeam : public TsVGenerator
{
public:
	WeightedDiscBeam(TsParameterManager* pM, TsGeometryManager* gM,
					 TsGeneratorManager* pgM, G4String sourceName);
	~WeightedDiscBeam();

	void ResolveParameters();
	void CacheGeometryPointers();
	void GeneratePrimaries(G4Event*);

private:
	void ComputeShadow();

	G4double fDiscRadius;
	G4double fBiasRadius;
	G4double fBiasFraction;
	G4double fTargetX, fTargetY, fTargetZ;

	G4bool fShadowValid;
	G4double fShadowX, fShadowY;        // target projection, component-local
	G4double fWeightInside, fWeightOutside;
};

#endif
