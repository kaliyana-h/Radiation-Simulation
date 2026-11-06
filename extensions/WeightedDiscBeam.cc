// Particle Generator for WeightedDiscBeam
//
// lunarsim position-biased GCR disc source (variance reduction for the small
// crew phantom). See WeightedDiscBeam.hh for the estimator. Parameters, all
// relative to So/<source>/ :
//   s:Component        group whose XY plane holds the disc, local +Z = ray direction
//   s:BeamParticle, BeamEnergySpectrum*   as for a TOPAS "Beam" source
//   d:DiscRadius       disc radius (the Beam source's BeamPositionCutoffX/Y)
//   d:BiasTargetX/Y/Z  target centre in WORLD coordinates (optional, default 0)
//   d:BiasRadius       shadow-circle radius = target radius + margin
//   u:BiasFraction     fraction of histories drawn from the shadow, 0 <= f < 1
//
// TARGET-BLIND: the bias uses only the target's position and size, never a dose
// value; the weights make the estimator unbiased for every scorer that multiplies
// by the track weight.

#include "WeightedDiscBeam.hh"

#include "TsParameterManager.hh"

#include "G4Event.hh"
#include "G4ThreeVector.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"

#include <cmath>
#include <sstream>

WeightedDiscBeam::WeightedDiscBeam(TsParameterManager* pM, TsGeometryManager* gM,
								   TsGeneratorManager* pgM, G4String sourceName)
	: TsVGenerator(pM, gM, pgM, sourceName),
	  fDiscRadius(0.), fBiasRadius(0.), fBiasFraction(0.),
	  fTargetX(0.), fTargetY(0.), fTargetZ(0.),
	  fShadowValid(false), fShadowX(0.), fShadowY(0.),
	  fWeightInside(1.), fWeightOutside(1.)
{
	ResolveParameters();
}


WeightedDiscBeam::~WeightedDiscBeam()
{
}


void WeightedDiscBeam::ResolveParameters()
{
	TsVGenerator::ResolveParameters();

	fDiscRadius = fPm->GetDoubleParameter(GetFullParmName("DiscRadius"), "Length");

	fBiasFraction = 0.;
	if (fPm->ParameterExists(GetFullParmName("BiasFraction")))
		fBiasFraction = fPm->GetUnitlessParameter(GetFullParmName("BiasFraction"));

	fBiasRadius = 0.;
	if (fBiasFraction > 0.)
		fBiasRadius = fPm->GetDoubleParameter(GetFullParmName("BiasRadius"), "Length");

	fTargetX = fTargetY = fTargetZ = 0.;
	if (fPm->ParameterExists(GetFullParmName("BiasTargetX")))
		fTargetX = fPm->GetDoubleParameter(GetFullParmName("BiasTargetX"), "Length");
	if (fPm->ParameterExists(GetFullParmName("BiasTargetY")))
		fTargetY = fPm->GetDoubleParameter(GetFullParmName("BiasTargetY"), "Length");
	if (fPm->ParameterExists(GetFullParmName("BiasTargetZ")))
		fTargetZ = fPm->GetDoubleParameter(GetFullParmName("BiasTargetZ"), "Length");

	if (fDiscRadius <= 0. || fBiasFraction < 0. || fBiasFraction >= 1. ||
		(fBiasFraction > 0. && (fBiasRadius <= 0. || fBiasRadius > fDiscRadius))) {
		std::ostringstream msg;
		msg << "Source " << GetName() << ": need DiscRadius > 0, 0 <= BiasFraction < 1 and "
			<< "0 < BiasRadius <= DiscRadius (got R=" << fDiscRadius / cm << " cm, f="
			<< fBiasFraction << ", r_b=" << fBiasRadius / cm << " cm)";
		G4Exception("WeightedDiscBeam::ResolveParameters", "WDB001", FatalException,
					msg.str().c_str());
	}

	// p_true = 1/(pi R^2) everywhere on the disc; q = (1-f) p_true + f/(pi r_b^2)
	// inside the shadow circle, (1-f) p_true outside.
	const G4double area_ratio = (fBiasFraction > 0.)
		? (fDiscRadius * fDiscRadius) / (fBiasRadius * fBiasRadius) : 0.;
	fWeightOutside = 1. / (1. - fBiasFraction);
	fWeightInside = 1. / ((1. - fBiasFraction) + fBiasFraction * area_ratio);

	fShadowValid = false;
}


void WeightedDiscBeam::CacheGeometryPointers()
{
	TsVGenerator::CacheGeometryPointers();
	fShadowValid = false;             // component may have moved
}


// Find the target's projection onto the disc plane in component-local coords by
// pushing probe primaries through the SAME local->world transform the Beam source
// uses (TransformPrimaryForComponent), so no rotation convention is assumed.
void WeightedDiscBeam::ComputeShadow()
{
	const G4double L = 1. * m;
	TsPrimaryParticle o = {}, px = {}, py = {};
	o.posX = 0.; o.posY = 0.; o.posZ = 0.;
	o.dCos1 = 0.; o.dCos2 = 0.; o.dCos3 = 1.;
	px = o; px.posX = L;
	py = o; py.posY = L;
	TransformPrimaryForComponent(&o);
	TransformPrimaryForComponent(&px);
	TransformPrimaryForComponent(&py);

	const G4ThreeVector O(o.posX, o.posY, o.posZ);
	const G4ThreeVector ex = (G4ThreeVector(px.posX, px.posY, px.posZ) - O).unit();
	const G4ThreeVector ey = (G4ThreeVector(py.posX, py.posY, py.posZ) - O).unit();
	const G4ThreeVector d = G4ThreeVector(o.dCos1, o.dCos2, o.dCos3).unit();
	const G4ThreeVector T(fTargetX, fTargetY, fTargetZ);

	if (std::fabs(ex.dot(ey)) > 1e-4 || std::fabs(ex.dot(d)) > 1e-4 ||
		std::fabs(ey.dot(d)) > 1e-4) {
		G4Exception("WeightedDiscBeam::ComputeShadow", "WDB002", FatalException,
					"component frame is not orthonormal after transform");
	}
	const G4ThreeVector v = T - O;
	fShadowX = v.dot(ex);
	fShadowY = v.dot(ey);

	if (fBiasFraction > 0. &&
		std::hypot(fShadowX, fShadowY) + fBiasRadius > fDiscRadius) {
		std::ostringstream msg;
		msg << "Source " << GetName() << ": shadow circle (centre "
			<< std::hypot(fShadowX, fShadowY) / cm << " cm off-axis, radius "
			<< fBiasRadius / cm << " cm) does not fit inside the disc (R="
			<< fDiscRadius / cm << " cm)";
		G4Exception("WeightedDiscBeam::ComputeShadow", "WDB003", FatalException,
					msg.str().c_str());
	}
	fShadowValid = true;
}


void WeightedDiscBeam::GeneratePrimaries(G4Event* anEvent)
{
	if (CurrentSourceHasGeneratedEnough())
		return;

	if (!fShadowValid)
		ComputeShadow();

	G4double x, y;
	if (fBiasFraction > 0. && G4UniformRand() < fBiasFraction) {
		const G4double r = fBiasRadius * std::sqrt(G4UniformRand());
		const G4double a = twopi * G4UniformRand();
		x = fShadowX + r * std::cos(a);
		y = fShadowY + r * std::sin(a);
	} else {
		const G4double r = fDiscRadius * std::sqrt(G4UniformRand());
		const G4double a = twopi * G4UniformRand();
		x = r * std::cos(a);
		y = r * std::sin(a);
	}
	const G4double dx = x - fShadowX, dy = y - fShadowY;
	const G4bool inside = fBiasFraction > 0. && (dx * dx + dy * dy) < fBiasRadius * fBiasRadius;

	TsPrimaryParticle p = {};
	p.posX = x;
	p.posY = y;
	p.posZ = 0.;
	p.dCos1 = 0.;
	p.dCos2 = 0.;
	p.dCos3 = 1.;
	p.weight = inside ? fWeightInside : fWeightOutside;
	p.isNewHistory = true;
	SetEnergy(p);
	SetParticleType(p);
	TransformPrimaryForComponent(&p);
	GenerateOnePrimary(anEvent, p);
	AddPrimariesToEvent(anEvent);
}
