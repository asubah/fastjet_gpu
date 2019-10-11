#ifndef PseudoJet_h
#define PseudoJet_h

using GridIndexType = int;
using ParticleIndexType = int;

struct PseudoJet {
  int index;
  bool isJet;
  double px;
  double py;
  double pz;
  double E;
};

struct PseudoJetExt{
  int index;
  bool isJet;
  double px;
  double py;
  double pz;
  double E;
  double rap;
  double phi;
  double diB;
  GridIndexType i;
  GridIndexType j;
};

struct Dist {
  double distance;
  ParticleIndexType i;
  ParticleIndexType j;
};

#endif  //  PseudoJet_h
