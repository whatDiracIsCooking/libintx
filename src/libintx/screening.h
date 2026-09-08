#ifndef LIBINTX_SCREENING_H
#define LIBINTX_SCREENING_H

#include "libintx/forward.h"

namespace libintx {

  /// Shell-pair magnitude bounds -- the screening input a conventional
  /// four-centre build takes.
  ///
  /// `max2(i,j)` is the Schwarz bound of the (i,j) shell pair,
  /// sqrt(max |(ij|ij)|), and `max()` the largest of them. A quartet is
  /// dropped when its bound -- max2(i,j)*max2(k,l) times the relevant block of
  /// the density -- fails `skip()`.
  ///
  /// This is the shape JEngine::Screening and KEngine::Screening had in
  /// common, hoisted so that the conventional J engine, the K engine and the
  /// density-fitted J engine all screen through one type and one
  /// `make_schwarz_screening` result serves all three. The DF engine's
  /// per-auxiliary-shell bound `max1(int)` is the part that is *not* shared:
  /// it means nothing without an auxiliary basis, so it stays on
  /// JEngine::Screening.
  struct PairScreening {
    virtual ~PairScreening() {}
    virtual float max2(int,int) const = 0;
    virtual float max() const = 0;
    virtual bool skip(float) const = 0;
  };

}

#endif /* LIBINTX_SCREENING_H */
