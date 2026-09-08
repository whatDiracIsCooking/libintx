#ifndef LIBINTX_JENGINE_H
#define LIBINTX_JENGINE_H

#include "libintx/forward.h"
#include "libintx/screening.h"
#include <functional>

namespace libintx {

  /// Coulomb-matrix engine.
  ///
  ///   J[mu,nu] = sum_{lambda,sigma} (mu nu | lambda sigma) D[lambda,sigma]
  ///
  /// The interface says nothing about *how* J is built: the density-fitted
  /// engine (libintx::gpu::make_df_jengine) and the conventional,
  /// integral-direct ones (libintx::md::make_jengine,
  /// libintx::gpu::make_jengine_direct) both implement it. D is read and J is
  /// written through shell-block tile callbacks so a distributed caller never
  /// has to materialise either matrix.
  struct JEngine {

    using TileIndex = std::pair<size_t,size_t>;
    using TileIn = std::function<bool(TileIndex, TileIndex, double*)>;
    using TileOut = std::function<bool(TileIndex, TileIndex, const double*)>;
    using AllSum = std::function<void(double*, size_t)>;

    struct Screening;

    virtual ~JEngine() = default;
    virtual void J(const TileIn &D, const TileOut &J, const AllSum&) = 0;

  };

  /// The density-fitted engine's screening: pair bounds plus the
  /// per-auxiliary-shell bound the fitting basis needs.
  ///
  /// A conventional J or K build has no auxiliary bra, so it screens on
  /// libintx::PairScreening -- this class's base -- and a Screening built for
  /// the DF engine is usable there unchanged.
  struct JEngine::Screening : PairScreening {
    //virtual float max1() const = 0;
    virtual float max1(int) const = 0;
  };

}

#endif /* LIBINTX_JENGINE_H */
