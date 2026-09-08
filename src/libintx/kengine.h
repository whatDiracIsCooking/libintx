#ifndef LIBINTX_KENGINE_H
#define LIBINTX_KENGINE_H

#include "libintx/forward.h"
#include "libintx/screening.h"
#include <functional>

namespace libintx {

  /// Exchange-matrix engine, the K counterpart of libintx::JEngine.
  ///
  ///   K[mu,nu] = sum_{lambda,sigma} (mu lambda | nu sigma) D[lambda,sigma]
  ///
  /// D is read and K is written through the same shell-block tile callbacks
  /// JEngine uses, so a caller that already drives a J engine over a
  /// distributed matrix drives this one the same way: the tile indices are
  /// [first,last) basis-function ranges of a shell, and a TileOut called with
  /// a null pointer is a query ("do you want this tile at all?").
  ///
  /// Two families of engine implement this interface, additively. The
  /// integral-direct engines (libintx::md::make_kengine,
  /// libintx::gpu::make_kengine) are conventional four-centre builds: no
  /// auxiliary basis, no V^-1 transform. The density-fitted engines
  /// (libintx::md::make_df_kengine, libintx::gpu::make_df_kengine) factor the
  /// ERI through an auxiliary basis the way the DF JEngine does, so they take
  /// a df_basis and a MetricTransform and are exact only to the fitting
  /// error. Both answer K() identically otherwise; a caller picks one at
  /// construction and never sees the difference again.
  ///
  /// The Fock matrix of a closed-shell SCF is F = H + J - K/2 with J from a
  /// JEngine and K from here, both contracted against the same
  /// D = 2 C_occ C_occ^T. libintx::md::make_jengine is the conventional J
  /// engine that pairs with the direct K engine exactly -- same driver, same
  /// integrals, no fitting error to reconcile.
  struct KEngine {

    using TileIndex = std::pair<size_t,size_t>;
    using TileIn = std::function<bool(TileIndex, TileIndex, double*)>;
    using TileOut = std::function<bool(TileIndex, TileIndex, const double*)>;
    using AllSum = std::function<void(double*, size_t)>;

    /// Shell-pair magnitude bounds. A conventional K build has no auxiliary
    /// bra, so this is libintx::PairScreening itself rather than a K-specific
    /// interface: the same object screens J, K, or both at once. The
    /// density-fitted engines take it too, and get less out of it -- with no
    /// max1() there is no bound on the auxiliary bra to screen against.
    using Screening = libintx::PairScreening;

    /// The density-fitting metric transform, for the DF K engines only.
    ///
    /// Called as V_linv(X, n) with X a **row-major** naux x n matrix -- n
    /// columns of naux auxiliary-basis coefficients, consecutive rows n apart
    /// -- and must replace it in place with V^-1 X, where V[P,Q] = (P|Q) is
    /// the Coulomb metric over the auxiliary basis. That is the same map the
    /// DF JEngine's V_linv applies, generalised from one column to n; a DF K
    /// build has to transform the whole (P|mu nu) tensor, not a single vector.
    using MetricTransform = std::function<void(double*, size_t)>;

    virtual ~KEngine() = default;

    /// Contract the density read through @p D into the exchange matrix written
    /// through @p K. @p AllSum reduces the engine's K over whatever the caller
    /// distributes across, before the tiles are handed out; an empty AllSum is
    /// a no-op.
    virtual void K(const TileIn &D, const TileOut &K, const AllSum&) = 0;

  };

}

#endif /* LIBINTX_KENGINE_H */
