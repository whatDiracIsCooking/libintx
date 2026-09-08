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
  /// This is a conventional four-centre engine: no auxiliary basis, no V^-1
  /// transform. The Fock matrix of a closed-shell SCF is F = H + J - K/2 with
  /// J from a JEngine and K from here, both contracted against the same
  /// D = 2 C_occ C_occ^T. libintx::md::make_jengine is the conventional J
  /// engine that pairs with this one exactly -- same driver, same integrals,
  /// no fitting error to reconcile.
  struct KEngine {

    using TileIndex = std::pair<size_t,size_t>;
    using TileIn = std::function<bool(TileIndex, TileIndex, double*)>;
    using TileOut = std::function<bool(TileIndex, TileIndex, const double*)>;
    using AllSum = std::function<void(double*, size_t)>;

    /// Shell-pair magnitude bounds. A conventional K build has no auxiliary
    /// bra, so this is libintx::PairScreening itself rather than a K-specific
    /// interface: the same object screens J, K, or both at once.
    using Screening = libintx::PairScreening;

    virtual ~KEngine() = default;

    /// Contract the density read through @p D into the exchange matrix written
    /// through @p K. @p AllSum reduces the engine's K over whatever the caller
    /// distributes across, before the tiles are handed out; an empty AllSum is
    /// a no-op.
    virtual void K(const TileIn &D, const TileOut &K, const AllSum&) = 0;

  };

}

#endif /* LIBINTX_KENGINE_H */
