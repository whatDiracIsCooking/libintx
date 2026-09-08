#ifndef LIBINTX_GPU_ONEBODY_BASIS_H
#define LIBINTX_GPU_ONEBODY_BASIS_H

#include "libintx/ao/engine.h"
#include "libintx/shell.h"
#include "libintx/gpu/forward.h"
#include "libintx/gpu/api/api.h"
#include "libintx/gpu/md/basis.h"

#include <vector>

// The 2-centre device batch.
//
// The Coulomb path's gpu::md::make_basis is the wrong shape for the
// one-electron operators: it bakes E into a [ab,p] buffer indexed over
// nherm2(A+B), because that is what md3/md4 consume. Overlap needs only
// E(a,b,0), kinetic needs E at (A,B+2), and only the nuclear kernel wants the
// full nherm2(A+B) index -- so no single baked buffer serves all three without
// waste. What goes to the device here is the raw primitive pairs; each
// operator's kernel builds the slice of E it wants in shared memory, one
// primitive pair at a time (see gpu/onebody/kernel.h).

namespace libintx::gpu::md {

  /// A batch of 2-centre shell pairs, uploaded as raw primitives.
  ///
  /// One `compute` call is one bin, the same invariant as everywhere else in
  /// the tree: the angular momentum, the solid-harmonic flag AND the
  /// contraction degree `K = K_a*K_b` agree across every pair in the batch.
  /// Nothing pads. `make_basis` below checks it rather than letting a mixed
  /// batch produce garbage.
  struct GaussianPairs {
    /// The bin: angular momentum and solid-harmonic flag of the two shells.
    Shell first, second;
    /// Number of pairs in the batch.
    int N;
    /// Contraction degree `K_a*K_b`, identical for every pair.
    int K;
    /// Device pointer to `N` pairs.
    const Gaussian2 *data;
  };

  /// Device storage for a batch of `Gaussian2`.
  ///
  /// `Gaussian` carries default member initializers, so it is not trivially
  /// constructible and `device::vector<Gaussian2>` does not instantiate. The
  /// device only ever reads these -- it never constructs one -- so the batch
  /// is held as raw bytes and reinterpreted, the same way the Hermite data is
  /// (`Hermite::hdata`, `gpu/md/basis.h`).
  using Gaussian2Buffer = device::vector<char>;

  /// One point charge, as the device wants it.
  ///
  /// The host-side `Nuclear::Operator::Parameters` is a
  /// `std::vector< std::tuple<int, std::array<double,3> > >`; this is its
  /// device image. Only the electron-nuclear potential kernel reads it, but it
  /// is uploaded by the shared engine because it arrives through `set()`, once
  /// per geometry, not once per `compute` call.
  struct NuclearCenter {
    double Z;
    array<double,3> r;
  };

  /// Upload one bin of shell pairs and describe it.
  ///
  /// `ab` is the caller's device buffer, kept across calls so a `compute` loop
  /// does not reallocate per batch. Throws if `pairs` is empty or is not one
  /// bin.
  GaussianPairs make_basis(
    const Basis<Gaussian> &A,
    const Basis<Gaussian> &B,
    const std::vector<Index2> &pairs,
    Gaussian2Buffer &ab,
    gpuStream_t stream
  );

  /// Upload the point charges `set()` was given.
  void make_centers(
    const Nuclear::Operator::Parameters &params,
    device::vector<NuclearCenter> &centers,
    gpuStream_t stream
  );

}

#endif /* LIBINTX_GPU_ONEBODY_BASIS_H */
