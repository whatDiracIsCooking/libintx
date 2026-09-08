#ifndef LIBINTX_GPU_ERI_H
#define LIBINTX_GPU_ERI_H

#include "libintx/shell.h"
#include "libintx/gpu/forward.h"

#include <cstddef>

/// Full-ERI materialisation: the whole four-index tensor written out as an
/// (nbf^2 x nbf^2) matrix, so that a Fock build's J and K each become one GEMV
/// against the vectorised density.
///
/// This is the small-system path, and deliberately so. nbf^4 doubles is 134 MB
/// at nbf=64, 800 MB at 100, 2.1 GB at 128 and 34 GB at 256, so past a few
/// hundred basis functions the integral-direct engines (libintx::gpu::JEngine,
/// libintx::gpu::make_kengine) are the only option. What it buys below that is
/// an SCF whose J and K builds are a GEMV over a tensor computed once per
/// geometry rather than an integral pass per iteration.
///
/// The basis must carry primitive-normalized contraction coefficients -- i.e.
/// come from libintx::make_basis(..., normalize=true) -- for the same reason
/// the K engine requires it: a basis-set library's coefficients are defined
/// against normalized primitives, and for a contracted shell that is not an
/// overall per-basis-function scale.
namespace libintx::gpu::eri {

  /// Doubles in a format buffer: nbf^2 * nbf^2.
  ///
  /// Both formats are this size because they are the same tensor, differing
  /// only in which axes are paired into the row index.
  inline size_t format_size(const Basis<Gaussian> &basis) {
    const size_t n = basis.nbf()*basis.nbf();
    return n*n;
  }

  /// Whether format_size(basis) doubles fit in currently-free device memory,
  /// leaving `reserve` bytes for everything else that has to live there -- the
  /// integral engine's scratch, or a second format buffer.
  ///
  /// The caller owns the buffer, so this is advice rather than enforcement. It
  /// exists because nbf^4 grows fast enough that "will this fit" is the first
  /// question to ask of this whole approach, and because the alternative is an
  /// allocation failure that says nothing about why.
  bool format_fits(const Basis<Gaussian> &basis, size_t reserve = 0);

  /// Materialise the full ERI tensor into @p G as an (nbf^2 x nbf^2) matrix
  /// laid out for J = G . vec(D):
  ///
  ///   G[(mu,nu),(lambda,sigma)] = (mu nu | lambda sigma)
  ///
  /// with the composite index munu = mu*nbf + nu, and vec(D) the density in
  /// that same ordering, so that
  ///
  ///   J[mu,nu] = sum_{lambda,sigma} (mu nu|lambda sigma) D[lambda,sigma]
  ///
  /// @p G is device memory the caller owns and must hold format_size(basis)
  /// doubles. It is zeroed and then filled completely: nothing here screens,
  /// because a dropped quartet would leave a zero the GEMV cannot tell from a
  /// real one.
  ///
  /// The result is symmetric -- (mu nu|lambda sigma) = (lambda sigma|mu nu) --
  /// so a row-major and a column-major reading agree and dsymv applies.
  ///
  /// The call is issued on @p stream and synchronises it before returning.
  void jformat(const Basis<Gaussian> &basis, double *G, gpuStream_t stream = 0);

  /// Materialise the full ERI tensor into @p G as an (nbf^2 x nbf^2) matrix
  /// laid out for K = G . vec(D):
  ///
  ///   G[(mu,nu),(lambda,sigma)] = (mu lambda | nu sigma)
  ///
  /// with the same composite index munu = mu*nbf + nu as jformat(), so that
  ///
  ///   K[mu,nu] = sum_{lambda,sigma} (mu lambda|nu sigma) D[lambda,sigma]
  ///
  /// This is not new data -- it is jformat()'s tensor with axes 1 and 2
  /// transposed,
  ///
  ///   G_K[(mu,nu),(lambda,sigma)] = G_J[(mu,lambda),(nu,sigma)]
  ///
  /// -- but it cannot be read out of that buffer *as a GEMV*: the row K needs
  /// is scattered through G_J with stride nbf in one index and 1 in another,
  /// which is exactly what a GEMV cannot express. Writing the permutation at
  /// scatter time costs nothing, since the kernel is already choosing a
  /// destination per value, and buys a contiguous GEMV on every iteration
  /// after. That is what makes this a separate format rather than a second
  /// reading of one buffer.
  ///
  /// The contract is otherwise jformat()'s exactly: @p G is caller-owned
  /// device memory of format_size(basis) doubles, zeroed and then filled
  /// completely, and the result is symmetric -- the transpose of
  /// (mu lambda|nu sigma) is (lambda mu|sigma nu), equal to it by the
  /// within-pair symmetries -- so dsymv applies here too.
  void kformat(const Basis<Gaussian> &basis, double *G, gpuStream_t stream = 0);

}

#endif /* LIBINTX_GPU_ERI_H */
