// -*-c++-*-

// this must come first to resolve HIP device asserts
#include "libintx/gpu/api/runtime.h"

#include "libintx/gpu/eri/format.h"

/// The K-format kernel.
///
///   G[(mu,nu),(lambda,sigma)] = (mu lambda | nu sigma),  munu = mu*nbf + nu
///
/// the ERI's axes paired so that the exchange matrix falls out as
/// K = G . vec(D). This is the same tensor eri.jformat.cu writes, with axes 1
/// and 2 transposed; the transposition is applied where the scatter already
/// chooses a destination, because reading it back out of the J buffer would
/// cost the contiguity that makes the GEMV worth having.
///
/// Only the index expression -- KFormat, in eri/format.h -- differs. The
/// batching, the eight-fold orbit and the scatter are literally the J format's,
/// which is what the two formats' cross-check in the tests relies on.
namespace libintx::gpu::eri {

  void kformat(const Basis<Gaussian> &basis, double *G, gpuStream_t stream) {
    build<KFormat>(basis, G, stream);
  }

}
