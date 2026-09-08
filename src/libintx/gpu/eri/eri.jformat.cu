// -*-c++-*-

// this must come first to resolve HIP device asserts
#include "libintx/gpu/api/runtime.h"

#include "libintx/gpu/eri/format.h"

/// The J-format kernel.
///
///   G[(mu,nu),(lambda,sigma)] = (mu nu | lambda sigma),  munu = mu*nbf + nu
///
/// which is the ERI's own pairing of its axes, so the Coulomb matrix falls out
/// as J = G . vec(D). Everything but that one index expression -- JFormat, in
/// eri/format.h -- is shared with the K format; this translation unit is where
/// the scatter kernel is instantiated for it.
namespace libintx::gpu::eri {

  void jformat(const Basis<Gaussian> &basis, double *G, gpuStream_t stream) {
    build<JFormat>(basis, G, stream);
  }

}
