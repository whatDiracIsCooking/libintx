#ifndef LIBINTX_GPU_SCREENING_H
#define LIBINTX_GPU_SCREENING_H

#include "libintx/screening.h"
#include "libintx/shell.h"
#include "libintx/gpu/forward.h"

#include <memory>

namespace libintx::gpu {

  /// Schwarz bounds sqrt(max |(ij|ij)|) for every canonical shell pair,
  /// evaluated on the device. Same contract as
  /// libintx::md::make_schwarz_screening, and the two results are
  /// interchangeable -- a PairScreening is plain host data once it is built,
  /// and it screens the conventional J engine, the K engine, or both at once.
  std::shared_ptr<const libintx::PairScreening> make_schwarz_screening(
    const Basis<Gaussian> &basis,
    float threshold,
    gpuStream_t stream = 0
  );

}

#endif /* LIBINTX_GPU_SCREENING_H */
