#ifndef LIBINTX_GPU_KENGINE_H
#define LIBINTX_GPU_KENGINE_H

#include "libintx/kengine.h"
#include "libintx/shell.h"
#include "libintx/gpu/forward.h"

#include <memory>

namespace libintx::gpu {

  /// Device McMurchie-Davidson exchange engine: the same K build as
  /// libintx::md::make_kengine, driving libintx::gpu::md::IntegralEngine<4>
  /// instead of the host one. Both share libintx/kengine/md/driver.h -- the
  /// shell-pair binning, the screening and the eight-fold digest are one piece
  /// of code -- so the host engine is the reference this is checked against.
  ///
  /// The digest itself is on the host: the device produces the (ab|cd) batch
  /// into host-registered memory and the driver contracts it. That is the
  /// straightforward split, not the final one; a device-side digest is the
  /// obvious next step and does not change this interface.
  ///
  /// The primitive-normalization note on libintx::md::make_kengine applies
  /// here verbatim -- K, H_core and J have to agree on the convention.
  std::unique_ptr<libintx::KEngine> make_kengine(
    const Basis<Gaussian> &basis,
    std::shared_ptr<const libintx::KEngine::Screening> screening = nullptr,
    gpuStream_t stream = 0
  );

  /// Schwarz bounds sqrt(max |(ij|ij)|) for every canonical shell pair,
  /// evaluated on the device. Same contract as
  /// libintx::md::make_schwarz_screening, and the two are interchangeable --
  /// a Screening is plain host data once it is built.
  std::shared_ptr<const libintx::KEngine::Screening> make_schwarz_screening(
    const Basis<Gaussian> &basis,
    float threshold,
    gpuStream_t stream = 0
  );

}

#endif /* LIBINTX_GPU_KENGINE_H */
