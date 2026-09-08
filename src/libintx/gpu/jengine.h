#ifndef LIBINTX_GPU_JENGINE_H
#define LIBINTX_GPU_JENGINE_H

#include "libintx/jengine.h"
#include "libintx/screening.h"
#include "libintx/gpu/forward.h"
#include "libintx/gpu/screening.h"
#include <memory>

namespace libintx::gpu {

  /// The **density-fitted** Coulomb engine: two three-centre passes around an
  /// AllSum on the fitted vector X_Q, so it never forms a four-index
  /// (mu nu | lambda sigma). It needs an auxiliary basis and a caller-supplied
  /// metric solve @p V_linv, and it carries the fitting error of that
  /// auxiliary basis.
  ///
  /// Lives in libintx.gpu.jengine (src/libintx/gpu/jengine/md/).
  std::unique_ptr<libintx::JEngine> make_df_jengine(
    const Basis<Gaussian> &basis,
    const Basis<Gaussian> &df_basis,
    std::function<void(double*)> V_linv,
    std::shared_ptr<const libintx::JEngine::Screening> screening = nullptr
  );

  /// The former name of make_df_jengine, kept so existing callers keep
  /// compiling now that there are two GPU J engines to tell apart. Prefer
  /// make_df_jengine at new call sites.
  inline std::unique_ptr<libintx::JEngine> make_jengine(
    const Basis<Gaussian> &basis,
    const Basis<Gaussian> &df_basis,
    std::function<void(double*)> V_linv,
    std::shared_ptr<const libintx::JEngine::Screening> screening = nullptr)
  {
    return make_df_jengine(basis, df_basis, V_linv, screening);
  }

  /// The **conventional** (integral-direct) Coulomb engine: four centres, no
  /// auxiliary basis, no metric, no fitting error. Same libintx::JEngine
  /// interface as the DF engine above, so a caller swaps one for the other by
  /// changing the factory call.
  ///
  /// This is the same build as libintx::md::make_jengine on
  /// libintx::gpu::md::IntegralEngine<4>, and the same driver
  /// (libintx/fock/md/driver.h) as libintx::gpu::make_kengine with the other
  /// scatter -- so it is the J that pairs exactly with the conventional K.
  /// Lives in libintx.gpu.md4 (src/libintx/gpu/md/jengine.cc), next to the
  /// device K engine rather than in the DF engine's tree.
  ///
  /// As with the device K engine the digest runs on the host: the device
  /// produces the (ab|cd) batch into host-registered memory and the driver
  /// contracts it.
  std::unique_ptr<libintx::JEngine> make_jengine_direct(
    const Basis<Gaussian> &basis,
    std::shared_ptr<const libintx::PairScreening> screening = nullptr,
    gpuStream_t stream = 0
  );

}

#endif /* LIBINTX_GPU_JENGINE_H */
