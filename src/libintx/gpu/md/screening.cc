#include "libintx/gpu/screening.h"
#include "libintx/gpu/md/engine.h"
#include "libintx/gpu/md/buffer.h"
#include "libintx/fock/md/driver.h"

namespace libintx::gpu::md {

  std::shared_ptr<const libintx::PairScreening> make_schwarz_screening(
    const Basis<Gaussian> &basis,
    float threshold,
    gpuStream_t stream)
  {
    IntegralEngine<4> engine(basis, basis, stream);
    DeviceBuffer buffer(stream);
    return fock::md::schwarz_screening(basis, engine, buffer, threshold);
  }

} // libintx::gpu::md

std::shared_ptr<const libintx::PairScreening>
libintx::gpu::make_schwarz_screening(
  const Basis<Gaussian> &basis,
  float threshold,
  gpuStream_t stream)
{
  return md::make_schwarz_screening(basis, threshold, stream);
}
