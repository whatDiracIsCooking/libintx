#include "libintx/gpu/jengine.h"
#include "libintx/gpu/md/engine.h"
#include "libintx/gpu/md/buffer.h"
#include "libintx/fock/md/driver.h"

namespace libintx::gpu::md {

  namespace {

    /// The conventional (integral-direct) device J engine. Everything but the
    /// scatter is shared with the device K engine next door, and with both
    /// host engines: one driver, four call sites.
    struct JEngine : libintx::JEngine {

      JEngine(
        const Basis<Gaussian> &basis,
        std::shared_ptr<const libintx::PairScreening> screening,
        gpuStream_t stream)
        : basis_(basis), screening_(screening), stream_(stream)
      {
        auto norm2 = [&](int i, int j) -> float {
          return (screening_ ? screening_->max2(i,j) : 1.0f);
        };
        classes_ = fock::md::make_pair_classes(basis_, norm2);
      }

      void J(const TileIn &D, const TileOut &J, const AllSum &allsum) override {
        namespace fmd = fock::md;
        auto d = fmd::gather_density(basis_, D);
        fmd::Matrix j(basis_.nbf());

        IntegralEngine<4> engine(basis_, basis_, stream_);
        DeviceBuffer buffer(stream_);
        fmd::build(
          basis_, classes_, engine, buffer, screening_.get(), max_batch,
          fmd::coulomb(d, j)
        );

        if (allsum) allsum(j.data.data(), j.data.size());
        fmd::scatter_matrix(basis_, j, J);
      }

      /// Doubles in one integral batch. Smaller than the host engine's: this
      /// buffer is pinned host memory the device writes through, and pinning
      /// is a scarcer resource than plain heap.
      size_t max_batch = 1024*1024;

    private:
      Basis<Gaussian> basis_;
      std::shared_ptr<const libintx::PairScreening> screening_;
      gpuStream_t stream_;
      std::vector<fock::md::PairClass> classes_;

    };

  } // namespace

  std::unique_ptr<libintx::JEngine> make_jengine_direct(
    const Basis<Gaussian> &basis,
    std::shared_ptr<const libintx::PairScreening> screening,
    gpuStream_t stream)
  {
    return std::make_unique<JEngine>(basis, screening, stream);
  }

} // libintx::gpu::md

std::unique_ptr<libintx::JEngine> libintx::gpu::make_jengine_direct(
  const Basis<Gaussian> &basis,
  std::shared_ptr<const libintx::PairScreening> screening,
  gpuStream_t stream)
{
  return md::make_jengine_direct(basis, screening, stream);
}
