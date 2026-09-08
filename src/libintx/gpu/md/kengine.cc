#include "libintx/gpu/kengine.h"
#include "libintx/gpu/md/engine.h"
#include "libintx/gpu/md/buffer.h"
#include "libintx/fock/md/driver.h"

namespace libintx::gpu::md {

  namespace {

    struct KEngine : libintx::KEngine {

      KEngine(
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

      void K(const TileIn &D, const TileOut &K, const AllSum &allsum) override {
        namespace fmd = fock::md;
        auto d = fmd::gather_density(basis_, D);
        fmd::Matrix k(basis_.nbf());

        IntegralEngine<4> engine(basis_, basis_, stream_);
        DeviceBuffer buffer(stream_);
        fmd::build(
          basis_, classes_, engine, buffer, screening_.get(), max_batch,
          fmd::exchange(d, k)
        );

        if (allsum) allsum(k.data.data(), k.data.size());
        fmd::scatter_matrix(basis_, k, K);
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

  std::unique_ptr<libintx::KEngine> make_kengine(
    const Basis<Gaussian> &basis,
    std::shared_ptr<const libintx::PairScreening> screening,
    gpuStream_t stream)
  {
    return std::make_unique<KEngine>(basis, screening, stream);
  }

} // libintx::gpu::md

std::unique_ptr<libintx::KEngine> libintx::gpu::make_kengine(
  const Basis<Gaussian> &basis,
  std::shared_ptr<const libintx::PairScreening> screening,
  gpuStream_t stream)
{
  return md::make_kengine(basis, screening, stream);
}
