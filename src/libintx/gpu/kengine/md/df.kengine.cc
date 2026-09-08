#include "libintx/gpu/kengine.h"
#include "libintx/gpu/md/engine.h"
#include "libintx/gpu/api/api.h"
#include "libintx/gpu/kengine/md/buffer.h"
#include "libintx/gpu/kengine/md/df.h"

namespace libintx::gpu::md {

  namespace {

    /// The device density-fitted K engine: libintx/gpu/kengine/md/df.h driven
    /// by the device three-centre engine, with the same DeviceBuffer the
    /// integral-direct device engine uses. Everything but those two choices
    /// is shared with libintx::md::make_df_kengine.
    struct DFKEngine : libintx::KEngine {

      DFKEngine(
        const Basis<Gaussian> &basis,
        const Basis<Gaussian> &df_basis,
        MetricTransform v_linv,
        std::shared_ptr<const Screening> screening,
        gpuStream_t stream)
        : basis_(basis),
          df_basis_(df_basis),
          v_linv_(std::move(v_linv)),
          screening_(screening),
          stream_(stream)
      {
        auto norm2 = [&](int i, int j) -> float {
          return (screening_ ? screening_->max2(i,j) : 1.0f);
        };
        classes_ = kengine::md::make_pair_classes(basis_, norm2);
        aux_classes_ = kengine::md::df::make_aux_classes(df_basis_);
      }

      void K(const TileIn &D, const TileOut &K, const AllSum &allsum) override {
        namespace kmd = kengine::md;
        auto d = kmd::gather_density(basis_, D);
        kmd::Matrix k(basis_.nbf());

        // Bra is the auxiliary basis, ket is the AO basis on both slots.
        IntegralEngine<3> engine(df_basis_, basis_, stream_);
        DeviceBuffer buffer(stream_);
        kmd::df::build(
          basis_, df_basis_, aux_classes_, classes_, engine, buffer,
          d, v_linv_, screening_.get(), max_batch, k
        );

        if (allsum) allsum(k.data.data(), k.data.size());
        kmd::scatter_exchange(basis_, k, K);
      }

      /// Doubles in one integral batch. Smaller than the host engine's: this
      /// buffer is pinned host memory the device writes through, and pinning
      /// is a scarcer resource than plain heap.
      size_t max_batch = 1024*1024;

    private:
      Basis<Gaussian> basis_, df_basis_;
      MetricTransform v_linv_;
      std::shared_ptr<const Screening> screening_;
      gpuStream_t stream_;
      std::vector<kengine::md::PairClass> classes_;
      std::vector<kengine::md::df::AuxClass> aux_classes_;

    };

  } // namespace

  std::unique_ptr<libintx::KEngine> make_df_kengine(
    const Basis<Gaussian> &basis,
    const Basis<Gaussian> &df_basis,
    libintx::KEngine::MetricTransform v_linv,
    std::shared_ptr<const libintx::KEngine::Screening> screening,
    gpuStream_t stream)
  {
    return std::make_unique<DFKEngine>(
      basis, df_basis, std::move(v_linv), screening, stream
    );
  }

} // libintx::gpu::md

std::unique_ptr<libintx::KEngine> libintx::gpu::make_df_kengine(
  const Basis<Gaussian> &basis,
  const Basis<Gaussian> &df_basis,
  libintx::KEngine::MetricTransform V_linv,
  std::shared_ptr<const libintx::KEngine::Screening> screening,
  gpuStream_t stream)
{
  return md::make_df_kengine(
    basis, df_basis, std::move(V_linv), screening, stream
  );
}
