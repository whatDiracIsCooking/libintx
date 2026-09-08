#include "libintx/ao/md/kengine.h"
#include "libintx/ao/md/engine.h"
#include "libintx/fock/md/df.h"

namespace libintx::md {

  namespace {

    /// The host density-fitted K engine. Everything but the engine type and
    /// the buffer is in libintx/fock/md/df.h, which the device engine drives
    /// the same way.
    struct DFKEngine : libintx::KEngine {

      DFKEngine(
        const Basis<Gaussian> &basis,
        const Basis<Gaussian> &df_basis,
        MetricTransform v_linv,
        std::shared_ptr<const libintx::PairScreening> screening,
        int num_threads)
        : num_threads(num_threads),
          basis_(basis),
          df_basis_(df_basis),
          shared_basis_(std::make_shared< Basis<Gaussian> >(basis)),
          shared_df_basis_(std::make_shared< Basis<Gaussian> >(df_basis)),
          v_linv_(std::move(v_linv)),
          screening_(screening)
      {
        auto norm2 = [&](int i, int j) -> float {
          return (screening_ ? screening_->max2(i,j) : 1.0f);
        };
        classes_ = fock::md::make_pair_classes(basis_, norm2);
        aux_classes_ = fock::md::df::make_aux_classes(df_basis_);
      }

      void K(const TileIn &D, const TileOut &K, const AllSum &allsum) override {
        namespace fmd = fock::md;
        auto d = fmd::gather_density(basis_, D);
        fmd::Matrix k(basis_.nbf());

        // Bra is the auxiliary basis, ket is the AO basis on both slots.
        libintx::md::IntegralEngine<3> engine(shared_df_basis_, shared_basis_);
        engine.num_threads = this->num_threads;

        fmd::HostBuffer buffer;
        fmd::df::build(
          basis_, df_basis_, aux_classes_, classes_, engine, buffer,
          d, v_linv_, screening_.get(), max_batch, k
        );

        if (allsum) allsum(k.data.data(), k.data.size());
        fmd::scatter_matrix(basis_, k, K);
      }

      /// OpenMP threads handed to the three-centre engine.
      int num_threads = 1;
      /// Doubles in one integral batch; the batching is derived from it.
      size_t max_batch = 4*1024*1024;

    private:
      Basis<Gaussian> basis_, df_basis_;
      std::shared_ptr< Basis<Gaussian> > shared_basis_, shared_df_basis_;
      MetricTransform v_linv_;
      std::shared_ptr<const libintx::PairScreening> screening_;
      std::vector<fock::md::PairClass> classes_;
      std::vector<fock::md::df::AuxClass> aux_classes_;

    };

  } // namespace

  std::unique_ptr<libintx::KEngine> make_df_kengine(
    const Basis<Gaussian> &basis,
    const Basis<Gaussian> &df_basis,
    libintx::KEngine::MetricTransform V_linv,
    std::shared_ptr<const libintx::PairScreening> screening,
    int num_threads)
  {
    return std::make_unique<DFKEngine>(
      basis, df_basis, std::move(V_linv), screening, num_threads
    );
  }

}
