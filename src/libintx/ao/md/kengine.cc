#include "libintx/ao/md/kengine.h"
#include "libintx/ao/md/engine.h"
#include "libintx/gpu/kengine/md/driver.h"

namespace libintx::md {

  namespace {

    using kengine::md::HostBuffer;

    struct KEngine : libintx::KEngine {

      KEngine(
        const Basis<Gaussian> &basis,
        std::shared_ptr<const Screening> screening,
        int num_threads)
        : num_threads(num_threads),
          basis_(basis),
          shared_basis_(std::make_shared< Basis<Gaussian> >(basis)),
          screening_(screening)
      {
        auto norm2 = [&](int i, int j) -> float {
          return (screening_ ? screening_->max2(i,j) : 1.0f);
        };
        classes_ = kengine::md::make_pair_classes(basis_, norm2);
      }

      void K(const TileIn &D, const TileOut &K, const AllSum &allsum) override {
        namespace kmd = kengine::md;
        auto d = kmd::gather_density(basis_, D);
        kmd::Matrix k(basis_.nbf());

        libintx::md::IntegralEngine<4> engine(shared_basis_);
        engine.num_threads = this->num_threads;

        HostBuffer buffer;
        kmd::build(
          basis_, classes_, engine, buffer, d, screening_.get(), max_batch, k
        );

        if (allsum) allsum(k.data.data(), k.data.size());
        kmd::scatter_exchange(basis_, k, K);
      }

      /// OpenMP threads handed to the four-centre engine.
      int num_threads = 1;
      /// Doubles in one integral batch; the pair chunking is derived from it.
      size_t max_batch = 4*1024*1024;

    private:
      Basis<Gaussian> basis_;
      std::shared_ptr< Basis<Gaussian> > shared_basis_;
      std::shared_ptr<const Screening> screening_;
      std::vector<kengine::md::PairClass> classes_;

    };

  } // namespace

  std::unique_ptr<libintx::KEngine> make_kengine(
    const Basis<Gaussian> &basis,
    std::shared_ptr<const libintx::KEngine::Screening> screening,
    int num_threads)
  {
    return std::make_unique<KEngine>(basis, screening, num_threads);
  }

  std::shared_ptr<const libintx::KEngine::Screening> make_schwarz_screening(
    const Basis<Gaussian> &basis,
    float threshold)
  {
    auto shared_basis = std::make_shared< Basis<Gaussian> >(basis);
    IntegralEngine<4> engine(shared_basis);
    HostBuffer buffer;
    return kengine::md::schwarz_screening(basis, engine, buffer, threshold);
  }

}
