#include "libintx/ao/md/jengine.h"
#include "libintx/ao/md/engine.h"
#include "libintx/fock/md/driver.h"

namespace libintx::md {

  namespace {

    struct JEngine : libintx::JEngine {

      JEngine(
        const Basis<Gaussian> &basis,
        std::shared_ptr<const libintx::PairScreening> screening,
        int num_threads)
        : num_threads(num_threads),
          basis_(basis),
          shared_basis_(std::make_shared< Basis<Gaussian> >(basis)),
          screening_(screening)
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

        libintx::md::IntegralEngine<4> engine(shared_basis_);
        engine.num_threads = this->num_threads;

        fmd::HostBuffer buffer;
        fmd::build(
          basis_, classes_, engine, buffer, screening_.get(), max_batch,
          fmd::coulomb(d, j)
        );

        if (allsum) allsum(j.data.data(), j.data.size());
        fmd::scatter_matrix(basis_, j, J);
      }

      /// OpenMP threads handed to the four-centre engine.
      int num_threads = 1;
      /// Doubles in one integral batch; the pair chunking is derived from it.
      size_t max_batch = 4*1024*1024;

    private:
      Basis<Gaussian> basis_;
      std::shared_ptr< Basis<Gaussian> > shared_basis_;
      std::shared_ptr<const libintx::PairScreening> screening_;
      std::vector<fock::md::PairClass> classes_;

    };

  } // namespace

  std::unique_ptr<libintx::JEngine> make_jengine(
    const Basis<Gaussian> &basis,
    std::shared_ptr<const libintx::PairScreening> screening,
    int num_threads)
  {
    return std::make_unique<JEngine>(basis, screening, num_threads);
  }

}
