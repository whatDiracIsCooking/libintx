#include "libintx/ao/md/screening.h"
#include "libintx/ao/md/engine.h"
#include "libintx/fock/md/driver.h"

namespace libintx::md {

  std::shared_ptr<const libintx::PairScreening> make_schwarz_screening(
    const Basis<Gaussian> &basis,
    float threshold)
  {
    auto shared_basis = std::make_shared< Basis<Gaussian> >(basis);
    IntegralEngine<4> engine(shared_basis);
    fock::md::HostBuffer buffer;
    return fock::md::schwarz_screening(basis, engine, buffer, threshold);
  }

}
