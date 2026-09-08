#include "libintx/gpu/eri.h"
#include "libintx/gpu/api/api.h"

#include <limits>

/// The parts of the full-ERI interface that are neither format-specific nor
/// device code, so that each eri.<x>format.cu holds exactly one kernel and one
/// entry point.
namespace libintx::gpu::eri {

  bool format_fits(const Basis<Gaussian> &basis, size_t reserve) {
    const size_t n2 = basis.nbf()*basis.nbf();
    // nbf^4 doubles overflows size_t somewhere past nbf ~ 46000 -- far beyond
    // anything this path is for, but the answer there is "no", not a wrapped
    // around "yes".
    constexpr size_t max = std::numeric_limits<size_t>::max();
    if (n2 && (max/sizeof(double))/n2 < n2) return false;
    const size_t bytes = n2*n2*sizeof(double);
    const auto [free, total] = device::memory_info();
    (void)total;
    if (bytes > free) return false;
    return (free - bytes) >= reserve;
  }

}
