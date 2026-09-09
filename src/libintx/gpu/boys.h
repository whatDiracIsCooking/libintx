#include "libintx/config.h"
#include "libintx/boys/gpu/chebyshev.h"

namespace libintx::gpu {

  /// One Chebyshev table for the whole device tree.
  ///
  /// The `+2` (rather than `+1`) is the derivative batches' one extra order:
  /// `gpu::md::IntegralEngine<N>::compute1` raises one side's Hermite L-sum by
  /// one, so the four-centre Boys order reaches `4*LMAX+1` and the
  /// three-centre one `2*LMAX+XMAX+1`. A value-only build never asks for it,
  /// and one extra order costs `(Order+1)*Segments` doubles.
  using Boys = boys::gpu::Chebyshev<7,std::max(LMAX*4,LMAX*2+XMAX)+2,117,117*7>;

  const Boys& boys();

}
