#include "libintx/gpu/md/engine.h"
#include "libintx/gpu/md/basis.h"
#include "libintx/config.h"
#include "libintx/utility.h"
#include <functional>
#include <stdexcept>

namespace libintx::gpu::md {

  struct IntegralEngine<3>::Memory {
    device::vector<Hermite> p;
    device::vector<double> q;
    std::array<device::vector<double>,1> buffer;
  };

  IntegralEngine<3>::IntegralEngine(const Basis<Gaussian> &bra, const Basis<Gaussian> &ket, gpuStream_t stream) {
    libintx_assert(!bra.empty());
    libintx_assert(!ket.empty());
    bra_ = bra;
    ket_ = ket;
    stream_ = stream;
    memory_.reset(new Memory);
  }

  IntegralEngine<3>::~IntegralEngine() {}

  void IntegralEngine<3>::compute(
    Operator op,
    const std::vector<Index1> &bra,
    const std::vector<Index2> &ket,
    BraKet<const double*> norms,
    double *V,
    const std::array<size_t,2> &dims)
  {

    // The device kernels do not screen on the pair norms yet; the argument is
    // here because ao::IntegralEngine<N> defines it, so the host and the
    // device engines can be driven through one interface (which is what the K
    // engine's shared driver does).
    (void)norms;

    using Kernel = std::function<void(
      IntegralEngine&, const Basis1&, const Basis2&, TensorRef<double,2>, gpuStream_t
    )>;

    static auto x_cd_kernels = make_array<Kernel,XMAX+1,2*LMAX+1>(
      [](auto x, auto cd) {
        return &IntegralEngine::compute<x,cd>;
      }
    );

    auto stream = this->stream_;
    auto p = make_basis(bra_, bra, this->memory_->p, stream);
    auto q = make_basis(ket_, ket_, ket, this->memory_->q, stream);
    auto kernel = x_cd_kernels[p.L][q.first.L+q.second.L];
    kernel(*this, p, q, TensorRef{V,dims}, stream);

  }

  void IntegralEngine<3>::compute1(
    Operator,
    int centre,
    const std::vector<Index1> &bra,
    const std::vector<Index2> &ket,
    BraKet<const double*> norms,
    double *V,
    const std::array<size_t,2> &dims)
  {

    (void)norms;

    if (centre == 0) {
      // The auxiliary shell is a single Hermite bra, not a pair: its
      // "transform" is the analytic hermite_to_cartesian + cartesian_to_pure
      // written into md3.kernel.h, not a per-batch coefficient block, so there
      // is no slot to bake dE/dP into. It is also unnecessary -- (P|cd)
      // depends on the three centres only through their differences, so
      // d/dP = -(d/dC + d/dD) and the caller has it for free.
      throw std::domain_error(
        "libintx::gpu::md::IntegralEngine<3>::compute1: no derivative with"
        " respect to the auxiliary centre; use d/dP = -(d/dC + d/dD)"
      );
    }
    if (centre != 1 && centre != 2) {
      throw std::domain_error(
        str("libintx::gpu::md::IntegralEngine<3>::compute1: centre=", centre)
      );
    }

    using Kernel = std::function<void(
      IntegralEngine&, const Basis1&, const Basis2&, TensorRef<double,2>, gpuStream_t
    )>;

    // Indexed by the ket batch's *Hermite* L-sum, which a derivative batch
    // carries one above its pair's -- see IntegralEngine<4>::compute1.
    static auto x_cd_kernels = make_array<Kernel,XMAX+1,2*LMAX+2>(
      [](auto x, auto cd) -> Kernel {
        return &IntegralEngine::compute1<x,cd>;
      }
    );

    auto stream = this->stream_;
    const size_t component = dims[0]*dims[1];

    auto p = make_basis(bra_, bra, this->memory_->p, stream);
    for (int x = 0; x < 3; ++x) {
      auto q = make_basis1(ket_, ket_, ket, centre-1, x, this->memory_->q, stream);
      auto kernel = x_cd_kernels[p.L][q.first.L+q.second.L+1];
      kernel(*this, p, q, TensorRef{V + x*component, dims}, stream);
    }

  }

  template<int Idx>
  double* IntegralEngine<3>::allocate(size_t size) {
    auto &v = std::get<Idx>(this->memory_->buffer);
    v.resize(size);
    return v.data();
  }

  template double* IntegralEngine<3>::allocate<0>(size_t size);

} // libintx::gpu::md

template<>
std::unique_ptr< libintx::gpu::IntegralEngine<3> > libintx::gpu::integral_engine(
  const Basis<Gaussian> &bra,
  const Basis<Gaussian> &ket,
  const gpuStream_t &stream)
{
  return std::make_unique< gpu::md::IntegralEngine<3> >(bra, ket, stream);
}
