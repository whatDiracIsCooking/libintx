#include "libintx/gpu/md/engine.h"
#include "libintx/gpu/md/basis.h"
#include "libintx/config.h"
#include "libintx/utility.h"
#include <functional>
#include <stdexcept>

namespace libintx::gpu::md {

  struct IntegralEngine<4>::Memory {
    device::vector<double> p;
    device::vector<double> q;
    std::array<device::vector<double>,3> buffer;
  };

  IntegralEngine<4>::IntegralEngine(const Basis<Gaussian> &bra, const Basis<Gaussian> &ket, gpuStream_t stream) {
    libintx_assert(!bra.empty());
    libintx_assert(!ket.empty());
    bra_ = bra;
    ket_ = ket;
    stream_ = stream;
    memory_.reset(new Memory);
  }

  IntegralEngine<4>::~IntegralEngine() {}

  void IntegralEngine<4>::compute(
    Operator,
    const std::vector<Index2> &bra,
    const std::vector<Index2> &ket,
    BraKet<const double*> norms,
    double *V,
    const std::array<size_t,2> &dims)
  {

    // See the note in md3.cc: accepted for interface parity with the host
    // engine, not yet used by the device kernels.
    (void)norms;

    using Kernel = std::function<void(
      IntegralEngine&, const Basis2&, const Basis2&, TensorRef<double,2>, gpuStream_t
    )>;

    static auto ab_cd_kernels = make_array<Kernel,2*LMAX+1,2*LMAX+1>(
      [](auto ab, auto cd) {
        return &IntegralEngine::compute<ab,cd>;
      }
    );

    auto stream = this->stream_;
    auto p = make_basis(bra_, bra_, bra, this->memory_->p, stream);
    auto q = make_basis(ket_, ket_, ket, this->memory_->q, stream);
    auto kernel = ab_cd_kernels[p.first.L+p.second.L][q.first.L+q.second.L];
    kernel(*this, p, q, TensorRef{V,dims}, stream);

  }

  void IntegralEngine<4>::compute1(
    Operator,
    int centre,
    const std::vector<Index2> &bra,
    const std::vector<Index2> &ket,
    BraKet<const double*> norms,
    double *V,
    const std::array<size_t,2> &dims)
  {

    (void)norms;

    if (centre < 0 || centre > 3) {
      throw std::domain_error(
        str("libintx::gpu::md::IntegralEngine<4>::compute1: centre=", centre)
      );
    }

    using Kernel = std::function<void(
      IntegralEngine&, const Basis2&, const Basis2&, TensorRef<double,2>, gpuStream_t
    )>;

    // Two tables, and neither is `compute`'s. A derivative batch raises
    // exactly one side's *Hermite* L-sum by one while its output stays shaped
    // for the pair, so the kernel index and the output shape come apart --
    // which `ab_cd_kernels[bra.L][ket.L]` cannot express, and which is the
    // coupling this entry point exists to break. The first index of each table
    // is the bra batch's Hermite L-sum, the second the ket's.
    static auto bra_kernels = make_array<Kernel,2*LMAX+2,2*LMAX+1>(
      [](auto ab, auto cd) -> Kernel {
        return &IntegralEngine::compute1<ab,cd,1,0>;
      }
    );
    static auto ket_kernels = make_array<Kernel,2*LMAX+1,2*LMAX+2>(
      [](auto ab, auto cd) -> Kernel {
        return &IntegralEngine::compute1<ab,cd,0,1>;
      }
    );

    auto stream = this->stream_;
    const size_t component = dims[0]*dims[1];

    if (centre < 2) {
      // the ket batch is the same for all three components
      auto q = make_basis(ket_, ket_, ket, this->memory_->q, stream);
      for (int x = 0; x < 3; ++x) {
        auto p = make_basis1(bra_, bra_, bra, centre, x, this->memory_->p, stream);
        auto kernel = bra_kernels[p.first.L+p.second.L+1][q.first.L+q.second.L];
        kernel(*this, p, q, TensorRef{V + x*component, dims}, stream);
      }
    }
    else {
      auto p = make_basis(bra_, bra_, bra, this->memory_->p, stream);
      for (int x = 0; x < 3; ++x) {
        auto q = make_basis1(ket_, ket_, ket, centre-2, x, this->memory_->q, stream);
        auto kernel = ket_kernels[p.first.L+p.second.L][q.first.L+q.second.L+1];
        kernel(*this, p, q, TensorRef{V + x*component, dims}, stream);
      }
    }

  }

  template<int Idx>
  double* IntegralEngine<4>::allocate(size_t size) {
    auto &v = std::get<Idx>(this->memory_->buffer);
    v.resize(size);
    return v.data();
  }

  template double* IntegralEngine<4>::allocate<0>(size_t size);
  template double* IntegralEngine<4>::allocate<1>(size_t size);
  template double* IntegralEngine<4>::allocate<2>(size_t size);

}

template<>
std::unique_ptr< libintx::gpu::IntegralEngine<4> > libintx::gpu::integral_engine(
  const Basis<Gaussian> &bra,
  const Basis<Gaussian> &ket,
  const gpuStream_t &stream)
{
  return std::make_unique< gpu::md::IntegralEngine<4> >(bra, ket, stream);
}
