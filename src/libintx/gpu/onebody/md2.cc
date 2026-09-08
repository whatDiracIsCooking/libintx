#include "libintx/gpu/onebody/engine.h"
#include "libintx/gpu/onebody/basis.h"
#include "libintx/config.h"
#include "libintx/utility.h"

#include <functional>
#include <stdexcept>

namespace libintx::gpu::md {

  namespace {

    const char* name(Operator op) {
      switch (op) {
      case Operator::Overlap: return "Overlap";
      case Operator::Kinetic: return "Kinetic";
      case Operator::Nuclear: return "Nuclear";
      case Operator::Coulomb: return "Coulomb";
      }
      return "?";
    }

  }

  struct IntegralEngine<2>::Memory {
    /// The current batch of shell pairs, kept across compute calls.
    Gaussian2Buffer ab;
    /// The point charges from the last set(). Empty until set() is called.
    device::vector<NuclearCenter> centers;
  };

  IntegralEngine<2>::IntegralEngine(
    const Basis<Gaussian> &bra,
    const Basis<Gaussian> &ket,
    gpuStream_t stream)
  {
    libintx_assert(!bra.empty());
    libintx_assert(!ket.empty());
    bra_ = bra;
    ket_ = ket;
    stream_ = stream;
    memory_.reset(new Memory);
  }

  IntegralEngine<2>::~IntegralEngine() {}

  void IntegralEngine<2>::set(const Nuclear::Operator::Parameters &params) {
    // Once per geometry, not once per compute call -- and re-uploaded on every
    // call, so a second set() is not quietly ignored.
    make_centers(params, this->memory_->centers, this->stream_);
  }

  template<int A, int B>
  void IntegralEngine<2>::compute(
    Operator op,
    const GaussianPairs &ab,
    double *V,
    size_t ldV,
    gpuStream_t stream)
  {
    (void)ab;
    (void)V;
    (void)ldV;
    (void)stream;
    // Scaffolding only: the three operator kernels land in gpu/overlap/,
    // gpu/kinetic/ and gpu/potential_en/ as follow-ups, each of which is a
    // kernel file, an entry here and a test case. Until then say so rather
    // than return an untouched buffer that reads as zeros.
    throw std::runtime_error(
      str(
        "libintx::gpu::md::IntegralEngine<2>::compute: operator ",
        name(op), " not implemented for (A|B)=(", A, "|", B, ")"
      )
    );
  }

  void IntegralEngine<2>::compute(
    Operator op,
    const std::vector<Index2> &ijs,
    double *V)
  {

    libintx_assert(!ijs.empty());

    using Kernel = std::function<void(
      IntegralEngine&, Operator, const GaussianPairs&, double*, size_t, gpuStream_t
    )>;

    // (LMAX+1)^2 entries -- 16 at LMAX=3. Small enough that the one-TU-per-
    // operator split gpu/onebody/CMakeLists.txt starts with is fine; the
    // Coulomb path needs one OBJECT library per (bra,ket) pair only because
    // md4 alone is (2*LMAX+1)^2 translation units.
    static auto kernels = make_array<Kernel,LMAX+1,LMAX+1>(
      [](auto a, auto b) {
        return &IntegralEngine::compute<a,b>;
      }
    );

    if (op == Operator::Nuclear) {
      libintx_assert(this->memory_->centers.size());
    }

    auto stream = this->stream_;
    auto ab = make_basis(bra_, ket_, ijs, this->memory_->ab, stream);
    libintx_assert(ab.first.L <= LMAX);
    libintx_assert(ab.second.L <= LMAX);
    auto kernel = kernels[ab.first.L][ab.second.L];
    kernel(*this, op, ab, V, ijs.size(), stream);

  }

} // libintx::gpu::md

template<>
std::unique_ptr< libintx::gpu::IntegralEngine<2> > libintx::gpu::integral_engine(
  const Basis<Gaussian> &bra,
  const Basis<Gaussian> &ket,
  const gpuStream_t &stream)
{
  return std::make_unique< gpu::md::IntegralEngine<2> >(bra, ket, stream);
}
