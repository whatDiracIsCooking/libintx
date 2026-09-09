#include "libintx/gpu/onebody/engine.h"
#include "libintx/gpu/onebody/basis.h"
#include "libintx/gpu/overlap/overlap.h"
#include "libintx/gpu/kinetic/kinetic.h"
#include "libintx/gpu/potential_en/potential_en.h"
#include "libintx/gpu/coulomb2/coulomb2.h"
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
    // Every `Operator` this engine can be handed now has a kernel, and each is
    // dispatched below to its own translation unit -- which owns the (A|B)
    // instantiations its kernel is compiled into -- so NOTHING reaches this
    // table. It stays because that is what a fifth operator lands in before it
    // has a kernel: an unimplemented operator has to say so rather than return
    // an untouched buffer that reads as zeros.
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

    auto stream = this->stream_;
    auto ab = make_basis(bra_, ket_, ijs, this->memory_->ab, stream);

    // Coulomb first, and before the LMAX bound below: (P|Q) is the
    // density-fitting metric, so its shells come from the AUXILIARY basis and
    // reach XMAX rather than LMAX. gpu/coulomb2/ sizes its own (A|B) table for
    // max(LMAX,XMAX) and asserts against that.
    if (op == Operator::Coulomb) {
      onebody::coulomb2(ab, V, ijs.size(), stream);
      return;
    }

    libintx_assert(ab.first.L <= LMAX);
    libintx_assert(ab.second.L <= LMAX);

    // Each operator that has a kernel is one line here, dispatching to the
    // translation unit nvcc compiled it into; that TU owns its own (A|B)
    // table, so the kernel table stays next to the kernels.
    if (op == Operator::Overlap) {
      onebody::overlap(ab, V, ijs.size(), stream);
      return;
    }

    if (op == Operator::Kinetic) {
      onebody::kinetic(ab, V, ijs.size(), stream);
      return;
    }

    if (op == Operator::Nuclear) {
      // Empty until set() has been called; potential_en says so rather than
      // reading an empty device::vector.
      const auto &centers = this->memory_->centers;
      onebody::potential_en(
        ab, centers.data(), (int)centers.size(), V, ijs.size(), stream
      );
      return;
    }

    auto kernel = kernels[ab.first.L][ab.second.L];
    kernel(*this, op, ab, V, ijs.size(), stream);

  }

  void IntegralEngine<2>::compute1(
    Operator op,
    const std::vector<Index2> &ijs,
    double *V)
  {

    libintx_assert(!ijs.empty());

    auto stream = this->stream_;
    auto ab = make_basis(bra_, ket_, ijs, this->memory_->ab, stream);
    libintx_assert(ab.first.L <= LMAX);
    libintx_assert(ab.second.L <= LMAX);

    // Overlap and Kinetic are one line each, exactly as in `compute` above:
    // their whole derivative fits this buffer. Anything without a derivative
    // kernel gets the treatment `compute` gives Coulomb -- say so, rather than
    // hand back a buffer that reads as a zero gradient, which is a
    // plausible-looking answer and the reason a missing derivative term is
    // hard to spot downstream.
    if (op == Operator::Overlap) {
      onebody::overlap1(ab, V, ijs.size(), stream);
      return;
    }

    if (op == Operator::Kinetic) {
      onebody::kinetic1(ab, V, ijs.size(), stream);
      return;
    }

    if (op == Operator::Nuclear) {
      // Nuclear HAS a derivative kernel; what it does not have is a single
      // output buffer. Sending the caller to the overload that takes both
      // beats writing the shell half here and silently dropping the
      // Hellmann-Feynman term, which is the one failure mode of this operator
      // a finite-difference test over shell centres alone would not catch.
      throw std::runtime_error(
        "libintx::gpu::md::IntegralEngine<2>::compute1: Operator::Nuclear has"
        " a second, nucleus-indexed derivative term (Hellmann-Feynman) that"
        " does not fit this buffer; use the compute1(op,ijs,dV,dVC) overload"
      );
    }

    throw std::runtime_error(
      str(
        "libintx::gpu::md::IntegralEngine<2>::compute1: no derivative kernel"
        " for operator ", name(op)
      )
    );

  }

  void IntegralEngine<2>::compute1(
    Operator op,
    const std::vector<Index2> &ijs,
    double *dV,
    double *dVC)
  {

    libintx_assert(!ijs.empty());

    if (op != Operator::Nuclear) {
      // Only the electron-nuclear potential has an operator that moves. For
      // the other two `dVC` would be a buffer of structural zeros, and a
      // caller writing one is confused about which term it is asking for.
      throw std::runtime_error(
        str(
          "libintx::gpu::md::IntegralEngine<2>::compute1: operator ", name(op),
          " has no nucleus-indexed derivative term; use the 3-argument overload"
        )
      );
    }

    auto stream = this->stream_;
    auto ab = make_basis(bra_, ket_, ijs, this->memory_->ab, stream);
    libintx_assert(ab.first.L <= LMAX);
    libintx_assert(ab.second.L <= LMAX);

    // Empty until set() has been called; potential_en1 says so rather than
    // reading an empty device::vector.
    const auto &centers = this->memory_->centers;
    onebody::potential_en1(
      ab, centers.data(), (int)centers.size(), dV, dVC, ijs.size(), stream
    );

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
