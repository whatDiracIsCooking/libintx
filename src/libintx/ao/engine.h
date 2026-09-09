#ifndef LIBINTX_AO_ENGINE_H
#define LIBINTX_AO_ENGINE_H

#include "libintx/forward.h"
#include "libintx/shell.h"

#include <vector>
#include <array>
#include <tuple>
#include <cstddef>
#include <functional>
#include <memory>

namespace libintx {

  struct Overlap::Operator::Parameters {
  };

  struct Kinetic::Operator::Parameters {
  };

  struct Coulomb::Operator::Parameters {
  };

  /// The point charges the electron-nuclear attraction operator sums over.
  ///
  /// **`centers[i]` is nucleus `i`, and a derivative's nucleus index is this
  /// one.** The tuple carries a charge and a position and no atom index, the
  /// same gap `Basis<Gaussian>` has for shells -- so rather than take a
  /// separate map, the *order* is the convention: the Hellmann-Feynman
  /// gradient `ao::IntegralEngine<2>::compute1` writes is indexed by position
  /// in this vector, and a caller that built it from its atom list in atom
  /// order can scatter straight into that atom's slot.
  ///
  /// Two consequences worth stating. A caller that reorders, filters or
  /// deduplicates its atoms on the way in must apply the same permutation on
  /// the way out; and an atom that carries both a nucleus and basis functions
  /// -- every atom in a normal molecule -- receives a contribution from this
  /// index *and* from the shell derivatives, into the same slot, so the
  /// scatter accumulates rather than assigns.
  struct Nuclear::Operator::Parameters {
    using Zr = std::tuple< int,std::array<double,3> >;
    std::vector<Zr> centers;
  };

}

namespace libintx::ao {

  template<>
  struct IntegralEngine<2> : IntegralEngine<> {

    virtual ~IntegralEngine() = default;
    virtual void set(const Nuclear::Operator::Parameters&) = 0;
    virtual void compute(Operator, const std::vector<Index2>&, double*) = 0;

    /// First geometric derivative of a 2-centre operator, with respect to the
    /// **bra** centre.
    ///
    /// `Operator` is `{Overlap,Kinetic,Nuclear,Coulomb}` and a derivative is
    /// not a fifth member of it -- it is a derivative *of* one -- so this is
    /// its own entry point rather than a defaulted `deriv` argument on
    /// `compute`. (A default argument on a virtual binds to the *static* type,
    /// which is exactly the wrong thing for an interface two engines override.)
    /// The `1` is the derivative *order*, not a centre count.
    ///
    /// `V` is the `compute` layout with one more, slowest-varying index:
    ///
    ///     V[ij + (na + nb*npure(A) + x*npure(A)*npure(B))*ldV]
    ///
    /// `ldV = ij.size()`, `x` in `{0,1,2}` selecting `d/dA_x`. The `x = 0`
    /// block therefore has exactly the shape and stride `compute` writes.
    ///
    /// **For `Overlap` and `Kinetic` the bra derivative is the whole answer.**
    /// `S` and `T` depend on the two centres only through `r_a - r_b`, so
    /// `d/dA + d/dB = 0` and the ket derivative is the negative of this one,
    /// elementwise. A caller assembling a gradient scatters `+V` onto the bra
    /// shell's atom and `-V` onto the ket shell's -- and must *accumulate*,
    /// because a pair with both shells on one atom hits the same slot twice
    /// (and then cancels, as it must).
    ///
    /// **`Nuclear` is the exception, and this overload throws for it.** `V`
    /// depends on the nuclear positions as well as on the two shell centres,
    /// so `d/dB` is NOT `-d/dA`: what vanishes is the three-way sum
    /// `dV/dA + dV/dB + sum_C dV/dR_C`. The operator-derivative term is
    /// indexed by nucleus rather than by shell pair and does not fit this
    /// buffer's shape, so it has its own -- see the 4-argument overload below.
    /// Throwing here is deliberate: a `V` gradient missing the
    /// Hellmann-Feynman term is the dominant part of the force on a charged
    /// atom, and it is smooth, plausible and wrong rather than obviously
    /// broken.
    ///
    /// Throws for an operator this engine has no derivative kernel for.
    virtual void compute1(Operator, const std::vector<Index2>&, double*) = 0;

    /// First geometric derivative of a 2-centre operator whose *operator* also
    /// moves -- i.e. `Operator::Nuclear`, and nothing else in this set.
    ///
    /// `dV` is exactly the 3-argument overload's buffer: the bra derivative,
    /// `dV/dA_x`, in
    ///
    ///     dV[ij + (na + nb*npure(A) + x*npure(A)*npure(B))*ldV]
    ///
    /// `dVC` is the Hellmann-Feynman term, `dV/dR_C,x` -- one such block per
    /// nucleus, the nucleus as one more, slowest index:
    ///
    ///     dVC[ij + (na + nb*npure(A) + x*npure(A)*npure(B)
    ///               + c*3*npure(A)*npure(B))*ldV]
    ///
    /// so `dVC` is `ncenters` times the size of `dV`, and `c` indexes
    /// `Nuclear::Operator::Parameters::centers` in the order `set()` was given
    /// them -- which that struct now documents as the atom order.
    ///
    /// The caller obtains `dV/dB` the way it does for `S`: from the transposed
    /// bin, since `V(a,b) = V(b,a)`. The three sets then satisfy
    /// `dV/dA + dV/dB + sum_C dV/dR_C = 0` elementwise, which is the check
    /// that pins the two contributions against each other.
    ///
    /// Throws for any operator but `Nuclear`, and for an engine with no
    /// derivative kernel.
    virtual void compute1(
      Operator, const std::vector<Index2>&, double *dV, double *dVC
    ) = 0;

    void overlap(const std::vector<Index2> &ij, double *V) {
      this->compute(Overlap,ij,V);
    };

    /// `d(mu|nu)/dA_x` for `Operator::Overlap`; see `compute1`.
    void overlap1(const std::vector<Index2> &ij, double *V) {
      this->compute1(Overlap,ij,V);
    };

    void kinetic(const std::vector<Index2> &ij, double *V) {
      this->compute(Kinetic,ij,V);
    }

    void nuclear(const std::vector<Index2> &ij, double *V) {
      this->compute(Nuclear,ij,V);
    }

  };

  template<>
  struct IntegralEngine<3> : IntegralEngine<> {

    virtual ~IntegralEngine() = default;

    virtual void compute(
      Operator,
      const std::vector<Index1> &bra,
      const std::vector<Index2> &ket,
      BraKet<const double*> norms,
      double*,
      const std::array<size_t,2> &dims
    ) = 0;

    void coulomb(
      const std::vector<Index1> &bra,
      const std::vector<Index2> &ket,
      BraKet<const double*> norms,
      double* V,
      const std::array<size_t,2> &dims)
    {
      this->compute(libintx::Coulomb,bra,ket,norms,V,dims);
    }

  };

  template<>
  struct IntegralEngine<4> : IntegralEngine<> {
    virtual ~IntegralEngine() = default;
    virtual void compute(
      Operator op,
      const std::vector<Index2> &bra,
      const std::vector<Index2> &ket,
      BraKet<const double*> norms,
      double *V,
      const std::array<size_t,2> &dims
    ) = 0;
    size_t max_memory = 0;
  };

  template<int ... Args>
  std::unique_ptr< IntegralEngine<Args...> > integral_engine(
    const Basis<Gaussian>&,
    const Basis<Gaussian>&
  ) = delete;

  template<>
  std::unique_ptr< IntegralEngine<2> > integral_engine(
    const Basis<Gaussian>&,
    const Basis<Gaussian>&
  );

  template<>
  std::unique_ptr< IntegralEngine<3> > integral_engine(
    const Basis<Gaussian>&,
    const Basis<Gaussian>&
  );

  template<>
  std::unique_ptr< IntegralEngine<4> > integral_engine(
    const Basis<Gaussian>&,
    const Basis<Gaussian>&
  );

}

#endif /* LIBINTX_AO_ENGINE_H */
