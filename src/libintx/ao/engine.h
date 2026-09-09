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
    /// **Only the bra derivative is computed, and that is the whole answer.**
    /// A 2-centre integral depends on the two centres only through
    /// `r_a - r_b`, so `d/dA + d/dB = 0` and the ket derivative is the
    /// negative of this one, elementwise. A caller assembling a gradient
    /// scatters `+V` onto the bra shell's atom and `-V` onto the ket shell's
    /// -- and must *accumulate*, because a pair with both shells on one atom
    /// hits the same slot twice (and then cancels, as it must).
    ///
    /// Throws for an operator this engine has no derivative kernel for.
    virtual void compute1(Operator, const std::vector<Index2>&, double*) = 0;

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

    /// First geometric derivative of the batch with respect to one centre.
    ///
    /// Shape and naming follow `IntegralEngine<2>::compute1`: a separate
    /// virtual rather than a fifth `Operator` or a defaulted argument on
    /// `compute`, and the `1` is the derivative *order*. What an N-centre
    /// integral needs on top of the 2-centre one is a *which centre* selector,
    /// because there is no `d/dA = -d/dB` identity to collapse them:
    /// translational invariance relates only the sum over all N.
    ///
    /// `V` is the `compute` layout with the Cartesian component as one more,
    /// slowest index -- three back-to-back blocks of exactly the shape and
    /// stride `compute` writes, so component `x` starts at
    /// `V + x*dims[0]*dims[1]` and `dims` is the same `{rows, cols}` pair
    /// `compute` takes. `V` must therefore have room for `3*dims[0]*dims[1]`
    /// doubles.
    ///
    /// Throws for a centre or an operator this engine has no derivative kernel
    /// for; the whole host side throws today.
    /// @param centre 0 is the auxiliary (bra) shell, 1 and 2 the two ket
    ///        shells. `(P|cd)` depends on the three centres only through their
    ///        differences, so `d/dP = -(d/dC + d/dD)` and centre 0 is
    ///        redundant -- the device engine throws for it rather than compute
    ///        it a second, more expensive way.
    virtual void compute1(
      Operator,
      int centre,
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

    /// First geometric derivative of the batch with respect to one centre.
    ///
    /// Shape and naming follow `IntegralEngine<2>::compute1`: a separate
    /// virtual rather than a fifth `Operator` or a defaulted argument on
    /// `compute`, and the `1` is the derivative *order*. What an N-centre
    /// integral needs on top of the 2-centre one is a *which centre* selector,
    /// because there is no `d/dA = -d/dB` identity to collapse them:
    /// translational invariance relates only the sum over all N.
    ///
    /// `V` is the `compute` layout with the Cartesian component as one more,
    /// slowest index -- three back-to-back blocks of exactly the shape and
    /// stride `compute` writes, so component `x` starts at
    /// `V + x*dims[0]*dims[1]` and `dims` is the same `{rows, cols}` pair
    /// `compute` takes. `V` must therefore have room for `3*dims[0]*dims[1]`
    /// doubles.
    ///
    /// Throws for a centre or an operator this engine has no derivative kernel
    /// for; the whole host side throws today.
    /// @param centre 0..3, selecting a, b, c or d of `(ab|cd)`. All four are
    ///        computed rather than the fourth taken from
    ///        `sum_centres d/dX = 0`, so that identity stays available as a
    ///        test rather than being built in.
    virtual void compute1(
      Operator op,
      int centre,
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
