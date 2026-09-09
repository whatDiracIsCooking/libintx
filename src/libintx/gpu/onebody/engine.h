#ifndef LIBINTX_GPU_ONEBODY_ENGINE_H
#define LIBINTX_GPU_ONEBODY_ENGINE_H

#include "libintx/gpu/forward.h"
#include "libintx/gpu/engine.h"
#include "libintx/gpu/onebody/basis.h"
#include "libintx/shell.h"

#include <memory>
#include <vector>

// The device one-electron engine.
//
// Namespace: libintx::gpu::md, the same as the device Coulomb engines, even
// though the files sit outside gpu/md/. The algorithm is still
// McMurchie-Davidson -- a second namespace for the same method would buy
// nothing and cost a rename across four directories once the operator kernels
// land in gpu/overlap/, gpu/kinetic/ and gpu/potential_en/.

namespace libintx::gpu::md {

  template<int N>
  struct IntegralEngine;

  /// Device overlap / kinetic / electron-nuclear-potential / two-centre
  /// Coulomb engine.
  ///
  /// `Operator::Coulomb` is the two-centre metric `(P|Q)` over an auxiliary
  /// basis (`gpu/coulomb2/`), the one two-ELECTRON operator here and the one
  /// whose shells reach XMAX rather than LMAX. The host
  /// `md::IntegralEngine<2>` does not implement it.
  ///
  /// Same contract as the host `libintx::md::IntegralEngine<2>`, so a caller
  /// swaps host for device by changing which `integral_engine<2>` it calls --
  /// the property that lets a driver be written once against
  /// `ao::IntegralEngine<N>`.
  ///
  /// `compute` writes the host engine's layout byte for byte:
  ///
  ///     V[ij + (na + nb*npure(A))*ldV],   ldV = ijs.size()
  ///
  /// `V` is host memory the caller has registered with
  /// `gpu::host::register_pointer`, the same convention as the device md3/md4
  /// engines.
  template<>
  struct IntegralEngine<2> : gpu::IntegralEngine<2> {

    IntegralEngine(
      const Basis<Gaussian> &bra,
      const Basis<Gaussian> &ket,
      gpuStream_t stream
    );

    ~IntegralEngine();

    /// Upload the point charges the electron-nuclear potential contracts
    /// against. Called once per geometry; a second call replaces the first.
    void set(const Nuclear::Operator::Parameters&) override;

    void compute(Operator, const std::vector<Index2>&, double*) override;

    /// First derivative with respect to the bra centre; see
    /// `ao::IntegralEngine<2>::compute1` for the layout and for why the ket
    /// derivative is not computed.
    ///
    /// `Operator::Overlap` only. `Operator::Nuclear` has a derivative kernel
    /// but does not fit this signature -- its Hellmann-Feynman term is indexed
    /// by nucleus -- so it goes through the 4-argument overload and throws
    /// here. Kinetic has no derivative kernel yet and throws too, rather than
    /// leave a caller holding zeros.
    void compute1(Operator, const std::vector<Index2>&, double*) override;

    /// `dV/dA_x` into `dV` and the Hellmann-Feynman `dV/dR_C,x` into `dVC`;
    /// see `ao::IntegralEngine<2>::compute1` for both layouts.
    ///
    /// `Operator::Nuclear` only -- the other two operators do not move with
    /// the operator and have nothing to write into `dVC`. Requires `set()`.
    void compute1(
      Operator, const std::vector<Index2>&, double *dV, double *dVC
    ) override;

  private:

    template<int A, int B>
    void compute(Operator, const GaussianPairs&, double *V, size_t ldV, gpuStream_t);

  private:
    Basis<Gaussian> bra_, ket_;
    gpuStream_t stream_;
    struct Memory;
    std::unique_ptr<Memory> memory_;

  };

} // libintx::gpu::md

#endif /* LIBINTX_GPU_ONEBODY_ENGINE_H */
