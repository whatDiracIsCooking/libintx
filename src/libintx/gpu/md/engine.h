#ifndef LIBINTX_GPU_MD_ENGINE_H
#define LIBINTX_GPU_MD_ENGINE_H

#include "libintx/gpu/forward.h"
#include "libintx/gpu/engine.h"
#include "libintx/shell.h"
#include "libintx/tensor.h"
#include <memory>

namespace libintx::gpu::md {

  struct Basis1;
  struct Basis2;

  template<int N>
  struct IntegralEngine;

  template<>
  struct IntegralEngine<3> : gpu::IntegralEngine<3> {

    IntegralEngine(
      const Basis<Gaussian> &bra,
      const Basis<Gaussian> &ket,
      gpuStream_t stream
    );

    ~IntegralEngine();

    void compute(
      Operator op,
      const std::vector<Index1> &bra,
      const std::vector<Index2> &ket,
      BraKet<const double*> norms,
      double*,
      const std::array<size_t,2>&
    ) override;

    /// `d(P|cd)/dX` for `centre` 1 (c) or 2 (d); centre 0 (the auxiliary
    /// shell) throws, because `d/dP = -(d/dC + d/dD)`. See
    /// `ao::IntegralEngine<3>::compute1` for the layout.
    void compute1(
      Operator op,
      int centre,
      const std::vector<Index1> &bra,
      const std::vector<Index2> &ket,
      BraKet<const double*> norms,
      double*,
      const std::array<size_t,2>&
    ) override;

  private:

    template<int>
    double* allocate(size_t);

    template<int Bra, int Ket>
    void compute(const Basis1&, const Basis2&, TensorRef<double,2>, gpuStream_t);

    /// @tparam Ket the ket batch's *Hermite* L-sum, `C + D + 1` for a
    ///         derivative batch -- which is the whole point of a separate
    ///         entry point: `compute`'s table indexes the kernel by the ket
    ///         pair's own L-sum, and a derivative batch decouples the two.
    template<int X, int Ket>
    void compute1(const Basis1&, const Basis2&, TensorRef<double,2>, gpuStream_t);

    template<int,int,int>
    auto compute_v0(
      const Basis1& x,
      const Basis2& ket,
      TensorRef<double,2> XCD,
      gpuStream_t stream
    );

    /// @tparam DC extra ket Hermite degree: 0 for a value batch, 1 for a
    ///         derivative one. The algorithm is the same either way -- one
    ///         kernel over the Hermite indices and one GEMM against the
    ///         batch's coefficient block, with nothing that assumes the block
    ///         stops at the pair's own L-sum.
    template<int,int,int,int DC = 0>
    auto compute_v2(
      const Basis1& x,
      const Basis2& ket,
      TensorRef<double,2> XCD,
      gpuStream_t stream
    );

  private:

    Basis<Gaussian> bra_, ket_;
    gpuStream_t stream_;
    struct Memory;
    std::unique_ptr<Memory> memory_;

  };


  template<>
  struct IntegralEngine<4> : gpu::IntegralEngine<4> {

    IntegralEngine(
      const Basis<Gaussian> &bra,
      const Basis<Gaussian> &ket,
      gpuStream_t stream
    );

    ~IntegralEngine();

    void compute(
      Operator,
      const std::vector<Index2> &bra,
      const std::vector<Index2> &ket,
      BraKet<const double*> norms,
      double*,
      const std::array<size_t,2>&
    ) override;

    /// `d(ab|cd)/dX` for `centre` 0..3. See `ao::IntegralEngine<4>::compute1`
    /// for the layout.
    void compute1(
      Operator,
      int centre,
      const std::vector<Index2> &bra,
      const std::vector<Index2> &ket,
      BraKet<const double*> norms,
      double*,
      const std::array<size_t,2>&
    ) override;

  private:

    template<int Bra, int Ket>
    void compute(const Basis2&, const Basis2&, TensorRef<double,2>, gpuStream_t);

    /// @tparam Bra,Ket the two batches' *Hermite* L-sums; the differentiated
    ///         side is one more than its shell pair's. Exactly one of
    ///         `DA`, `DC` is 1 -- which side carries the derivative batch.
    template<int Bra, int Ket, int DA, int DC>
    void compute1(const Basis2&, const Basis2&, TensorRef<double,2>, gpuStream_t);

    template<int,int,int,int>
    auto compute_v0(
      const Basis2& bra,
      const Basis2& ket,
      TensorRef<double,2> ABCD,
      gpuStream_t stream
    );

    template<int,int,int,int>
    auto compute_v1(
      const Basis2& bra,
      const Basis2& ket,
      TensorRef<double,2> ABCD,
      gpuStream_t stream
    );

    /// @tparam DA,DC extra bra/ket Hermite degree, 1 on the differentiated
    ///         side of a derivative batch. Either raises only the Hermite
    ///         range the two GEMMs contract over; the output stays shaped for
    ///         the shell pairs.
    template<int,int,int,int,int DA = 0,int DC = 0>
    auto compute_v2(
      const Basis2& bra,
      const Basis2& ket,
      TensorRef<double,2> ABCD,
      gpuStream_t stream
    );

    template<int>
    double* allocate(size_t);

  private:
    Basis<Gaussian> bra_, ket_;
    gpuStream_t stream_;
    struct Memory;
    std::unique_ptr<Memory> memory_;

  };


} // libintx::gpu::md

#endif /* LIBINTX_GPU_MD_ENGINE_H */
