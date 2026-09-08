#ifndef LIBINTX_FOCK_MD_DF_H
#define LIBINTX_FOCK_MD_DF_H

#include "libintx/fock/md/driver.h"
#include "libintx/blas.h"
#include "libintx/kengine.h"
#include "libintx/screening.h"
#include "libintx/shell.h"
#include "libintx/utility.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <vector>

/// The density-fitted K build, the additive counterpart of driver.h.
///
/// driver.h contracts four-centre integrals as they are computed; this factors
/// them through an auxiliary basis first, exactly the way the DF J engine
/// (libintx::gpu::make_jengine) does for J:
///
///   (mu lambda | nu sigma) ~= sum_PQ (P|mu lambda) [V^-1]_PQ (Q|nu sigma)
///
/// with V[P,Q] = (P|Q) the Coulomb metric over the auxiliary basis. Writing
/// A[P,mu,nu] = (P|mu nu) and B = V^-1 A (contracted on the auxiliary index),
///
///   K[mu,nu] = sum_P (B_P D A_P)[mu,nu]
///
/// which is a pair of dense GEMMs per auxiliary function. Substituting the two
/// tensors is what buys the win over driver.h: three-centre integrals instead
/// of four, and the contraction that follows is BLAS rather than an eight-fold
/// permutation digest.
///
/// It is engine-parameterised the same way driver.h is -- the host
/// (libintx::md::IntegralEngine<3>) and device
/// (libintx::gpu::md::IntegralEngine<3>) three-centre engines share one
/// compute() contract, so each back end supplies only its engine type and its
/// integral buffer -- and it reuses driver.h's shell-pair binning, Matrix,
/// Density and tile plumbing rather than restating them. What it does not
/// share is the eight-fold permutation digest: a DF build never forms a
/// quartet, so there is no orbit to enumerate.
///
/// **This is an approximation.** Unlike driver.h it does not reproduce the
/// exact ERI, only the fit, so a DF K matrix agrees with a direct one to the
/// quality of the auxiliary basis and no further.
///
/// **Memory.** Both A and B are held in full: 2*naux*nbf^2 doubles, the same
/// all-in-core shape a textbook DF-K has, and the thing to attack first if
/// this is ever pointed at something large.
///
/// **Cost.** 2*naux*nbf^3 for the contraction. A DF-K in an SCF is usually
/// written against the occupied coefficients instead, which drops the nbf^3
/// to nbf^2*nocc -- but KEngine hands the engine a density through tile
/// callbacks, not a C_occ, and a general D has no factorisation to exploit
/// (it is not even guaranteed positive semi-definite through this interface).
/// So this is the right form *for this interface*; an occupied-space variant
/// would need one that passes C_occ.
namespace libintx::fock::md::df {

  /// One angular-momentum class of auxiliary shells: what can be handed to
  /// IntegralEngine<3> as a single bra batch.
  ///
  /// The three-centre bra is packed by md::make_basis<1>, which asserts that
  /// every shell in the batch shares an L and a contraction degree K, so this
  /// is the one-index analogue of make_pair_classes(): key on (L, pure, K),
  /// and the block dimension follows from the first two.
  struct AuxClass {
    int nbf = 0;
    std::vector<int> shells;
  };

  /// Split an auxiliary basis into batchable angular-momentum classes.
  inline std::vector<AuxClass> make_aux_classes(const Basis<Gaussian> &df_basis) {
    std::map< std::array<int,3>, AuxClass > classes;
    for (size_t i = 0; i < df_basis.size(); ++i) {
      const auto &x = df_basis[i];
      auto &c = classes[std::array<int,3>{ x.L, x.pure, x.K }];
      c.nbf = libintx::nbf(x);
      c.shells.push_back((int)i);
    }
    std::vector<AuxClass> v;
    v.reserve(classes.size());
    for (auto &kv : classes) v.push_back(std::move(kv.second));
    return v;
  }

  /// C(m,n) = alpha*A(m,k)*B(k,n) + beta*C(m,n), all three **row-major** and
  /// tightly packed.
  ///
  /// libintx::blas::gemm is column-major (it is cblas_dgemm), and a row-major
  /// matrix read column-major is its own transpose, so C^T = B^T A^T is the
  /// same call with the operands swapped and m/n exchanged. Doing that here
  /// once keeps the row-major layout the rest of the K engine uses --
  /// driver.h's Matrix, the tile callbacks -- from leaking transposes into
  /// everything that touches it.
  inline void gemm(
    size_t m, size_t n, size_t k,
    double alpha, const double *A, const double *B,
    double beta, double *C)
  {
    if (!m || !n || !k) return;
    blas::gemm(
      blas::NoTranspose, blas::NoTranspose,
      n, m, k,
      alpha,
      B, n,
      A, k,
      beta,
      C, n
    );
  }

  /// Evaluate A[P,mu,nu] = (P|mu nu) over the whole basis.
  ///
  /// @p A is naux blocks of a row-major nbf x nbf matrix, so element (P,mu,nu)
  /// lives at A[P*nbf*nbf + mu*nbf + nu]. It is written, not accumulated: the
  /// caller supplies a zeroed buffer and every pair the screening does not
  /// drop is filled in both orders.
  ///
  /// Screening is by AO pair only. libintx::PairScreening carries no auxiliary
  /// bound -- that is JEngine::Screening's max1(), which stays there because
  /// it means nothing to a conventional build -- so a pair goes on max2(i,j)
  /// against the largest possible partner, which is the bound driver.h
  /// applies and never more aggressive. Wiring max1() through to here is the
  /// obvious way to tighten this, and needs a screening type that has it.
  template<typename Engine, typename Buffer>
  void compute_eri3(
    const Basis<Gaussian> &basis,
    const Basis<Gaussian> &df_basis,
    const std::vector<AuxClass> &aux_classes,
    const std::vector<PairClass> &pair_classes,
    Engine &engine,
    Buffer &buffer,
    const libintx::PairScreening *screening,
    float dmax_global,
    size_t max_batch,
    double *A)
  {
    const size_t nbf = basis.nbf();
    const size_t nbf2 = nbf*nbf;

    std::vector<Index1> bra_index;
    std::vector<Index2> ket_index;

    for (const auto &X : aux_classes) {
      for (const auto &Q : pair_classes) {
        if (screening &&
            screening->skip(Q.max_norm*screening->max()*dmax_global)) {
          continue;
        }

        const size_t NX = (size_t)X.nbf;
        const size_t NC = (size_t)Q.nbf_first;
        const size_t ND = (size_t)Q.nbf_second;
        const size_t block = NX*NC*ND;

        // Bound a batch by the integral buffer, as driver.h does, and fill the
        // bra first: the three-centre kernels batch over the auxiliary index,
        // so a wide bra is the cheap direction.
        const size_t budget = std::max<size_t>(1, max_batch/block);
        const size_t nbra = std::max<size_t>(
          1, std::min(budget, X.shells.size())
        );
        const size_t nket = std::max<size_t>(1, budget/nbra);

        for (size_t i0 = 0; i0 < X.shells.size(); i0 += nbra) {
          size_t i1 = std::min(i0 + nbra, X.shells.size());
          bra_index.assign(X.shells.begin()+i0, X.shells.begin()+i1);

          for (size_t j0 = 0; j0 < Q.pairs.size(); j0 += nket) {
            size_t j1 = std::min(j0 + nket, Q.pairs.size());

            ket_index.clear();
            for (size_t j = j0; j < j1; ++j) {
              const auto &p = Q.pairs[j];
              if (screening &&
                  screening->skip(p.norm*screening->max()*dmax_global)) {
                continue;
              }
              ket_index.push_back(Index2{p.first, p.second});
            }
            if (ket_index.empty()) continue;

            const size_t M = bra_index.size(), N = ket_index.size();
            double *V = buffer.resize(M*N*block);
            std::array<size_t,2> dims = { M*NX, NC*ND*N };
            // No norms: the three-centre engines do not screen on them, and
            // the auxiliary bra has no bound to give them anyway.
            engine.compute(
              libintx::Coulomb, bra_index, ket_index,
              BraKet<const double*>{}, V, dims
            );
            buffer.synchronize();

            // V is column-major (M*NX) x (NC*ND*N): element (i,x,c,d,kl) sits
            // at i + M*(x + NX*(c + NC*(d + ND*kl))).
            for (size_t kl = 0; kl < N; ++kl) {
              auto [c0,d0] = ket_index[kl];
              const size_t mu0 = (size_t)basis.range(c0).begin();
              const size_t nu0 = (size_t)basis.range(d0).begin();
              for (size_t i = 0; i < M; ++i) {
                const size_t P0 = (size_t)df_basis.range(bra_index[i]).begin();
                for (size_t x = 0; x < NX; ++x) {
                  double *Ax = A + (P0 + x)*nbf2;
                  for (size_t c = 0; c < NC; ++c) {
                    for (size_t d = 0; d < ND; ++d) {
                      double v = V[i + M*(x + NX*(c + NC*(d + ND*kl)))];
                      // Only canonical pairs are computed, and (P|mu nu) is
                      // symmetric in mu,nu, so fill both halves.
                      Ax[(mu0+c)*nbf + (nu0+d)] = v;
                      Ax[(nu0+d)*nbf + (mu0+c)] = v;
                    }
                  }
                }
              }
            }

          }
        }
      }
    }
  }

  /// The density-fitted K build, parameterised on the three-centre engine.
  ///
  /// `Engine` needs
  ///   compute(Operator, const std::vector<Index1>&, const std::vector<Index2>&,
  ///           BraKet<const double*>, double*, const std::array<size_t,2>&)
  ///
  /// `Buffer` is driver.h's: resize(n) hands back a pointer the engine may
  /// write n doubles to and synchronize() is called before they are read.
  ///
  /// @p v_linv applies V^-1 over the auxiliary index -- see
  /// KEngine::MetricTransform for the layout it is handed.
  template<typename Engine, typename Buffer>
  void build(
    const Basis<Gaussian> &basis,
    const Basis<Gaussian> &df_basis,
    const std::vector<AuxClass> &aux_classes,
    const std::vector<PairClass> &pair_classes,
    Engine &engine,
    Buffer &buffer,
    const Density &d,
    const KEngine::MetricTransform &v_linv,
    const libintx::PairScreening *screening,
    size_t max_batch,
    Matrix &k)
  {
    libintx_assert(v_linv);

    const size_t nbf = basis.nbf();
    const size_t naux = df_basis.nbf();
    const size_t nbf2 = nbf*nbf;
    if (!nbf || !naux) return;

    std::vector<double> A(naux*nbf2, 0.0);
    compute_eri3(
      basis, df_basis, aux_classes, pair_classes, engine, buffer,
      screening, d.max, max_batch, A.data()
    );

    // B = V^-1 A over the auxiliary index. A is naux row-major nbf x nbf
    // blocks, which is the same bytes as a row-major naux x nbf^2 matrix --
    // exactly what MetricTransform is defined to take.
    std::vector<double> B(A);
    v_linv(B.data(), nbf2);

    // K[mu,nu] = sum_P (B_P D A_P)[mu,nu]. Symmetric overall even though no
    // single P term is: sum_P B_P (x) A_P (y) is sum_PQ A_P(x) [V^-1]_PQ A_Q(y),
    // which V^-1's own symmetry makes invariant under x <-> y.
    std::vector<double> T(nbf2);
    for (size_t p = 0; p < naux; ++p) {
      const double *Bp = B.data() + p*nbf2;
      const double *Ap = A.data() + p*nbf2;
      gemm(nbf, nbf, nbf, 1.0, Bp, d.matrix.data.data(), 0.0, T.data());
      gemm(nbf, nbf, nbf, 1.0, T.data(), Ap, 1.0, k.data.data());
    }
  }

}

#endif /* LIBINTX_FOCK_MD_DF_H */
