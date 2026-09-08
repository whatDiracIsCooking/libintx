#ifndef LIBINTX_TESTS_KENGINE_TEST_H
#define LIBINTX_TESTS_KENGINE_TEST_H

#include "test.h"

#include "libintx/kengine.h"
#include "libintx/shell.h"
#include "libintx/ao/md/reference.h"
#include "libintx/pure.reference.h"

#include <Eigen/Dense>

#include <utility>
#include <vector>

/// Shared scaffolding for the four K engine tests -- host and device, direct
/// and density-fitted. Everything here is deliberately the dumbest possible
/// caller: dense matrices, every tile wanted, every tile available, and
/// references that share no code with the engines they check.
///
/// The DF helpers below assume solid-harmonic shells, which is what
/// test::gaussian builds; libintx::pure::reference::transform is what turns
/// the cartesian reference integrals into the engines' basis.
namespace libintx::test::kengine {

  /// A dense row-major nbf x nbf matrix.
  struct Matrix {
    size_t n;
    std::vector<double> data;
    explicit Matrix(size_t n) : n(n), data(n*n, 0.0) {}
    double& operator()(size_t i, size_t j) { return data[i*n + j]; }
    double operator()(size_t i, size_t j) const { return data[i*n + j]; }
  };

  inline KEngine::TileIn tile_in(const Matrix &m) {
    return [&m](KEngine::TileIndex i, KEngine::TileIndex j, double *out) {
      size_t nj = j.second - j.first;
      for (size_t p = i.first; p < i.second; ++p) {
        for (size_t q = j.first; q < j.second; ++q) {
          out[(p - i.first)*nj + (q - j.first)] = m(p,q);
        }
      }
      return true;
    };
  }

  inline KEngine::TileOut tile_out(Matrix &m) {
    return [&m](KEngine::TileIndex i, KEngine::TileIndex j, const double *in) {
      if (!in) return true; // query form: every tile is wanted
      size_t nj = j.second - j.first;
      for (size_t p = i.first; p < i.second; ++p) {
        for (size_t q = j.first; q < j.second; ++q) {
          m(p,q) = in[(p - i.first)*nj + (q - j.first)];
        }
      }
      return true;
    };
  }

  /// A symmetric density with no structure an engine could exploit by
  /// accident.
  inline Matrix random_density(size_t n) {
    Matrix d(n);
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j <= i; ++j) {
        double v = test::random<double>(-0.5, 0.5);
        d(i,j) = v;
        d(j,i) = v;
      }
    }
    return d;
  }

  /// A small basis with a mix of angular momenta and contraction depths --
  /// the cases a K digest has to keep straight are equal shells (a==b), equal
  /// pairs ((ab)==(cd)) and mixed classes, so the basis has to contain repeats
  /// of the same L on different centres. Shells above the configured LMAX are
  /// dropped, so a case that only means something at higher L has to say so
  /// itself.
  inline Basis<Gaussian> make_test_basis(const std::vector<int> &Ls, int K) {
    Basis<Gaussian> basis;
    for (int L : Ls) {
      if (L > LMAX) continue;
      basis.push_back(test::gaussian(L, K, /*pure=*/true));
    }
    return basis;
  }

  /// A[P,mu,nu] = (P|mu nu) from libintx::md::reference, in the layout
  /// libintx::kengine::md::df::compute_eri3 produces: naux blocks of a
  /// row-major nbf x nbf matrix, element (P,mu,nu) at A[P*nbf*nbf + mu*nbf + nu].
  inline std::vector<double> reference_eri3(
    const Basis<Gaussian> &basis,
    const Basis<Gaussian> &df_basis)
  {
    const size_t nbf = basis.nbf();
    const size_t naux = df_basis.nbf();
    std::vector<double> A(naux*nbf*nbf, 0.0);
    for (size_t x = 0; x < df_basis.size(); ++x) {
      const auto &X = df_basis[x];
      for (size_t i = 0; i < basis.size(); ++i) {
        for (size_t j = 0; j < basis.size(); ++j) {
          const auto &C = basis[i];
          const auto &D = basis[j];
          auto v = zeros(npure(X.L), 1, npure(C.L), npure(D.L));
          {
            auto cartesian = zeros(ncart(X.L), 1, ncart(C.L), ncart(D.L));
            libintx::md::reference::compute(
              X, Unit<Gaussian>{}, C, D, cartesian
            );
            libintx::pure::reference::transform(
              X.L, 0, C.L, D.L, cartesian, v
            );
          }
          const int P0 = df_basis.range(x).begin();
          const int mu0 = basis.range(i).begin();
          const int nu0 = basis.range(j).begin();
          for (int p = 0; p < npure(X.L); ++p) {
            for (int c = 0; c < npure(C.L); ++c) {
              for (int d = 0; d < npure(D.L); ++d) {
                A[(size_t)(P0+p)*nbf*nbf + (size_t)(mu0+c)*nbf + (nu0+d)] =
                  v(p,0,c,d);
              }
            }
          }
        }
      }
    }
    return A;
  }

  /// The Coulomb metric V[P,Q] = (P|Q) over an auxiliary basis, from
  /// libintx::md::reference with a unit shell in each ket slot.
  inline Eigen::MatrixXd reference_metric(const Basis<Gaussian> &df_basis) {
    const size_t naux = df_basis.nbf();
    Eigen::MatrixXd V(naux, naux);
    V.setZero();
    for (size_t i = 0; i < df_basis.size(); ++i) {
      for (size_t j = 0; j < df_basis.size(); ++j) {
        const auto &P = df_basis[i];
        const auto &Q = df_basis[j];
        auto v = zeros(npure(P.L), 1, npure(Q.L), 1);
        {
          auto cartesian = zeros(ncart(P.L), 1, ncart(Q.L), 1);
          libintx::md::reference::compute(
            P, Unit<Gaussian>{}, Q, Unit<Gaussian>{}, cartesian
          );
          libintx::pure::reference::transform(
            P.L, 0, Q.L, 0, cartesian, v
          );
        }
        for (int p = 0; p < npure(P.L); ++p) {
          for (int q = 0; q < npure(Q.L); ++q) {
            V(df_basis.range(i).begin()+p, df_basis.range(j).begin()+q) =
              v(p,0,q,0);
          }
        }
      }
    }
    return V;
  }

  /// A KEngine::MetricTransform that left-multiplies by a fixed matrix.
  ///
  /// The engine hands it a row-major naux x n block; W is naux x naux and the
  /// result replaces the block. Passing something other than V^-1 is how the
  /// assembly tests below pin the layout down: a transposed or wrongly-indexed
  /// application survives a symmetric V^-1 and does not survive this.
  inline KEngine::MetricTransform metric_transform(Eigen::MatrixXd W) {
    return [W = std::move(W)](double *X, size_t n) {
      using RowMajor = Eigen::Matrix<
        double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor
      >;
      Eigen::Map<RowMajor> M(X, W.rows(), (Eigen::Index)n);
      M = (W*M).eval();
    };
  }

  /// The density-fitted exchange, assembled the long way:
  ///
  ///   B = W A,  K[mu,nu] = sum_P sum_{lambda,sigma} B[P,mu,lambda]
  ///                        D[lambda,sigma] A[P,nu,sigma]
  ///
  /// with A the reference three-centre integrals above. Shares no code with
  /// the engine -- no shell-pair binning, no batching, no BLAS -- so agreement
  /// is evidence rather than tautology.
  inline Matrix reference_df_exchange(
    const Basis<Gaussian> &basis,
    const Basis<Gaussian> &df_basis,
    const Matrix &d,
    const Eigen::MatrixXd &W)
  {
    const size_t nbf = basis.nbf();
    const size_t naux = df_basis.nbf();
    auto A = reference_eri3(basis, df_basis);

    std::vector<double> B(naux*nbf*nbf, 0.0);
    for (size_t p = 0; p < naux; ++p) {
      for (size_t q = 0; q < naux; ++q) {
        double w = W(p,q);
        if (!w) continue;
        for (size_t m = 0; m < nbf*nbf; ++m) {
          B[p*nbf*nbf + m] += w*A[q*nbf*nbf + m];
        }
      }
    }

    Matrix k(nbf);
    for (size_t p = 0; p < naux; ++p) {
      const double *Ap = A.data() + p*nbf*nbf;
      const double *Bp = B.data() + p*nbf*nbf;
      for (size_t mu = 0; mu < nbf; ++mu) {
        for (size_t nu = 0; nu < nbf; ++nu) {
          double v = 0;
          for (size_t lambda = 0; lambda < nbf; ++lambda) {
            for (size_t sigma = 0; sigma < nbf; ++sigma) {
              v += Bp[mu*nbf + lambda]*d(lambda,sigma)*Ap[nu*nbf + sigma];
            }
          }
          k(mu,nu) += v;
        }
      }
    }
    return k;
  }

}

#endif /* LIBINTX_TESTS_KENGINE_TEST_H */
