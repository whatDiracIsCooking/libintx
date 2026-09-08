#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "test.h"

#include "libintx/gpu/eri.h"
#include "libintx/gpu/api/api.h"
#include "libintx/ao/md/reference.h"
#include "libintx/pure.reference.h"

#include <vector>

using namespace libintx;
using libintx::test::zeros;

namespace {

  /// A dense row-major nbf x nbf matrix. `data` in this order is exactly the
  /// vec(D) the format's GEMV expects.
  struct Matrix {
    size_t n;
    std::vector<double> data;
    explicit Matrix(size_t n) : n(n), data(n*n, 0.0) {}
    double& operator()(size_t i, size_t j) { return data[i*n + j]; }
    double operator()(size_t i, size_t j) const { return data[i*n + j]; }
  };

  Matrix random_density(size_t n) {
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

  Basis<Gaussian> make_test_basis(const std::vector<int> &Ls, int K) {
    Basis<Gaussian> basis;
    for (int L : Ls) {
      if (L > LMAX) continue;
      basis.push_back(test::gaussian(L, K, /*pure=*/true));
    }
    return basis;
  }

  /// Shells of differing angular momentum *and* differing contraction depth:
  /// the batch key is (L, solid-harmonic, K_a*K_b), so a basis of uniform K
  /// never exercises more than one contraction class.
  Basis<Gaussian> make_mixed_basis() {
    Basis<Gaussian> basis;
    basis.push_back(test::gaussian(0, 1, true));
    basis.push_back(test::gaussian(1, 3, true));
    if (LMAX >= 2) basis.push_back(test::gaussian(2, 2, true));
    basis.push_back(test::gaussian(0, 2, true));
    return basis;
  }

  /// The full ERI in J layout, G[(mu,nu),(lambda,sigma)] = (mu nu|lambda
  /// sigma), summed over every shell quartet with no permutational symmetry at
  /// all and every integral taken from libintx::md::reference. It shares no
  /// code with the kernel -- not the orbit, not the batching, not the layout --
  /// so agreement is evidence rather than tautology.
  std::vector<double> reference_jformat(const Basis<Gaussian> &basis) {
    const size_t nbf = basis.nbf();
    const size_t n2 = nbf*nbf;
    std::vector<double> g(n2*n2, 0.0);
    const size_t ns = basis.size();
    for (size_t a = 0; a < ns; ++a) {
      for (size_t b = 0; b < ns; ++b) {
        for (size_t c = 0; c < ns; ++c) {
          for (size_t e = 0; e < ns; ++e) {
            const auto &A = basis[a];
            const auto &B = basis[b];
            const auto &C = basis[c];
            const auto &E = basis[e];
            auto abce = zeros(npure(A.L), npure(B.L), npure(C.L), npure(E.L));
            {
              auto cartesian = zeros(
                ncart(A.L), ncart(B.L), ncart(C.L), ncart(E.L)
              );
              libintx::md::reference::compute(A, B, C, E, cartesian);
              libintx::pure::reference::transform(
                A.L, B.L, C.L, E.L, cartesian, abce
              );
            }
            const int ma = basis.range(a).begin();
            const int mb = basis.range(b).begin();
            const int mc = basis.range(c).begin();
            const int me = basis.range(e).begin();
            for (int i = 0; i < npure(A.L); ++i) {
              for (int j = 0; j < npure(B.L); ++j) {
                for (int p = 0; p < npure(C.L); ++p) {
                  for (int q = 0; q < npure(E.L); ++q) {
                    const size_t row = (size_t)(ma+i)*nbf + (mb+j);
                    const size_t col = (size_t)(mc+p)*nbf + (me+q);
                    g[row*n2 + col] = abce(i,j,p,q);
                  }
                }
              }
            }
          }
        }
      }
    }
    return g;
  }

  /// J[mu,nu] = sum (mu nu|lambda sigma) D[lambda,sigma], by the same
  /// brute-force route and independent of the composite index convention --
  /// which is what makes it a check of the GEMV and not just of the tensor.
  Matrix reference_coulomb(const Basis<Gaussian> &basis, const Matrix &d) {
    Matrix j(basis.nbf());
    const size_t ns = basis.size();
    for (size_t a = 0; a < ns; ++a) {
      for (size_t b = 0; b < ns; ++b) {
        for (size_t c = 0; c < ns; ++c) {
          for (size_t e = 0; e < ns; ++e) {
            const auto &A = basis[a];
            const auto &B = basis[b];
            const auto &C = basis[c];
            const auto &E = basis[e];
            auto abce = zeros(npure(A.L), npure(B.L), npure(C.L), npure(E.L));
            {
              auto cartesian = zeros(
                ncart(A.L), ncart(B.L), ncart(C.L), ncart(E.L)
              );
              libintx::md::reference::compute(A, B, C, E, cartesian);
              libintx::pure::reference::transform(
                A.L, B.L, C.L, E.L, cartesian, abce
              );
            }
            const int ma = basis.range(a).begin();
            const int mb = basis.range(b).begin();
            const int mc = basis.range(c).begin();
            const int me = basis.range(e).begin();
            for (int i = 0; i < npure(A.L); ++i) {
              for (int jj = 0; jj < npure(B.L); ++jj) {
                for (int p = 0; p < npure(C.L); ++p) {
                  for (int q = 0; q < npure(E.L); ++q) {
                    j(ma+i, mb+jj) += abce(i,jj,p,q)*d(mc+p, me+q);
                  }
                }
              }
            }
          }
        }
      }
    }
    return j;
  }

  /// The device buffer, copied back.
  std::vector<double> jformat(const Basis<Gaussian> &basis) {
    const size_t size = gpu::eri::format_size(basis);
    gpu::device::vector<double> g;
    g.resize(size);
    gpu::eri::jformat(basis, g.data());
    std::vector<double> h(size);
    gpu::memcpy(h.data(), g.data(), sizeof(double)*size);
    return h;
  }

  void check_jformat(const Basis<Gaussian> &basis) {
    const size_t nbf = basis.nbf();
    REQUIRE(nbf > 0);
    const size_t n2 = nbf*nbf;

    CHECK(gpu::eri::format_size(basis) == n2*n2);
    CHECK(gpu::eri::format_fits(basis));

    const auto g = jformat(basis);
    const auto reference = reference_jformat(basis);
    REQUIRE(g.size() == reference.size());

    // Every element, not a sample: the kernel fills the whole tensor, and a
    // hole shows up here as a zero.
    for (size_t r = 0; r < n2; ++r) {
      for (size_t c = 0; c < n2; ++c) {
        auto expect = test::ReferenceValue(reference[r*n2 + c], 1e-9).at(r,c);
        CHECK(g[r*n2 + c] == expect);
      }
    }

    // (mu nu|lambda sigma) = (lambda sigma|mu nu): the matrix is symmetric, so
    // a caller may read it either way round and dsymv applies.
    for (size_t r = 0; r < n2; ++r) {
      for (size_t c = 0; c < r; ++c) {
        auto expect = test::ReferenceValue(g[c*n2 + r], 1e-12).at(r,c);
        CHECK(g[r*n2 + c] == expect);
      }
    }

    // ... and (mu nu|..) = (nu mu|..), which is a different symmetry and the
    // one that catches a row index assembled the wrong way round.
    for (size_t mu = 0; mu < nbf; ++mu) {
      for (size_t nu = 0; nu < nbf; ++nu) {
        for (size_t c = 0; c < n2; ++c) {
          auto expect = test::ReferenceValue(
            g[(nu*nbf + mu)*n2 + c], 1e-12
          ).at(mu,nu,c);
          CHECK(g[(mu*nbf + nu)*n2 + c] == expect);
        }
      }
    }

    // J = G . vec(D).
    const auto d = random_density(nbf);
    const auto j = reference_coulomb(basis, d);
    for (size_t r = 0; r < n2; ++r) {
      double v = 0;
      for (size_t c = 0; c < n2; ++c) {
        v += g[r*n2 + c]*d.data[c];
      }
      auto expect = test::ReferenceValue(j.data[r], 1e-9).at(r);
      CHECK(v == expect);
    }
  }

}

TEST_CASE("gpu.eri.jformat.ss") {
  check_jformat(make_test_basis({0,0,0}, 1));
}

TEST_CASE("gpu.eri.jformat.sp") {
  check_jformat(make_test_basis({0,1,1}, 1));
}

TEST_CASE("gpu.eri.jformat.spd") {
  check_jformat(make_test_basis({0,1,2}, 1));
}

TEST_CASE("gpu.eri.jformat.contracted") {
  check_jformat(make_test_basis({0,1,0}, 3));
}

TEST_CASE("gpu.eri.jformat.mixed") {
  check_jformat(make_mixed_basis());
}

TEST_CASE("gpu.eri.jformat.f") {
  if (LMAX < 3) return;
  check_jformat(make_test_basis({0,3}, 1));
}
