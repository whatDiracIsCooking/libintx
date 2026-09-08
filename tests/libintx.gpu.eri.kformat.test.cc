#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "test.h"

#include "libintx/gpu/eri.h"
#include "libintx/gpu/api/api.h"
#include "libintx/ao/md/kengine.h"
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

  KEngine::TileIn tile_in(const Matrix &m) {
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

  KEngine::TileOut tile_out(Matrix &m) {
    return [&m](KEngine::TileIndex i, KEngine::TileIndex j, const double *in) {
      if (!in) return true;
      size_t nj = j.second - j.first;
      for (size_t p = i.first; p < i.second; ++p) {
        for (size_t q = j.first; q < j.second; ++q) {
          m(p,q) = in[(p - i.first)*nj + (q - j.first)];
        }
      }
      return true;
    };
  }

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

  Basis<Gaussian> make_mixed_basis() {
    Basis<Gaussian> basis;
    basis.push_back(test::gaussian(0, 1, true));
    basis.push_back(test::gaussian(1, 3, true));
    if (LMAX >= 2) basis.push_back(test::gaussian(2, 2, true));
    basis.push_back(test::gaussian(0, 2, true));
    return basis;
  }

  /// The full ERI in K layout, G[(mu,nu),(lambda,sigma)] = (mu lambda|nu
  /// sigma), by brute force over every shell quartet with no permutational
  /// symmetry and every integral from libintx::md::reference. Shares no code
  /// with the kernel.
  std::vector<double> reference_kformat(const Basis<Gaussian> &basis) {
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
                    // (mu lambda | nu sigma) with mu = a:i, lambda = b:j,
                    // nu = c:p, sigma = e:q.
                    const size_t row = (size_t)(ma+i)*nbf + (mc+p);
                    const size_t col = (size_t)(mb+j)*nbf + (me+q);
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

  std::vector<double> format(
    const Basis<Gaussian> &basis,
    void (*build)(const Basis<Gaussian>&, double*, gpuStream_t))
  {
    const size_t size = gpu::eri::format_size(basis);
    gpu::device::vector<double> g;
    g.resize(size);
    build(basis, g.data(), 0);
    std::vector<double> h(size);
    gpu::memcpy(h.data(), g.data(), sizeof(double)*size);
    return h;
  }

  void check_kformat(const Basis<Gaussian> &basis) {
    const size_t nbf = basis.nbf();
    REQUIRE(nbf > 0);
    const size_t n2 = nbf*nbf;

    const auto g = format(basis, &gpu::eri::kformat);
    const auto reference = reference_kformat(basis);
    REQUIRE(g.size() == reference.size());

    for (size_t r = 0; r < n2; ++r) {
      for (size_t c = 0; c < n2; ++c) {
        auto expect = test::ReferenceValue(reference[r*n2 + c], 1e-9).at(r,c);
        CHECK(g[r*n2 + c] == expect);
      }
    }

    // The transpose of (mu lambda|nu sigma) is (lambda mu|sigma nu), which is
    // the same integral -- so this matrix is symmetric too, and dsymv applies.
    for (size_t r = 0; r < n2; ++r) {
      for (size_t c = 0; c < r; ++c) {
        auto expect = test::ReferenceValue(g[c*n2 + r], 1e-12).at(r,c);
        CHECK(g[r*n2 + c] == expect);
      }
    }

    // G_K[(mu,nu),(lambda,sigma)] == G_J[(mu,lambda),(nu,sigma)]: the two
    // formats are one tensor with axes 1 and 2 transposed. Cheap, and it pins
    // the two kernels to each other rather than only to the reference.
    const auto j = format(basis, &gpu::eri::jformat);
    REQUIRE(j.size() == g.size());
    for (size_t mu = 0; mu < nbf; ++mu) {
      for (size_t nu = 0; nu < nbf; ++nu) {
        for (size_t lambda = 0; lambda < nbf; ++lambda) {
          for (size_t sigma = 0; sigma < nbf; ++sigma) {
            const size_t k = (mu*nbf + nu)*n2 + (lambda*nbf + sigma);
            const size_t jj = (mu*nbf + lambda)*n2 + (nu*nbf + sigma);
            auto expect = test::ReferenceValue(j[jj], 1e-12).at(mu,nu,lambda,sigma);
            CHECK(g[k] == expect);
          }
        }
      }
    }

    // K = G . vec(D), against the host K engine -- which is itself checked
    // against a brute-force sum in libintx.kengine.test, so this is an
    // independent check of both the layout and the orbit handling.
    const auto d = random_density(nbf);
    Matrix k(nbf);
    libintx::md::make_kengine(basis)->K(tile_in(d), tile_out(k), nullptr);
    for (size_t r = 0; r < n2; ++r) {
      double v = 0;
      for (size_t c = 0; c < n2; ++c) {
        v += g[r*n2 + c]*d.data[c];
      }
      auto expect = test::ReferenceValue(k.data[r], 1e-9).at(r);
      CHECK(v == expect);
    }
  }

}

TEST_CASE("gpu.eri.kformat.ss") {
  check_kformat(make_test_basis({0,0,0}, 1));
}

TEST_CASE("gpu.eri.kformat.sp") {
  check_kformat(make_test_basis({0,1,1}, 1));
}

TEST_CASE("gpu.eri.kformat.spd") {
  check_kformat(make_test_basis({0,1,2}, 1));
}

TEST_CASE("gpu.eri.kformat.contracted") {
  check_kformat(make_test_basis({0,1,0}, 3));
}

TEST_CASE("gpu.eri.kformat.mixed") {
  check_kformat(make_mixed_basis());
}

TEST_CASE("gpu.eri.kformat.f") {
  if (LMAX < 3) return;
  check_kformat(make_test_basis({0,3}, 1));
}
