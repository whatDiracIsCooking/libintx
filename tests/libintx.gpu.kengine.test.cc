#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "test.h"

#include "libintx/ao/md/kengine.h"
#include "libintx/gpu/kengine.h"

#include <vector>

using namespace libintx;

namespace {

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

  /// The host engine is the reference here rather than a brute-force sum: the
  /// two share the driver, so what this pins down is the device four-centre
  /// engine and the pinned-memory path, and libintx.kengine.test is what pins
  /// down the driver itself against libintx::md::reference.
  void check_gpu_kengine(const Basis<Gaussian> &basis, float threshold) {
    const size_t n = basis.nbf();
    REQUIRE(n > 0);
    auto d = random_density(n);

    std::shared_ptr<const KEngine::Screening> host_screening, gpu_screening;
    if (threshold > 0) {
      host_screening = libintx::md::make_schwarz_screening(basis, threshold);
      gpu_screening = libintx::gpu::make_schwarz_screening(basis, threshold);
      // The two Schwarz passes are the same integrals on different hardware.
      for (size_t i = 0; i < basis.size(); ++i) {
        for (size_t j = 0; j < basis.size(); ++j) {
          auto expect = test::ReferenceValue(
            host_screening->max2((int)i,(int)j), 1e-5
          ).at(i,j);
          CHECK(gpu_screening->max2((int)i,(int)j) == expect);
        }
      }
    }

    Matrix host(n), device(n);
    libintx::md::make_kengine(basis, host_screening)
      ->K(tile_in(d), tile_out(host), nullptr);
    libintx::gpu::make_kengine(basis, gpu_screening)
      ->K(tile_in(d), tile_out(device), nullptr);

    const double epsilon = (threshold > 0 ? 10.0*threshold : 1e-9);
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        auto expect = test::ReferenceValue(host(i,j), epsilon).at(i,j);
        CHECK(device(i,j) == expect);
      }
    }
  }

}

TEST_CASE("gpu.kengine.ss") {
  check_gpu_kengine(make_test_basis({0,0,0}, 1), 0);
}

TEST_CASE("gpu.kengine.sp") {
  check_gpu_kengine(make_test_basis({0,1,1}, 1), 0);
}

TEST_CASE("gpu.kengine.spd") {
  check_gpu_kengine(make_test_basis({0,1,2}, 1), 0);
}

TEST_CASE("gpu.kengine.contracted") {
  check_gpu_kengine(make_test_basis({0,1,0}, 3), 0);
}

TEST_CASE("gpu.kengine.screened") {
  check_gpu_kengine(make_test_basis({0,1,2,0}, 2), 1e-12f);
}
