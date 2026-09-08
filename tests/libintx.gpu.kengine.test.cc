#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "test.h"
#include "kengine.test.h"

#include "libintx/ao/md/kengine.h"
#include "libintx/gpu/kengine.h"

#include <Eigen/Dense>
#include <vector>

using namespace libintx;
using libintx::test::kengine::Matrix;
using libintx::test::kengine::tile_in;
using libintx::test::kengine::tile_out;
using libintx::test::kengine::random_density;
using libintx::test::kengine::make_test_basis;
using libintx::test::kengine::metric_transform;
using libintx::test::kengine::reference_metric;

namespace {

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
          // (double) deliberately: max2 returns float, and against
          // ReferenceValue the float overload set is ambiguous under C++20's
          // reversed operator== rewriting.
          CHECK((double)gpu_screening->max2((int)i,(int)j) == expect);
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

namespace {

  /// The host DF engine is the reference for the device one, the same way the
  /// host direct engine is for the device direct engine: the two share
  /// libintx/gpu/kengine/md/df.h, so what this pins down is the device
  /// three-centre engine and the pinned-memory path. The DF assembly itself is
  /// pinned down against libintx::md::reference by libintx.df.kengine.test.
  void check_gpu_df_kengine(
    const Basis<Gaussian> &basis,
    const Basis<Gaussian> &df_basis,
    float threshold)
  {
    const size_t n = basis.nbf();
    REQUIRE(n > 0);
    auto d = random_density(n);
    auto v_linv = metric_transform(reference_metric(df_basis).inverse());

    std::shared_ptr<const KEngine::Screening> host_screening, gpu_screening;
    if (threshold > 0) {
      host_screening = libintx::md::make_schwarz_screening(basis, threshold);
      gpu_screening = libintx::gpu::make_schwarz_screening(basis, threshold);
    }

    Matrix host(n), device(n);
    libintx::md::make_df_kengine(basis, df_basis, v_linv, host_screening)
      ->K(tile_in(d), tile_out(host), nullptr);
    libintx::gpu::make_df_kengine(basis, df_basis, v_linv, gpu_screening)
      ->K(tile_in(d), tile_out(device), nullptr);

    const double epsilon = (threshold > 0 ? 10.0*threshold : 1e-8);
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        auto expect = test::ReferenceValue(host(i,j), epsilon).at(i,j);
        CHECK(device(i,j) == expect);
      }
    }
  }

}

TEST_CASE("gpu.df.kengine.ss") {
  check_gpu_df_kengine(
    make_test_basis({0,0,0}, 1), make_test_basis({0,0}, 1), 0
  );
}

TEST_CASE("gpu.df.kengine.spd") {
  check_gpu_df_kengine(
    make_test_basis({0,1,2}, 1), make_test_basis({0,1,2}, 1), 0
  );
}

TEST_CASE("gpu.df.kengine.contracted") {
  check_gpu_df_kengine(
    make_test_basis({0,1,0}, 3), make_test_basis({0,1}, 2), 0
  );
}

TEST_CASE("gpu.df.kengine.screened") {
  check_gpu_df_kengine(
    make_test_basis({0,1,2,0}, 2), make_test_basis({0,1,2}, 1), 1e-12f
  );
}
