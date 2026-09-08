#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "test.h"
#include "kengine.test.h"

#include "libintx/ao/md/kengine.h"

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
using libintx::test::kengine::reference_df_exchange;

namespace {

  /// An auxiliary basis, built the same way as an AO one but bounded by XMAX
  /// rather than LMAX -- the three-centre kernel table is instantiated over
  /// (X, C, D) with X up to LIBINTX_MAX_X.
  Basis<Gaussian> make_df_basis(const std::vector<int> &Ls, int K) {
    Basis<Gaussian> basis;
    for (int L : Ls) {
      if (L > XMAX) continue;
      basis.push_back(test::gaussian(L, K, /*pure=*/true));
    }
    return basis;
  }

  /// A dense naux x naux matrix with no symmetry at all.
  ///
  /// The engine's contract is "replace X with W X"; a real caller passes the
  /// symmetric V^-1, which cannot tell a correct application from a
  /// transposed one. This can.
  Eigen::MatrixXd random_metric_transform(size_t n) {
    Eigen::MatrixXd W(n,n);
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        W(i,j) = test::random<double>(-1.0, 1.0);
      }
    }
    return W;
  }

  /// The engine against reference_df_exchange: same fitting coefficients, the
  /// three-centre integrals from libintx::md::reference instead of the MD
  /// engine, and the contraction written out as loops instead of GEMMs.
  void check_df_kengine(
    const Basis<Gaussian> &basis,
    const Basis<Gaussian> &df_basis,
    float threshold)
  {
    const size_t n = basis.nbf();
    REQUIRE(n > 0);
    REQUIRE(df_basis.nbf() > 0);

    auto d = random_density(n);
    auto W = random_metric_transform(df_basis.nbf());
    auto ref = reference_df_exchange(basis, df_basis, d, W);

    std::shared_ptr<const KEngine::Screening> screening;
    if (threshold > 0) {
      screening = libintx::md::make_schwarz_screening(basis, threshold);
    }
    auto engine = libintx::md::make_df_kengine(
      basis, df_basis, metric_transform(W), screening
    );

    Matrix k(n);
    engine->K(tile_in(d), tile_out(k), nullptr);

    // Screened runs are only accurate to the screening threshold; the
    // unscreened one has to reproduce the reference to round-off.
    const double epsilon = (threshold > 0 ? 10.0*threshold : 1e-8);
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        auto expect = test::ReferenceValue(ref(i,j), epsilon).at(i,j);
        CHECK(k(i,j) == expect);
      }
    }
  }

}

TEST_CASE("libintx.df.kengine.ss") {
  check_df_kengine(make_test_basis({0,0,0}, 1), make_df_basis({0,0}, 1), 0);
}

TEST_CASE("libintx.df.kengine.sp") {
  check_df_kengine(make_test_basis({0,1,1}, 1), make_df_basis({0,1}, 1), 0);
}

TEST_CASE("libintx.df.kengine.spd") {
  check_df_kengine(make_test_basis({0,1,2}, 1), make_df_basis({0,1,2}, 1), 0);
}

TEST_CASE("libintx.df.kengine.contracted") {
  // Contracted shells of differing depth on both sides: the auxiliary bra and
  // the AO ket are binned on contraction degree independently, and md's
  // three-centre basis packing asserts rather than pads if a batch mixes two.
  Basis<Gaussian> basis;
  for (int L : {0,1,0}) {
    if (L > LMAX) continue;
    basis.push_back(gto::normalized<Shell>(test::gaussian(L, 3, true)));
  }
  basis.push_back(gto::normalized<Shell>(test::gaussian(0, 1, true)));

  Basis<Gaussian> df_basis;
  df_basis.push_back(gto::normalized<Shell>(test::gaussian(0, 2, true)));
  df_basis.push_back(gto::normalized<Shell>(test::gaussian(0, 1, true)));
  if (XMAX >= 1) {
    df_basis.push_back(gto::normalized<Shell>(test::gaussian(1, 2, true)));
  }
  check_df_kengine(basis, df_basis, 0);
}

TEST_CASE("libintx.df.kengine.screened") {
  check_df_kengine(
    make_test_basis({0,1,2,0}, 2), make_df_basis({0,1,2}, 1), 1e-12f
  );
}

TEST_CASE("libintx.df.kengine.f") {
  // Only meaningful when the build was configured for f -- below that
  // make_test_basis drops the shell and this is a weaker copy of spd above.
  if (LMAX < 3) return;
  check_df_kengine(make_test_basis({0,1,3}, 1), make_df_basis({0,1,2}, 1), 0);
}

TEST_CASE("libintx.df.kengine.symmetry") {
  // With the real V^-1 -- symmetric -- K is symmetric for a symmetric D, even
  // though no single auxiliary function's contribution B_P D A_P is.
  auto basis = make_test_basis({0,1,0,2}, 2);
  auto df_basis = make_df_basis({0,1,2}, 1);
  const size_t n = basis.nbf();
  auto d = random_density(n);
  auto engine = libintx::md::make_df_kengine(
    basis, df_basis, metric_transform(reference_metric(df_basis).inverse())
  );
  Matrix k(n);
  engine->K(tile_in(d), tile_out(k), nullptr);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < i; ++j) {
      auto expect = test::ReferenceValue(k(j,i), 1e-9).at(i,j);
      CHECK(k(i,j) == expect);
    }
  }
}

TEST_CASE("libintx.df.kengine.allsum") {
  // AllSum sees the assembled K before the tiles go out, and whatever it
  // leaves behind is what the caller receives -- same contract as the direct
  // engine's.
  auto basis = make_test_basis({0,1}, 1);
  auto df_basis = make_df_basis({0,0}, 1);
  const size_t n = basis.nbf();
  auto d = random_density(n);
  auto engine = libintx::md::make_df_kengine(
    basis, df_basis, metric_transform(reference_metric(df_basis).inverse())
  );
  Matrix k(n);
  size_t seen = 0;
  engine->K(
    tile_in(d), tile_out(k),
    [&](double *v, size_t size) {
      seen = size;
      for (size_t i = 0; i < size; ++i) v[i] *= 2.0;
    }
  );
  CHECK(seen == n*n);

  Matrix plain(n);
  engine->K(tile_in(d), tile_out(plain), nullptr);
  for (size_t i = 0; i < n*n; ++i) {
    auto expect = test::ReferenceValue(2.0*plain.data[i], 1e-12).at(i);
    CHECK(k.data[i] == expect);
  }
}

TEST_CASE("libintx.df.kengine.exact_fit") {
  // Density fitting is a projection of each product density onto the span of
  // the auxiliary basis, in the Coulomb metric. Put every product density
  // *in* that span and the projection is the identity, so DF-K has to
  // reproduce the direct four-centre K to round-off.
  //
  // Uncontracted s shells make that constructible: the product of two s
  // primitives with exponents a1 and a2 is an s primitive with exponent
  // a1+a2 at their centre of charge, so three auxiliary functions span the
  // three products of a two-shell basis exactly.
  //
  // This is the test that separates the DF *mathematics* -- V^-1 applied over
  // the right index, the right pair of contractions -- from the assembly the
  // other cases check. Nothing here is tuned: change the metric transform to
  // its transpose, or contract the two tensors the other way round, and the
  // agreement is gone.
  const double a1 = 0.37, a2 = 0.61;
  const array<double,3> r1 = { 0.0, 0.0, 0.0 };
  const array<double,3> r2 = { 0.3, -0.7, 0.5 };
  auto s_shell = [](double a, const array<double,3> &r) {
    std::vector<Gaussian::Primitive> ps = {{ a, 1.0 }};
    return gto::normalized<Shell>(Gaussian(0, r, ps, /*pure=*/true));
  };

  Basis<Gaussian> basis;
  basis.push_back(s_shell(a1, r1));
  basis.push_back(s_shell(a2, r2));

  Basis<Gaussian> df_basis;
  df_basis.push_back(s_shell(2*a1, r1));
  df_basis.push_back(s_shell(2*a2, r2));
  df_basis.push_back(s_shell(a1+a2, center_of_charge(a1, r1, a2, r2)));

  const size_t n = basis.nbf();
  auto d = random_density(n);

  Matrix direct(n), df(n);
  libintx::md::make_kengine(basis)->K(tile_in(d), tile_out(direct), nullptr);
  libintx::md::make_df_kengine(
    basis, df_basis, metric_transform(reference_metric(df_basis).inverse())
  )->K(tile_in(d), tile_out(df), nullptr);

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      auto expect = test::ReferenceValue(direct(i,j), 1e-8).at(i,j);
      CHECK(df(i,j) == expect);
    }
  }
}
