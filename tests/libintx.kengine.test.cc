#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "test.h"
#include "kengine.test.h"

#include "libintx/ao/md/kengine.h"
#include "libintx/ao/md/reference.h"
#include "libintx/pure.reference.h"

#include <vector>

using namespace libintx;
using libintx::test::zeros;
using libintx::test::kengine::Matrix;
using libintx::test::kengine::tile_in;
using libintx::test::kengine::tile_out;
using libintx::test::kengine::random_density;
using libintx::test::kengine::make_test_basis;

namespace {

  /// K[mu,nu] = sum (mu lambda | nu sigma) D[lambda,sigma], summed over every
  /// shell quartet with no permutational symmetry at all and every integral
  /// taken from libintx::md::reference. Slow and stupid on purpose: it shares
  /// no code with the engine, so agreement is evidence rather than tautology.
  Matrix reference_exchange(const Basis<Gaussian> &basis, const Matrix &d) {
    Matrix k(basis.nbf());
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
                    k(ma+i, mc+p) += abce(i,j,p,q)*d(mb+j, me+q);
                  }
                }
              }
            }
          }
        }
      }
    }
    return k;
  }

  void check_kengine(const Basis<Gaussian> &basis, float threshold) {
    const size_t n = basis.nbf();
    REQUIRE(n > 0);

    auto d = random_density(n);
    auto ref = reference_exchange(basis, d);

    std::shared_ptr<const KEngine::Screening> screening;
    if (threshold > 0) {
      screening = libintx::md::make_schwarz_screening(basis, threshold);
    }
    auto engine = libintx::md::make_kengine(basis, screening);

    Matrix k(n);
    engine->K(tile_in(d), tile_out(k), nullptr);

    // Screened runs are only accurate to the screening threshold; the
    // unscreened one has to reproduce the reference to round-off.
    const double epsilon = (threshold > 0 ? 10.0*threshold : 1e-9);
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        auto expect = test::ReferenceValue(ref(i,j), epsilon).at(i,j);
        CHECK(k(i,j) == expect);
      }
    }
  }

}

TEST_CASE("libintx.kengine.symmetry") {
  // K is symmetric for a symmetric D; a digest that mishandles one leg of the
  // permutation orbit breaks this before it breaks the values.
  auto basis = make_test_basis({0,1,0,2}, 2);
  const size_t n = basis.nbf();
  auto d = random_density(n);
  auto engine = libintx::md::make_kengine(basis);
  Matrix k(n);
  engine->K(tile_in(d), tile_out(k), nullptr);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < i; ++j) {
      auto expect = test::ReferenceValue(k(j,i), 1e-12).at(i,j);
      CHECK(k(i,j) == expect);
    }
  }
}

TEST_CASE("libintx.kengine.allsum") {
  // AllSum sees the assembled K before the tiles go out, and whatever it
  // leaves behind is what the caller receives.
  auto basis = make_test_basis({0,1}, 1);
  const size_t n = basis.nbf();
  auto d = random_density(n);
  auto engine = libintx::md::make_kengine(basis);
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

TEST_CASE("libintx.kengine.ss") {
  check_kengine(make_test_basis({0,0,0}, 1), 0);
}

TEST_CASE("libintx.kengine.sp") {
  check_kengine(make_test_basis({0,1,1}, 1), 0);
}

TEST_CASE("libintx.kengine.spd") {
  check_kengine(make_test_basis({0,1,2}, 1), 0);
}

TEST_CASE("libintx.kengine.contracted") {
  // Contracted shells of differing depth: the case where taking coefficients
  // "as given" instead of primitive-normalized silently produces a different
  // basis rather than a rescaled one.
  Basis<Gaussian> basis;
  for (int L : {0,1,0}) {
    if (L > LMAX) continue;
    basis.push_back(gto::normalized<Shell>(test::gaussian(L, 3, true)));
  }
  basis.push_back(gto::normalized<Shell>(test::gaussian(0, 1, true)));
  check_kengine(basis, 0);
}

TEST_CASE("libintx.kengine.screened") {
  check_kengine(make_test_basis({0,1,2,0}, 2), 1e-12f);
}

TEST_CASE("libintx.kengine.f") {
  // Only meaningful when the build was configured for f: at LIBINTX_MAX_L < 3
  // the kernel table has no entry and make_test_basis would silently drop the
  // shell, leaving a weaker version of the spd case above.
  if (LMAX < 3) return;
  check_kengine(make_test_basis({0,1,3}, 1), 0);
}
