#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "test.h"

#include "libintx/ao/md/engine.h"
#include "libintx/ao/md/jengine.h"
#include "libintx/ao/md/kengine.h"
#include "libintx/ao/md/reference.h"
#include "libintx/fock/md/driver.h"
#include "libintx/pure.reference.h"

#include <vector>

using namespace libintx;
using libintx::test::zeros;

namespace {

  /// A dense row-major nbf x nbf matrix, and the tile callbacks that hand it
  /// to a JEngine. Deliberately the dumbest possible caller: every tile is
  /// wanted, every tile is available.
  struct Matrix {
    size_t n;
    std::vector<double> data;
    explicit Matrix(size_t n) : n(n), data(n*n, 0.0) {}
    double& operator()(size_t i, size_t j) { return data[i*n + j]; }
    double operator()(size_t i, size_t j) const { return data[i*n + j]; }
  };

  JEngine::TileIn tile_in(const Matrix &m) {
    return [&m](JEngine::TileIndex i, JEngine::TileIndex j, double *out) {
      size_t nj = j.second - j.first;
      for (size_t p = i.first; p < i.second; ++p) {
        for (size_t q = j.first; q < j.second; ++q) {
          out[(p - i.first)*nj + (q - j.first)] = m(p,q);
        }
      }
      return true;
    };
  }

  JEngine::TileOut tile_out(Matrix &m) {
    return [&m](JEngine::TileIndex i, JEngine::TileIndex j, const double *in) {
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

  /// A symmetric density with no structure the engine could exploit by
  /// accident.
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

  /// J[mu,nu] = sum (mu nu | lambda sigma) D[lambda,sigma], summed over every
  /// shell quartet with no permutational symmetry at all and every integral
  /// taken from libintx::md::reference. Slow and stupid on purpose: it shares
  /// no code with the engine, so agreement is evidence rather than tautology.
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
              for (int k = 0; k < npure(B.L); ++k) {
                for (int p = 0; p < npure(C.L); ++p) {
                  for (int q = 0; q < npure(E.L); ++q) {
                    j(ma+i, mb+k) += abce(i,k,p,q)*d(mc+p, me+q);
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

  /// A small basis with a mix of angular momenta and contraction depths --
  /// the cases the digest has to keep straight are equal shells (a==b),
  /// equal pairs ((ab)==(cd)) and mixed classes, so the basis has to contain
  /// repeats of the same L on different centres.
  Basis<Gaussian> make_test_basis(const std::vector<int> &Ls, int K) {
    Basis<Gaussian> basis;
    for (int L : Ls) {
      if (L > LMAX) continue;
      basis.push_back(test::gaussian(L, K, /*pure=*/true));
    }
    return basis;
  }

  void check_jengine(const Basis<Gaussian> &basis, float threshold) {
    const size_t n = basis.nbf();
    REQUIRE(n > 0);

    auto d = random_density(n);
    auto ref = reference_coulomb(basis, d);

    std::shared_ptr<const PairScreening> screening;
    if (threshold > 0) {
      screening = libintx::md::make_schwarz_screening(basis, threshold);
    }
    auto engine = libintx::md::make_jengine(basis, screening);

    Matrix j(n);
    engine->J(tile_in(d), tile_out(j), nullptr);

    // Screened runs are only accurate to the screening threshold; the
    // unscreened one has to reproduce the reference to round-off.
    const double epsilon = (threshold > 0 ? 10.0*threshold : 1e-9);
    for (size_t i = 0; i < n; ++i) {
      for (size_t k = 0; k < n; ++k) {
        auto expect = test::ReferenceValue(ref(i,k), epsilon).at(i,k);
        CHECK(j(i,k) == expect);
      }
    }
  }

}

TEST_CASE("libintx.jengine.symmetry") {
  // J is symmetric for a symmetric D; a digest that mishandles one leg of the
  // permutation orbit breaks this before it breaks the values.
  auto basis = make_test_basis({0,1,0,2}, 2);
  const size_t n = basis.nbf();
  auto d = random_density(n);
  auto engine = libintx::md::make_jengine(basis);
  Matrix j(n);
  engine->J(tile_in(d), tile_out(j), nullptr);
  for (size_t i = 0; i < n; ++i) {
    for (size_t k = 0; k < i; ++k) {
      auto expect = test::ReferenceValue(j(k,i), 1e-12).at(i,k);
      CHECK(j(i,k) == expect);
    }
  }
}

TEST_CASE("libintx.jengine.allsum") {
  // AllSum sees the assembled J before the tiles go out, and whatever it
  // leaves behind is what the caller receives.
  auto basis = make_test_basis({0,1}, 1);
  const size_t n = basis.nbf();
  auto d = random_density(n);
  auto engine = libintx::md::make_jengine(basis);
  Matrix j(n);
  size_t seen = 0;
  engine->J(
    tile_in(d), tile_out(j),
    [&](double *v, size_t size) {
      seen = size;
      for (size_t i = 0; i < size; ++i) v[i] *= 2.0;
    }
  );
  CHECK(seen == n*n);

  Matrix plain(n);
  engine->J(tile_in(d), tile_out(plain), nullptr);
  for (size_t i = 0; i < n*n; ++i) {
    auto expect = test::ReferenceValue(2.0*plain.data[i], 1e-12).at(i);
    CHECK(j.data[i] == expect);
  }
}

TEST_CASE("libintx.jengine.ss") {
  check_jengine(make_test_basis({0,0,0}, 1), 0);
}

TEST_CASE("libintx.jengine.sp") {
  check_jengine(make_test_basis({0,1,1}, 1), 0);
}

TEST_CASE("libintx.jengine.spd") {
  check_jengine(make_test_basis({0,1,2}, 1), 0);
}

TEST_CASE("libintx.jengine.contracted") {
  // Contracted shells of differing depth: the case where taking coefficients
  // "as given" instead of primitive-normalized silently produces a different
  // basis rather than a rescaled one.
  Basis<Gaussian> basis;
  for (int L : {0,1,0}) {
    if (L > LMAX) continue;
    basis.push_back(gto::normalized<Shell>(test::gaussian(L, 3, true)));
  }
  basis.push_back(gto::normalized<Shell>(test::gaussian(0, 1, true)));
  check_jengine(basis, 0);
}

TEST_CASE("libintx.jengine.screened") {
  check_jengine(make_test_basis({0,1,2,0}, 2), 1e-12f);
}

TEST_CASE("libintx.jengine.f") {
  // Only meaningful when the build was configured for f: at LIBINTX_MAX_L < 3
  // the kernel table has no entry and make_test_basis would silently drop the
  // shell, leaving a weaker version of the spd case above.
  if (LMAX < 3) return;
  check_jengine(make_test_basis({0,1,3}, 1), 0);
}

TEST_CASE("libintx.jengine.fused") {
  // The point of parameterising the driver on a *list* of terms: one sweep
  // over the integrals feeding both scatters. This drives fock::md::build
  // directly rather than through an engine, and has to agree element for
  // element with the two engines run separately -- if it does not, the two
  // digests are interfering through the shared orbit or the shared V.
  auto basis = make_test_basis({0,1,2,0}, 2);
  const size_t n = basis.nbf();
  auto density = random_density(n);

  Matrix j_only(n), k_only(n);
  libintx::md::make_jengine(basis)->J(tile_in(density), tile_out(j_only), nullptr);
  libintx::md::make_kengine(basis)->K(tile_in(density), tile_out(k_only), nullptr);

  namespace fmd = libintx::fock::md;
  auto classes = fmd::make_pair_classes(basis, [](int,int) { return 1.0f; });
  auto d = fmd::gather_density(basis, tile_in(density));
  fmd::Matrix j(n), k(n);
  auto shared_basis = std::make_shared< Basis<Gaussian> >(basis);
  libintx::md::IntegralEngine<4> engine(shared_basis);
  fmd::HostBuffer buffer;
  fmd::build(
    basis, classes, engine, buffer, nullptr, 4*1024*1024,
    fmd::coulomb(d, j), fmd::exchange(d, k)
  );

  for (size_t p = 0; p < n; ++p) {
    for (size_t q = 0; q < n; ++q) {
      CHECK(j(p,q) == test::ReferenceValue(j_only(p,q), 1e-12).at(p,q));
      CHECK(k(p,q) == test::ReferenceValue(k_only(p,q), 1e-12).at(p,q));
    }
  }
}
