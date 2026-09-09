// Derivative ERI batches: gpu::md::IntegralEngine<4>::compute1 and
// gpu::md::IntegralEngine<3>::compute1, at the *integral* level.
//
// The oracle is the engine's own value path with a shell centre moved, so this
// file references nothing -- no libintx::md::reference, no host engine. What
// it checks is exactly the claim the derivative batch makes: that
// `compute1(centre, ...)` is d/d(centre) of `compute(...)`.
//
//  - Central finite differences, five-point at h = 0.0025. Two points leave
//    ~1e-6 and cannot see a wrong 2*alpha; five leave ~1e-9, well inside the
//    1e-7 the cases assert. (This is the stencil libintx.gpu.md2.test's
//    overlap_gradient_reference uses, for the same reason.)
//  - Translational invariance, elementwise: sum over all centres is zero to
//    round-off. It references nothing at all and catches component mixing --
//    a swapped pair of Cartesian components passes no finite-difference test
//    but would also fail this one, and this one is ~50x cheaper.
//
// Mutation coverage is not expressible here (a test cannot mutate the kernel
// it links). The derivative coefficient itself is mutation-tested in the host
// shim harness described in CLAUDE.md.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "test.h"

#include "libintx/gpu/api/api.h"
#include "libintx/gpu/engine.h"

#include <set>

using namespace libintx;
using libintx::test::zeros;

namespace {

  namespace gpu = libintx::gpu;

  /// Five-point central difference: sum of these weights over `h*offset`,
  /// divided by 12h.
  constexpr double fd_h = 0.0025;
  constexpr double fd_w[4] = { -1.0, +8.0, -8.0, +1.0 };
  constexpr double fd_d[4] = { +2.0, +1.0, -1.0, -2.0 };

  /// A copy of `basis` with the named shells moved by `dx` along axis `x`.
  /// Rebuilt rather than edited in place: Basis<Shell> maintains a
  /// basis-function range table and exposes only a const `operator[]`.
  Basis<Gaussian> shifted(
    const Basis<Gaussian> &basis, const std::vector<int> &shells,
    int x, double dx)
  {
    std::set<int> move(shells.begin(), shells.end());
    Basis<Gaussian> out;
    int i = 0;
    for (auto g : basis) {
      if (move.count(i++)) g.r[x] += dx;
      out.push_back(g);
    }
    return out;
  }

  /// The shell indices a pair list uses in slot 0 or 1. `test::make_basis`
  /// gives every pair its own two shells, so moving all of slot `s` at once is
  /// exactly "move this pair's slot-`s` centre", per pair.
  std::vector<int> slot(const std::vector<Index2> &idx, int s) {
    std::vector<int> v;
    for (auto [i,j] : idx) v.push_back(s ? j : i);
    return v;
  }

  std::vector<int> slot(const std::vector<Index1> &idx) {
    return std::vector<int>(idx.begin(), idx.end());
  }

  /// Relative difference with an absolute floor, so a near-zero element is not
  /// compared against the finite difference's own noise.
  void close(double got, double ref, double scale, double tol) {
    double m = std::max({std::abs(got), std::abs(ref), scale});
    CHECK(std::abs(got-ref) <= tol*m);
  }

  // ------------------------------------------------------------------ md4

  void eri4_gradient_subcase(
    int A, int B, int C, int D, std::pair<int,int> K = {1,1})
  {

    printf("d(%i%i|%i%i)/dX K={%i,%i}\n", A, B, C, D, K.first, K.second);

    const int M = 5, N = 4;
    const int NA = npure(A), NB = npure(B), NC = npure(C), ND = npure(D);
    const size_t rows = (size_t)M*NA*NB;
    const size_t cols = (size_t)NC*ND*N;

    auto [bra,ijs] = test::make_basis<2>({A,B}, {K.first,1}, M);
    auto [ket,kls] = test::make_basis<2>({C,D}, {K.second,1}, N);

    gpuStream_t stream = 0;

    auto value = [&](const Basis<Gaussian> &p, const Basis<Gaussian> &q) {
      auto v = zeros(M,NA,NB,NC,ND,N);
      gpu::host::register_pointer(v.data(), v.size());
      auto md = gpu::integral_engine<4>(p, q, stream);
      md->compute(Coulomb, ijs, kls, {}, v.data(), {rows,cols});
      gpu::stream::synchronize(stream);
      gpu::host::unregister_pointer(v.data());
      return v;
    };

    // the analytic batches, one per centre; component is the slowest index
    std::vector<decltype(zeros(M,NA,NB,NC,ND,N,3))> analytic;
    for (int centre = 0; centre < 4; ++centre) {
      auto g = zeros(M,NA,NB,NC,ND,N,3);
      gpu::host::register_pointer(g.data(), g.size());
      auto md = gpu::integral_engine<4>(bra, ket, stream);
      md->compute1(Coulomb, centre, ijs, kls, {}, g.data(), {rows,cols});
      gpu::stream::synchronize(stream);
      gpu::host::unregister_pointer(g.data());
      analytic.push_back(std::move(g));
    }

    // a scale for the absolute floor: the largest value in the batch
    double scale = 0;
    {
      auto v = value(bra,ket);
      for (int i = 0; i < v.size(); ++i) {
        scale = std::max(scale, std::abs(v.data()[i]));
      }
      scale = std::max(scale, 1e-6)*1e-4;
    }

    for (int centre = 0; centre < 4; ++centre) {
      auto shells = slot(centre < 2 ? ijs : kls, centre%2);
      for (int x = 0; x < 3; ++x) {
        auto fd = zeros(M,NA,NB,NC,ND,N);
        for (int s = 0; s < 4; ++s) {
          double dx = fd_d[s]*fd_h;
          auto v = (centre < 2
            ? value(shifted(bra,shells,x,dx), ket)
            : value(bra, shifted(ket,shells,x,dx)));
          for (int i = 0; i < fd.size(); ++i) {
            fd.data()[i] += fd_w[s]*v.data()[i]/(12*fd_h);
          }
        }
        const auto &g = analytic[centre];
        for (int i = 0; i < fd.size(); ++i) {
          close(g.data()[i + x*fd.size()], fd.data()[i], scale, 1e-7);
        }
      }
    }

    // translational invariance, elementwise -- references nothing
    {
      size_t n = (size_t)analytic[0].size();
      for (size_t i = 0; i < n; ++i) {
        double s = 0;
        for (int centre = 0; centre < 4; ++centre) s += analytic[centre].data()[i];
        close(s, 0.0, scale, 1e-9);
      }
    }

  }

  // ------------------------------------------------------------------ md3

  void eri3_gradient_subcase(int X, int C, int D, std::pair<int,int> K = {1,1})
  {

    printf("d(%i|%i%i)/dX K={%i,%i}\n", X, C, D, K.first, K.second);

    const int M = 5, N = 4;
    const int NX = npure(X), NC = npure(C), ND = npure(D);
    const size_t rows = (size_t)M*NX;
    const size_t cols = (size_t)NC*ND*N;

    auto [bra,is] = test::make_basis<1>({X}, {K.first}, M);
    auto [ket,kls] = test::make_basis<2>({C,D}, {K.second,1}, N);

    gpuStream_t stream = 0;

    auto value = [&](const Basis<Gaussian> &p, const Basis<Gaussian> &q) {
      auto v = zeros(M,NX,NC,ND,N);
      gpu::host::register_pointer(v.data(), v.size());
      auto md = gpu::integral_engine<3>(p, q, stream);
      md->compute(Coulomb, is, kls, {}, v.data(), {rows,cols});
      gpu::stream::synchronize(stream);
      gpu::host::unregister_pointer(v.data());
      return v;
    };

    std::vector<decltype(zeros(M,NX,NC,ND,N,3))> analytic;
    for (int centre = 1; centre <= 2; ++centre) {
      auto g = zeros(M,NX,NC,ND,N,3);
      gpu::host::register_pointer(g.data(), g.size());
      auto md = gpu::integral_engine<3>(bra, ket, stream);
      md->compute1(Coulomb, centre, is, kls, {}, g.data(), {rows,cols});
      gpu::stream::synchronize(stream);
      gpu::host::unregister_pointer(g.data());
      analytic.push_back(std::move(g));
    }

    double scale = 0;
    {
      auto v = value(bra,ket);
      for (int i = 0; i < v.size(); ++i) {
        scale = std::max(scale, std::abs(v.data()[i]));
      }
      scale = std::max(scale, 1e-6)*1e-4;
    }

    for (int centre = 1; centre <= 2; ++centre) {
      auto shells = slot(kls, centre-1);
      for (int x = 0; x < 3; ++x) {
        auto fd = zeros(M,NX,NC,ND,N);
        for (int s = 0; s < 4; ++s) {
          auto v = value(bra, shifted(ket,shells,x,fd_d[s]*fd_h));
          for (int i = 0; i < fd.size(); ++i) {
            fd.data()[i] += fd_w[s]*v.data()[i]/(12*fd_h);
          }
        }
        const auto &g = analytic[centre-1];
        for (int i = 0; i < fd.size(); ++i) {
          close(g.data()[i + x*fd.size()], fd.data()[i], scale, 1e-7);
        }
      }
    }

    // The auxiliary centre is not computed; d/dP = -(d/dC + d/dD) is what
    // makes that legitimate, so check it against a finite difference in P
    // rather than assume it.
    {
      auto shells = slot(is);
      for (int x = 0; x < 3; ++x) {
        auto fd = zeros(M,NX,NC,ND,N);
        for (int s = 0; s < 4; ++s) {
          auto v = value(shifted(bra,shells,x,fd_d[s]*fd_h), ket);
          for (int i = 0; i < fd.size(); ++i) {
            fd.data()[i] += fd_w[s]*v.data()[i]/(12*fd_h);
          }
        }
        size_t n = (size_t)fd.size();
        for (size_t i = 0; i < n; ++i) {
          double dP = -(analytic[0].data()[i + x*n] + analytic[1].data()[i + x*n]);
          close(dP, fd.data()[i], scale, 1e-7);
        }
      }
    }

  }

}

#define ERI4_GRADIENT_SUBCASE(A,B,C,D,Ks)                       \
  if (test::enabled(A,B,C,D)) {                                 \
    SUBCASE(str("d(",A,B,"|",C,D,")/dX").c_str()) {             \
      for (auto K : Ks) { eri4_gradient_subcase(A,B,C,D,K); }   \
    }                                                           \
  }

#define ERI3_GRADIENT_SUBCASE(X,C,D,Ks)                         \
  if (test::enabled(X,C,D)) {                                   \
    SUBCASE(str("d(",X,"|",C,D,")/dX").c_str()) {               \
      for (auto K : Ks) { eri3_gradient_subcase(X,C,D,K); }     \
    }                                                           \
  }

/// The full (A,B|C,D) sweep, uncontracted. (LMAX+1)^4 bins is 256 at LMAX=3
/// and each runs 12 analytic plus 48 value batches, so this is the expensive
/// case; the contraction sweep below is deliberately not crossed with it.
TEST_CASE("gpu.deriv.eri4") {
  const std::vector< std::pair<int,int> > K = { {1,1} };
  for (int A = 0; A <= LMAX; ++A) {
    for (int B = 0; B <= LMAX; ++B) {
      for (int C = 0; C <= LMAX; ++C) {
        for (int D = 0; D <= LMAX; ++D) {
          ERI4_GRADIENT_SUBCASE(A,B,C,D,K);
        }
      }
    }
  }
}

/// Contraction depth, on the diagonal of the sweep above. A wrong exponent in
/// the derivative relation -- the ket's for the bra's, say -- is invisible at
/// K = 1 for an uncontracted pair whose two shells happen to share one, and
/// this is what sees it.
TEST_CASE("gpu.deriv.eri4.contracted") {
  const std::vector< std::pair<int,int> > K = { {1,5}, {3,5} };
  for (int L = 0; L <= LMAX; ++L) {
    ERI4_GRADIENT_SUBCASE(L,L,L,L,K);
    ERI4_GRADIENT_SUBCASE(L,0,0,L,K);
    ERI4_GRADIENT_SUBCASE(0,L,L,0,K);
  }
}

TEST_CASE("gpu.deriv.eri3") {
  const std::vector< std::pair<int,int> > K = { {1,1}, {1,5}, {3,5} };
  for (int X = 0; X <= XMAX; ++X) {
    for (int C = 0; C <= LMAX; ++C) {
      for (int D = 0; D <= LMAX; ++D) {
        ERI3_GRADIENT_SUBCASE(X,C,D,K);
      }
    }
  }
}

/// What the entry points refuse, and why. A silently-zero derivative reads as
/// a converged gradient, so every unimplemented case has to throw.
TEST_CASE("gpu.deriv.interface") {

  gpuStream_t stream = 0;

  auto [bra,ijs] = test::make_basis<2>({0,0}, {1,1}, 2);
  auto [aux,is] = test::make_basis<1>({0}, {1}, 2);

  SUBCASE("md4 centre range") {
    auto md = gpu::integral_engine<4>(bra, bra, stream);
    auto v = zeros(2,1,1,1,1,2,3);
    gpu::host::register_pointer(v.data(), v.size());
    CHECK_THROWS(md->compute1(Coulomb, -1, ijs, ijs, {}, v.data(), {2,2}));
    CHECK_THROWS(md->compute1(Coulomb, 4, ijs, ijs, {}, v.data(), {2,2}));
    CHECK_NOTHROW(md->compute1(Coulomb, 0, ijs, ijs, {}, v.data(), {2,2}));
    gpu::stream::synchronize(stream);
    gpu::host::unregister_pointer(v.data());
  }

  SUBCASE("md3 has no auxiliary-centre derivative") {
    auto md = gpu::integral_engine<3>(aux, bra, stream);
    auto v = zeros(2,1,1,1,2,3);
    gpu::host::register_pointer(v.data(), v.size());
    // d/dP is the caller's, as -(d/dC + d/dD); the engine says so rather than
    // returning zeros.
    CHECK_THROWS(md->compute1(Coulomb, 0, is, ijs, {}, v.data(), {2,2}));
    CHECK_THROWS(md->compute1(Coulomb, 3, is, ijs, {}, v.data(), {2,2}));
    CHECK_NOTHROW(md->compute1(Coulomb, 1, is, ijs, {}, v.data(), {2,2}));
    gpu::stream::synchronize(stream);
    gpu::host::unregister_pointer(v.data());
  }

  SUBCASE("the host engines throw") {
    auto md4 = ao::integral_engine<4>(bra, bra);
    auto md3 = ao::integral_engine<3>(aux, bra);
    auto v = zeros(2,1,1,1,1,2,3);
    CHECK_THROWS(md4->compute1(Coulomb, 0, ijs, ijs, {}, v.data(), {2,2}));
    CHECK_THROWS(md3->compute1(Coulomb, 1, is, ijs, {}, v.data(), {2,2}));
  }

}
