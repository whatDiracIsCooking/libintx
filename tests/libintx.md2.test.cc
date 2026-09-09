#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "test.h"

#include "libintx/ao/md/reference.h"
#include "libintx/ao/md/engine.h"
#include "libintx/pure.reference.h"

using namespace libintx;
using test::zeros;

template<typename Operator>
auto params(Operator op) {
  if constexpr (op == Nuclear) {
    std::vector< std::tuple<int, std::array<double,3> > > params;
    for (size_t i = 0; i < 8+1; ++i) {
      params.push_back(
        {
          test::random<int>(1,100),
          test::random<double,3>(-1,+1)
        }
      );
    }
    return params;
  }
  else {
    return None;
  }
}

template<typename Operator>
void md2_test_subcase(Operator op, int A, int B, std::pair<int,int> K = {1,1}) {

  printf("(%i|%i) K={%i,%i}\n", A, B, K.first, K.second);

  int M = 32+3;
  int N = 16+1;

  int NA = npure(A);
  int NB = npure(B);

  auto [basis,ijs] = test::make_basis<2>({A,B}, {K.first,K.second}, M*N);

  auto result = zeros(ijs.size(),NA,NB);
  auto md = libintx::ao::integral_engine<2>(basis, basis);
  auto op_params = params(op);
  if constexpr (op == Nuclear) {
    md->set(typename Operator::Operator::Parameters{op_params});
  }
  md->compute(op,ijs,result.data());

  for (size_t ij = 0; ij < ijs.size(); ++ij) {
    auto [i,j] = ijs[ij];
    auto ab_ref = zeros(npure(A), npure(B));
    {
      auto ab_cartesian = zeros(ncart(A), ncart(B));
      libintx::md::reference::compute2<op>(basis[i], basis[j], op_params, ab_cartesian);
      libintx::pure::reference::transform(
        A, B,
        ab_cartesian,
        ab_ref
      );
    }
    test::check2(
      [&](auto ref, auto ... idx) {
        //printf("(%i) = %f\n", ij, result(ij,idx...));
        CHECK(result(ij,idx...) == ref.epsilon(1e-10));
      },
      ab_ref
    );
  }

}

#define LIBINTX_MD2_TEST_SUBCASE(Operator,A,B,Ks)               \
  if (test::enabled(A,B)) {                                     \
    SUBCASE(str(#Operator " (A|B)=(",A,B,")").c_str()) {        \
      printf(# Operator "\n");                                  \
      for (auto K : Ks) {                                       \
        md2_test_subcase(Operator,A,B,K);                       \
      }                                                         \
    }                                                           \
  }

std::vector< std::pair<int,int> > Ks = {
  {1,1}, {1,5}, {3,5}
};

#define LIBINTX_MD2_TEST_CASE(Operator)                 \
  TEST_CASE("libintx.md2." # Operator) {                \
    for (int a = 0; a <= LMAX; ++a) {                   \
      for (int b = 0; b <= LMAX; ++b) {                 \
        LIBINTX_MD2_TEST_SUBCASE(Operator,a,b,Ks);      \
      }                                                 \
    }                                                   \
  }

LIBINTX_MD2_TEST_CASE(Overlap);
LIBINTX_MD2_TEST_CASE(Kinetic);
LIBINTX_MD2_TEST_CASE(Nuclear);

// The host one-electron gradients, dO/dA_x, through
// md::IntegralEngine<2>::compute1.
//
// These are the first gradient cases in the tree that a CPU build can execute.
// Everything under tests/libintx.gpu.* needs CUDA to compile, so until this
// file carried them, nothing in CI -- which builds no CUDA at all -- gated any
// derivative code.
//
// Oracle: central finite differences of libintx::md::reference::compute2<Op>,
// five-point at h = 0.0025, so the tolerance is set by the kernel and not by
// the stencil. The engine appears on only one side of the comparison, and the
// reference shares no code with it.
namespace {

  /// `g` with its centre displaced by `h` along axis `x`. The primitives are
  /// copied verbatim -- moving a shell does not renormalize it.
  Gaussian displaced(const Gaussian &g, int x, double h) {
    auto r = g.r;
    r[x] += h;
    std::vector<Gaussian::Primitive> prims(g.K);
    for (int k = 0; k < g.K; ++k) prims[k] = g.prims[k];
    return Gaussian(g.L, r, prims, g.pure);
  }

  /// The solid-harmonic `(a|Op|b)` block of one shell pair, from the reference.
  template<typename Op>
  auto value_reference(Op, const Gaussian &a, const Gaussian &b) {
    auto pure = zeros(npure(a.L), npure(b.L));
    auto cart = zeros(ncart(a.L), ncart(b.L));
    libintx::md::reference::compute2<Op{}>(a, b, None, cart);
    libintx::pure::reference::transform(a.L, b.L, cart, pure);
    return pure;
  }

  /// `d/dA_x` of that block, by central differences on the bra centre:
  ///
  ///     df/dx ~ [ -f(2h) + 8*f(h) - 8*f(-h) + f(-2h) ] / (12h)
  template<typename Op>
  auto gradient_reference(Op op, const Gaussian &a, const Gaussian &b, int x) {
    const double h = 0.0025;
    const double w[4] = { -1.0, +8.0, -8.0, +1.0 };
    const double d[4] = { +2*h, +h, -h, -2*h };
    auto g = zeros(npure(a.L), npure(b.L));
    for (int s = 0; s < 4; ++s) {
      auto f = value_reference(op, displaced(a,x,d[s]), b);
      for (int nb = 0; nb < npure(b.L); ++nb) {
        for (int na = 0; na < npure(a.L); ++na) {
          g(na,nb) += w[s]*f(na,nb)/(12*h);
        }
      }
    }
    return g;
  }

  /// One bin through compute1.
  template<typename Op>
  auto gradient(
    Op op, const Basis<Gaussian> &basis, const std::vector<Index2> &ijs,
    int A, int B)
  {
    auto G = zeros(ijs.size(), npure(A), npure(B), 3);
    auto md = libintx::ao::integral_engine<2>(basis, basis);
    md->compute1(op,ijs,G.data());
    return G;
  }

  /// The (A|B) x {1,1}/{1,5}/{3,5} sweep against those finite differences.
  ///
  /// Written once over the operator: only the kernels differ between Overlap
  /// and Kinetic, while what a derivative has to satisfy does not.
  template<typename Op>
  void gradient_sweep(Op op) {
    for (int A = 0; A <= LMAX; ++A) {
      for (int B = 0; B <= LMAX; ++B) {
        if (!test::enabled(A,B)) continue;
        SUBCASE(str("(",A,"|",B,")").c_str()) {
          for (auto K : Ks) {
            printf("(%i|%i) K={%i,%i}\n", A, B, K.first, K.second);
            auto [basis,ijs] = test::make_basis<2>({A,B}, {K.first,K.second}, 8);
            auto G = gradient(op, basis, ijs, A, B);
            for (size_t ij = 0; ij < ijs.size(); ++ij) {
              auto [i,j] = ijs[ij];
              for (int x = 0; x < 3; ++x) {
                auto ref = gradient_reference(op, basis[i], basis[j], x);
                for (int nb = 0; nb < npure(B); ++nb) {
                  for (int na = 0; na < npure(A); ++na) {
                    auto v = test::ReferenceValue(ref(na,nb)).at(ij,na,nb,x);
                    CHECK(G(ij,na,nb,x) == v.epsilon(1e-7));
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  /// Translational invariance, elementwise -- and the sharpest check
  /// available, because it references nothing.
  ///
  /// dO/dA + dO/dB = 0, and only the bra derivative is computed, so the
  /// relation is a tautology unless the ket derivative is obtained
  /// independently. It is: both operators are symmetric, O(a,b) = O(b,a), so
  /// the bra derivative of the swapped (B|A) bin IS dO/dB of this one, and
  ///
  ///     G_(A|B)[ij,na,nb,x] == -G_(B|A)[ji,nb,na,x]
  ///
  /// walks both index orders and both contraction depths to say so. For
  /// Kinetic it is also the derivative analogue of compute2's transpose
  /// branch, which the derivative path deliberately does not replicate.
  template<typename Op>
  void gradient_translation(Op op) {

    const int n = 5;

    for (int A = 0; A <= LMAX; ++A) {
      for (int B = 0; B <= LMAX; ++B) {
        if (!test::enabled(A,B)) continue;
        SUBCASE(str("(",A,"|",B,") + (",B,"|",A,")^T == 0").c_str()) {

          Basis<Gaussian> basis;
          for (int i = 0; i < n; ++i) basis.push_back(test::gaussian(A,3));
          for (int j = 0; j < n; ++j) basis.push_back(test::gaussian(B,1));

          std::vector<Index2> ab, ba;
          for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
              ab.push_back({i,n+j});
              ba.push_back({n+j,i});
            }
          }

          auto Gab = gradient(op, basis, ab, A, B);
          auto Gba = gradient(op, basis, ba, B, A);

          for (size_t ij = 0; ij < ab.size(); ++ij) {
            for (int x = 0; x < 3; ++x) {
              for (int nb = 0; nb < npure(B); ++nb) {
                for (int na = 0; na < npure(A); ++na) {
                  auto ref = test::ReferenceValue(-Gba(ij,nb,na,x)).at(ij,na,nb,x);
                  CHECK(Gab(ij,na,nb,x) == ref.epsilon(1e-10));
                }
              }
            }
          }

        }
      }
    }

    // Both shells on one centre. dO/dA + dO/dB = 0 makes the *atom* derivative
    // vanish, which is what a caller's scatter has to reproduce by
    // accumulating +V and -V into the same slot. The kernel's own dO/dA for
    // such a pair is generally NOT zero above L = 0 -- the sweep above is what
    // pins that -- so this checks finiteness, and exact zero only at L = 0.
    for (int L = 0; L <= LMAX; ++L) {
      if (!test::enabled(L,L)) continue;
      SUBCASE(str("(",L,"|",L,") on one centre").c_str()) {
        auto g = test::gaussian(L,3);
        Basis<Gaussian> basis;
        basis.push_back(g);
        basis.push_back(g);
        std::vector<Index2> ijs = { {0,1} };
        auto G = gradient(op, basis, ijs, L, L);
        for (int x = 0; x < 3; ++x) {
          for (int nb = 0; nb < npure(L); ++nb) {
            for (int na = 0; na < npure(L); ++na) {
              CHECK(std::isfinite(G(0,na,nb,x)));
              if (L == 0) {
                auto ref = test::ReferenceValue(0.0).at(na,nb,x);
                CHECK(G(0,na,nb,x) == ref.epsilon(1e-12));
              }
            }
          }
        }
      }
    }

  }

}

TEST_CASE("libintx.md2.Overlap.gradient") { gradient_sweep(Overlap); }
TEST_CASE("libintx.md2.Kinetic.gradient") { gradient_sweep(Kinetic); }

TEST_CASE("libintx.md2.Overlap.gradient.translation") {
  gradient_translation(Overlap);
}
TEST_CASE("libintx.md2.Kinetic.gradient.translation") {
  gradient_translation(Kinetic);
}

// What compute1 answers and what it refuses.
//
// Nuclear is the interesting one: it has no host kernel *yet*, but even when it
// gets one it will throw here, because its Hellmann-Feynman term is indexed by
// nucleus and does not fit this buffer. Coulomb is not a two-centre operator
// this engine implements on either path.
TEST_CASE("libintx.md2.gradient.interface") {

  auto [basis,ijs] = test::make_basis<2>({0,0}, {1,1}, 8);
  auto md = libintx::ao::integral_engine<2>(basis, basis);
  std::vector<double> G(ijs.size()*npure(0)*npure(0)*3, 0.0);
  std::vector<double> GC(G.size()*2, 0.0);

  CHECK_NOTHROW(md->compute1(Overlap,ijs,G.data()));
  CHECK_NOTHROW(md->compute1(Kinetic,ijs,G.data()));
  CHECK_THROWS(md->compute1(Nuclear,ijs,G.data()));
  CHECK_THROWS(md->compute1(Coulomb,ijs,G.data()));
  CHECK_THROWS(md->compute1(Nuclear,ijs,G.data(),GC.data()));

  // The x = 0 block has exactly the shape and stride `compute` writes, which
  // is the half of the layout contract a value comparison cannot see.
  CHECK(G.size() == 3*ijs.size()*npure(0)*npure(0));
}
