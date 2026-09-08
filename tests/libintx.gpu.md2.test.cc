// The device one-electron engine, gpu::md::IntegralEngine<2>.
//
// A direct mirror of tests/libintx.md2.test.cc -- same test::make_basis<2>,
// same test::check2, same libintx::md::reference::compute2<Op> reference, same
// 1e-10 tolerance, same {1,1}/{1,5}/{3,5} contraction sweep -- so the three
// operator issues (overlap, kinetic, electron-nuclear potential) each turn on
// one LIBINTX_GPU_MD2_TEST_CASE line and add only whatever is specific to that
// operator.
//
// Every case additionally compares against the HOST md::IntegralEngine<2> on
// the same input. The reference pins the values; the host engine pins the
// output layout, which is the whole point of a device engine that claims to be
// a drop-in replacement for it.
//
// Overlap has a kernel (gpu/overlap/); kinetic and the electron-nuclear
// potential do not, which is what the scaffolding case at the bottom still
// checks -- the factory, the batching invariant, the point-charge upload
// through set(), and the (A|B) dispatch reporting the operator it cannot do
// yet rather than handing back a buffer of zeros.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "test.h"

#include "libintx/ao/engine.h"
#include "libintx/ao/md/reference.h"
#include "libintx/pure.reference.h"

#include "libintx/gpu/api/api.h"
#include "libintx/gpu/engine.h"

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
void gpu_md2_test_subcase(Operator op, int A, int B, std::pair<int,int> K = {1,1}) {

  namespace gpu = libintx::gpu;

  printf("(%i|%i) K={%i,%i}\n", A, B, K.first, K.second);

  int M = 32+3;
  int N = 16+1;

  int NA = npure(A);
  int NB = npure(B);

  auto [basis,ijs] = test::make_basis<2>({A,B}, {K.first,K.second}, M*N);

  auto result = zeros(ijs.size(),NA,NB);
  gpu::host::register_pointer(result.data(), result.size());

  gpuStream_t stream = 0;
  auto md = libintx::gpu::integral_engine<2>(basis, basis, stream);
  auto op_params = params(op);
  if constexpr (op == Nuclear) {
    md->set(typename Operator::Operator::Parameters{op_params});
  }
  md->compute(op,ijs,result.data());
  gpu::stream::synchronize(stream);

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
        CHECK(result(ij,idx...) == ref.epsilon(1e-10));
      },
      ab_ref
    );
  }

  // The same input through the host engine. The reference above has already
  // pinned the values; what this adds is the layout -- the device engine
  // promises `V[ij + (na + nb*npure(A))*ldV]` byte for byte, and a transposed
  // or mis-strided write is invisible to a per-pair value check on a symmetric
  // operator.
  {
    auto host = zeros(ijs.size(),NA,NB);
    auto md_host = libintx::ao::integral_engine<2>(basis, basis);
    if constexpr (op == Nuclear) {
      md_host->set(typename Operator::Operator::Parameters{op_params});
    }
    md_host->compute(op,ijs,host.data());
    for (size_t ij = 0; ij < ijs.size(); ++ij) {
      for (int nb = 0; nb < NB; ++nb) {
        for (int na = 0; na < NA; ++na) {
          auto ref = test::ReferenceValue(host(ij,na,nb)).at(ij,na,nb);
          CHECK(result(ij,na,nb) == ref.epsilon(1e-10));
        }
      }
    }
  }

  gpu::host::unregister_pointer(result.data());

}

#define LIBINTX_GPU_MD2_TEST_SUBCASE(Operator,A,B,Ks)           \
  if (test::enabled(A,B)) {                                     \
    SUBCASE(str(#Operator " (A|B)=(",A,B,")").c_str()) {        \
      printf(# Operator "\n");                                  \
      for (auto K : Ks) {                                       \
        gpu_md2_test_subcase(Operator,A,B,K);                   \
      }                                                         \
    }                                                           \
  }

std::vector< std::pair<int,int> > Ks = {
  {1,1}, {1,5}, {3,5}
};

#define LIBINTX_GPU_MD2_TEST_CASE(Operator)                     \
  TEST_CASE("libintx.gpu.md2." # Operator) {                    \
    for (int a = 0; a <= LMAX; ++a) {                           \
      for (int b = 0; b <= LMAX; ++b) {                         \
        LIBINTX_GPU_MD2_TEST_SUBCASE(Operator,a,b,Ks);          \
      }                                                         \
    }                                                           \
  }

LIBINTX_GPU_MD2_TEST_CASE(Overlap);

// The two remaining operator issues each turn one of these on:
//
//   LIBINTX_GPU_MD2_TEST_CASE(Kinetic);
//   LIBINTX_GPU_MD2_TEST_CASE(Nuclear);

// A shell scaled to unit norm: primitive-normalized coefficients (the
// convention libintx::make_basis(..., normalize=true) applies, and the one the
// whole tree assumes), then the contraction scaled so <g|g> = 1.
//
// test::gaussian's coefficients are neither, deliberately -- a basis whose
// overlap is the identity would hide an index or layout mistake.
inline auto normalized(const Gaussian &g) {
  auto n = libintx::gto::normalized<Shell>(g);
  double f = libintx::gto::normalization_factor<Shell>(n);
  std::vector<Gaussian::Primitive> prims(n.K);
  for (int k = 0; k < n.K; ++k) {
    prims[k] = { n.prims[k].a, f*n.prims[k].C };
  }
  return Gaussian(n.L, n.r, prims, n.pure);
}

// S is the operator where a normalization mistake looks least like a bug: it
// comes out symmetric, positive definite and plausible, just wrong. These two
// properties catch the whole class, and neither is visible in the per-pair
// sweep above, which compares against a reference that would carry the same
// mistake.
TEST_CASE("libintx.gpu.md2.Overlap.normalization") {

  namespace gpu = libintx::gpu;

  gpuStream_t stream = 0;
  const int n = 8;

  for (int L = 0; L <= LMAX; ++L) {
    for (int K : { 1, 3 }) {

      if (!test::enabled(L,L)) continue;

      SUBCASE(str("(",L,"|",L,") K=",K).c_str()) {

        Basis<Gaussian> basis;
        std::vector<Index2> ijs;
        for (int i = 0; i < n; ++i) {
          basis.push_back(normalized(test::gaussian(L,K)));
        }
        // Every (i,j), so the batch is one bin and carries both triangles.
        for (int i = 0; i < n; ++i) {
          for (int j = 0; j < n; ++j) {
            ijs.push_back({i,j});
          }
        }

        int N = npure(L);
        auto S = zeros(ijs.size(),N,N);
        gpu::host::register_pointer(S.data(), S.size());
        auto md = libintx::gpu::integral_engine<2>(basis, basis, stream);
        md->compute(Overlap,ijs,S.data());
        gpu::stream::synchronize(stream);
        gpu::host::unregister_pointer(S.data());

        for (int i = 0; i < n; ++i) {
          for (int j = 0; j < n; ++j) {
            for (int a = 0; a < N; ++a) {
              for (int b = 0; b < N; ++b) {
                // S == S^T, across the shell index and the component index at
                // once: <mu|nu> = <nu|mu>.
                auto s = test::ReferenceValue(S(j*n+i,b,a)).at(i,j,a,b);
                CHECK(S(i*n+j,a,b) == s.epsilon(1e-10));
              }
            }
          }
          // A normalized shell against itself is the identity: unit diagonal,
          // and the solid harmonics of one shell are orthogonal on their own
          // centre.
          for (int a = 0; a < N; ++a) {
            for (int b = 0; b < N; ++b) {
              auto s = test::ReferenceValue(a == b ? 1.0 : 0.0).at(i,a,b);
              CHECK(S(i*n+i,a,b) == s.epsilon(1e-10));
            }
          }
        }

      }
    }
  }

}

TEST_CASE("libintx.gpu.md2.scaffolding") {

  namespace gpu = libintx::gpu;

  gpuStream_t stream = 0;

  auto [basis,ijs] = test::make_basis<2>({0,0}, {1,1}, 8);
  auto md = libintx::gpu::integral_engine<2>(basis, basis, stream);
  REQUIRE(md);

  std::vector<double> V(ijs.size()*npure(0)*npure(0), 0.0);

  SUBCASE("operators not implemented yet") {
    // Every (A|B) entry of the dispatch table is wired and reachable; kinetic
    // and the electron-nuclear potential have no kernel yet. Saying so beats
    // returning zeros.
    CHECK_THROWS(md->compute(Kinetic,ijs,V.data()));
  }

  SUBCASE("nuclear parameters") {
    // Nuclear without set() is an error, not a silent zero.
    CHECK_THROWS(md->compute(Nuclear,ijs,V.data()));
    auto p = params(Nuclear);
    md->set(Nuclear::Operator::Parameters{p});
    // Uploaded now; a second set() replaces it rather than being ignored.
    md->set(Nuclear::Operator::Parameters{params(Nuclear)});
    CHECK_THROWS(md->compute(Nuclear,ijs,V.data()));
  }

  SUBCASE("a batch is one bin") {
    // Angular momentum, the solid-harmonic flag and the contraction degree all
    // have to agree across the batch; nothing pads.
    auto [mixed,mixed_ijs] = test::make_basis<2>({0,0}, {1,1}, 4);
    mixed.push_back(test::gaussian(1,1));
    mixed.push_back(test::gaussian(0,1));
    mixed_ijs.push_back({(int)mixed.size()-2, (int)mixed.size()-1});
    auto engine = libintx::gpu::integral_engine<2>(mixed, mixed, stream);
    CHECK_THROWS(engine->compute(Overlap,mixed_ijs,V.data()));

    auto [contracted,contracted_ijs] = test::make_basis<2>({0,0}, {1,1}, 4);
    contracted.push_back(test::gaussian(0,3));
    contracted.push_back(test::gaussian(0,1));
    contracted_ijs.push_back({(int)contracted.size()-2, (int)contracted.size()-1});
    auto engine2 = libintx::gpu::integral_engine<2>(contracted, contracted, stream);
    CHECK_THROWS(engine2->compute(Overlap,contracted_ijs,V.data()));
  }

  SUBCASE("empty batch") {
    std::vector<Index2> none;
    CHECK_THROWS(md->compute(Overlap,none,V.data()));
  }

}
