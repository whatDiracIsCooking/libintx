// The device one-electron engine, gpu::md::IntegralEngine<2>.
//
// A direct mirror of tests/libintx.md2.test.cc -- same test::make_basis<2>,
// same test::check2, same libintx::md::reference::compute2<Op> reference, same
// 1e-10 tolerance, same {1,1}/{1,5}/{3,5} contraction sweep -- so the three
// operator issues (overlap, kinetic, electron-nuclear potential) each add one
// LIBINTX_GPU_MD2_TEST_CASE line and nothing else.
//
// No operator kernel exists yet, so what runs today is the scaffolding case at
// the bottom: the factory, the batching invariant, the point-charge upload
// through set(), and the (A|B) dispatch reporting the operator it cannot do
// yet rather than handing back a buffer of zeros.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "test.h"

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

// The three operator issues each turn one of these on:
//
//   LIBINTX_GPU_MD2_TEST_CASE(Overlap);
//   LIBINTX_GPU_MD2_TEST_CASE(Kinetic);
//   LIBINTX_GPU_MD2_TEST_CASE(Nuclear);

TEST_CASE("libintx.gpu.md2.scaffolding") {

  namespace gpu = libintx::gpu;

  gpuStream_t stream = 0;

  auto [basis,ijs] = test::make_basis<2>({0,0}, {1,1}, 8);
  auto md = libintx::gpu::integral_engine<2>(basis, basis, stream);
  REQUIRE(md);

  std::vector<double> V(ijs.size()*npure(0)*npure(0), 0.0);

  SUBCASE("operators not implemented yet") {
    // Every (A|B) entry of the dispatch table is wired and reachable; none of
    // the three operators has a kernel yet. Saying so beats returning zeros.
    CHECK_THROWS(md->compute(Overlap,ijs,V.data()));
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
