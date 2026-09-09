// The device one-electron engine, gpu::md::IntegralEngine<2>.
//
// A direct mirror of tests/libintx.md2.test.cc -- same test::make_basis<2>,
// same test::check2, same libintx::md::reference::compute2<Op> reference, same
// 1e-10 tolerance, same {1,1}/{1,5}/{3,5} contraction sweep.
//
// Every case additionally compares against the HOST md::IntegralEngine<2> on
// the same input. The reference pins the values; the host engine pins the
// output layout, which is the whole point of a device engine that claims to be
// a drop-in replacement for it.
//
// All four operators have kernels now -- overlap (gpu/overlap/), kinetic
// (gpu/kinetic/), the electron-nuclear potential (gpu/potential_en/) and the
// two-centre Coulomb metric (gpu/coulomb2/) -- so each adds a case for whatever
// the sweep cannot see: normalization for overlap, the two index orders for
// kinetic, the per-molecule parameter set for the potential, symmetry and
// positive definiteness for the metric.
//
// Coulomb does NOT use LIBINTX_GPU_MD2_TEST_CASE, and cannot: neither
// libintx::md::reference::compute2<Coulomb> nor the host md::IntegralEngine<2>
// implements a two-centre Coulomb integral (the host `compute` has no
// `if (op == Coulomb)` branch at all and leaves the buffer untouched). Its
// sweep carries its own oracle -- see below. The other three do.
//
// All three additionally have a DERIVATIVE kernel on compute1 -- dS/dX, dT/dX
// and dV/dX -- and their cases sit lower down. They carry their own oracle,
// central finite differences of the same libintx::md::reference, because there
// is no host derivative engine to compare against. dS/dX and dT/dX share their
// two cases (gradient_sweep and gradient_translation, written once over the
// operator): what a derivative has to satisfy does not differ between them,
// only the kernel does. dV/dX has its own, because its operator moves too --
// the differences there displace the NUCLEI as well as the shell centres, and
// a sweep over shell centres alone passes with the Hellmann-Feynman term
// entirely absent.
//
// What the scaffolding case at the bottom still checks is the engine around
// them: the factory, the batching invariant, the point-charge upload through
// set(), and that the (A|B) value fallback table is now unreachable -- every
// operator is dispatched before it.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "test.h"

#include "libintx/ao/engine.h"
#include "libintx/ao/md/reference.h"
#include "libintx/pure.reference.h"

#include "libintx/gpu/api/api.h"
#include "libintx/gpu/engine.h"

#include <Eigen/Dense>

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
LIBINTX_GPU_MD2_TEST_CASE(Kinetic);
LIBINTX_GPU_MD2_TEST_CASE(Nuclear);

// Operator::Coulomb -- the two-centre metric (P|Q), the density-fitting metric
// gpu::make_df_jengine and gpu::make_df_kengine make the caller supply V^-1 of.
//
// Three things make this case its own rather than one more
// LIBINTX_GPU_MD2_TEST_CASE line:
//
//  - THE ORACLE. `libintx::md::reference::Integral<Op>` is specialized for
//    Overlap, Kinetic and Nuclear only, so `reference::compute2<Coulomb>` does
//    not compile; and the host `md::IntegralEngine<2>::compute` has no Coulomb
//    branch, so it returns the buffer it was handed, untouched -- a comparison
//    against it would be a comparison against zeros. The oracle here is
//    `reference::compute(P, Unit, Q, Unit, ...)`, the four-centre reference
//    with a unit shell in each ket slot, which is exactly how
//    tests/kengine.test.h's `reference_metric` builds V today. It shares no
//    code with the kernel.
//
//  - THE RANGE. The auxiliary basis reaches XMAX (`LIBINTX_MAX_X`, default
//    LMAX+1), not LMAX, and gpu/coulomb2/ sizes its (A|B) table for
//    max(LMAX,XMAX). `test::enabled(A,B)` bounds at LMAX and is therefore the
//    wrong guard; the loop bound below is CMAX.
//
//  - THE BASIS IS NORMALIZED. `test::gaussian`'s raw `C = 1/a` coefficients put
//    the metric's entries at 1e4-1e6, where the tree's absolute-ish comparison
//    (`max(|a|,|b|,1)*epsilon`) sits below the double-precision noise floor of
//    a sum that large -- (P|Q) grows with the coefficients on BOTH sides of
//    1/r12 and with no overlap factor to damp it, which no one-electron
//    operator does. A real auxiliary basis is primitive-normalized, which is
//    what `libintx::make_basis(..., normalize=true)` applies and what this
//    uses. Measured on the host shim harness: normalized, the worst error over
//    the whole (A|B) x {1,1}/{1,5}/{3,5} sweep at LMAX=3, XMAX=4 is 4e-12;
//    unnormalized it is 2.4e-8 absolute, which is the same 1.5e-11 relative to
//    the block maximum.
namespace coulomb2 {

  /// What this operator's (A|B) table spans -- see gpu/coulomb2/coulomb2.cu.
  constexpr int CMAX = std::max(LMAX,XMAX);

  /// A primitive-normalized auxiliary shell.
  inline auto aux(int L, int K) {
    return libintx::gto::normalized<Shell>(test::gaussian(L,K));
  }

  /// (P|Q) as a pure npure(P) x 1 x npure(Q) x 1 block, from the four-centre
  /// reference with a unit ket on each side.
  inline auto reference(const Gaussian &P, const Gaussian &Q) {
    auto v = zeros(npure(P.L), 1, npure(Q.L), 1);
    auto cartesian = zeros(ncart(P.L), 1, ncart(Q.L), 1);
    libintx::md::reference::compute(
      P, Unit<Gaussian>{}, Q, Unit<Gaussian>{}, cartesian
    );
    libintx::pure::reference::transform(P.L, 0, Q.L, 0, cartesian, v);
    return v;
  }

}

TEST_CASE("libintx.gpu.md2.Coulomb") {

  namespace gpu = libintx::gpu;

  gpuStream_t stream = 0;
  // Fewer pairs than the one-electron sweeps' 595: the oracle is the
  // FOUR-centre reference, which is what makes libintx.md4.test slow.
  const int M = 24;

  for (int A = 0; A <= coulomb2::CMAX; ++A) {
    for (int B = 0; B <= coulomb2::CMAX; ++B) {

      SUBCASE(str("Coulomb (A|B)=(",A,B,")").c_str()) {

        printf("Coulomb\n");

        for (auto K : Ks) {

          printf("(%i|%i) K={%i,%i}\n", A, B, K.first, K.second);

          Basis<Gaussian> basis;
          std::vector<Index2> ijs;
          for (int i = 0; i < M; ++i) {
            basis.push_back(coulomb2::aux(A,K.first));
            basis.push_back(coulomb2::aux(B,K.second));
            ijs.push_back({2*i, 2*i+1});
          }

          int NA = npure(A);
          int NB = npure(B);

          auto result = zeros(ijs.size(),NA,NB);
          gpu::host::register_pointer(result.data(), result.size());
          auto md = libintx::gpu::integral_engine<2>(basis, basis, stream);
          md->compute(Coulomb,ijs,result.data());
          gpu::stream::synchronize(stream);

          for (size_t ij = 0; ij < ijs.size(); ++ij) {
            auto [i,j] = ijs[ij];
            auto ref = coulomb2::reference(basis[i], basis[j]);
            for (int nb = 0; nb < NB; ++nb) {
              for (int na = 0; na < NA; ++na) {
                auto v = test::ReferenceValue(ref(na,0,nb,0)).at(ij,na,nb);
                CHECK(result(ij,na,nb) == v.epsilon(1e-10));
              }
            }
          }

          gpu::host::unregister_pointer(result.data());

        }

      }
    }
  }

}

// What the sweep above cannot see, and what a DF caller actually depends on.
//
// (P|Q) is the one integral in this file whose CONSUMER checks nothing: a DF
// engine inverts V and contracts with it, and a V that is symmetric and
// plausible but wrong produces a smooth, plausible, wrong J or K. So the two
// structural properties are worth stating directly, against no reference:
//
//  - V == V^T. The bin (A|B) and the bin (B|A) are separate kernel launches on
//    separate instantiations, so this walks both index orders -- and it is also
//    the only layout check available here, the host engine having no Coulomb to
//    compare a layout against.
//  - V is positive definite. It is a Gram matrix of the Coulomb inner product,
//    so it must be; a sign or phase error anywhere in the Hermite double sum
//    shows up as an indefinite metric long before it shows up as a wrong SCF.
//    Cholesky is also literally what a DF caller does to it.
TEST_CASE("libintx.gpu.md2.Coulomb.metric") {

  namespace gpu = libintx::gpu;

  gpuStream_t stream = 0;

  // A real auxiliary basis: every angular momentum this operator spans, three
  // shells of each, all at the same contraction depth so a bin is keyed by the
  // two angular momenta alone.
  const int K = 2;
  Basis<Gaussian> aux;
  for (int L = 0; L <= coulomb2::CMAX; ++L) {
    for (int i = 0; i < 3; ++i) aux.push_back(coulomb2::aux(L,K));
  }

  const size_t naux = aux.nbf();
  Eigen::MatrixXd V(naux,naux);
  V.setZero();

  auto md = libintx::gpu::integral_engine<2>(aux, aux, stream);

  for (int A = 0; A <= coulomb2::CMAX; ++A) {
    for (int B = 0; B <= coulomb2::CMAX; ++B) {

      std::vector<Index2> ijs;
      for (size_t i = 0; i < aux.size(); ++i) {
        for (size_t j = 0; j < aux.size(); ++j) {
          if (aux[i].L != A || aux[j].L != B) continue;
          ijs.push_back({(int)i,(int)j});
        }
      }
      if (ijs.empty()) continue;

      int NA = npure(A), NB = npure(B);
      auto block = zeros(ijs.size(),NA,NB);
      gpu::host::register_pointer(block.data(), block.size());
      md->compute(Coulomb,ijs,block.data());
      gpu::stream::synchronize(stream);
      gpu::host::unregister_pointer(block.data());

      for (size_t ij = 0; ij < ijs.size(); ++ij) {
        auto [i,j] = ijs[ij];
        for (int nb = 0; nb < NB; ++nb) {
          for (int na = 0; na < NA; ++na) {
            V(aux.range(i).begin()+na, aux.range(j).begin()+nb) =
              block(ij,na,nb);
          }
        }
      }

    }
  }

  SUBCASE("V == V^T") {
    for (size_t i = 0; i < naux; ++i) {
      for (size_t j = 0; j < naux; ++j) {
        auto v = test::ReferenceValue(V(j,i)).at(i,j);
        CHECK(V(i,j) == v.epsilon(1e-10));
      }
    }
  }

  SUBCASE("V is positive definite") {
    // Not vacuously: a zero matrix is positive SEMI-definite and would pass a
    // sloppier check, so the diagonal is required to be nonzero first.
    for (size_t i = 0; i < naux; ++i) {
      CHECK(V(i,i) > 0);
      CHECK(std::isfinite(V(i,i)));
    }
    Eigen::LLT<Eigen::MatrixXd> llt(V);
    CHECK(llt.info() == Eigen::Success);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(V);
    CHECK(es.info() == Eigen::Success);
    CHECK(es.eigenvalues().minCoeff() > 0);
  }

}

// The three things the (A|B) sweep above cannot see, all of them specific to
// the electron-nuclear potential's per-molecule parameter set.
TEST_CASE("libintx.gpu.md2.Nuclear.parameters") {

  namespace gpu = libintx::gpu;

  gpuStream_t stream = 0;
  const int n = 6;

  for (int L = 0; L <= LMAX; ++L) {

    if (!test::enabled(L,L)) continue;

    SUBCASE(str("(",L,"|",L,")").c_str()) {

      Basis<Gaussian> basis;
      std::vector<Index2> ijs;
      for (int i = 0; i < n; ++i) {
        basis.push_back(test::gaussian(L,3));
      }
      // Every (i,j), so the batch is one bin and carries both triangles.
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
          ijs.push_back({i,j});
        }
      }

      int N = npure(L);
      auto md = libintx::gpu::integral_engine<2>(basis, basis, stream);

      auto compute = [&](const auto &p, auto &V) {
        md->set(Nuclear::Operator::Parameters{p});
        gpu::host::register_pointer(V.data(), V.size());
        md->compute(Nuclear,ijs,V.data());
        gpu::stream::synchronize(stream);
        gpu::host::unregister_pointer(V.data());
      };

      auto p1 = params(Nuclear);
      auto V1 = zeros(ijs.size(),N,N);
      compute(p1, V1);

      // V == V^T, across the shell index and the component index at once:
      // <mu|V|nu> = <nu|V|mu>.
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
          for (int a = 0; a < N; ++a) {
            for (int b = 0; b < N; ++b) {
              auto v = test::ReferenceValue(V1(j*n+i,b,a)).at(i,j,a,b);
              CHECK(V1(i*n+j,a,b) == v.epsilon(1e-10));
            }
          }
        }
      }

      // set() twice with different centres. The upload is per geometry, not
      // per compute, so a cached first upload would silently answer the first
      // set() for ever -- a real bug class for a geometry optimisation or a
      // finite-difference gradient.
      auto p2 = params(Nuclear);
      auto V2 = zeros(ijs.size(),N,N);
      compute(p2, V2);
      {
        double diff = 0;
        for (size_t i = 0; i < V1.size(); ++i) {
          diff = std::max(diff, std::fabs(V1.data()[i] - V2.data()[i]));
        }
        CHECK(diff > 1e-6);
      }
      // ... and back, which pins that it is the parameters being read and not
      // just an upload that moved.
      auto V3 = zeros(ijs.size(),N,N);
      compute(p1, V3);
      for (size_t i = 0; i < V1.size(); ++i) {
        auto v = test::ReferenceValue(V1.data()[i]).at(i);
        CHECK(V3.data()[i] == v.epsilon(1e-10));
      }

      // A Z = 0 nucleus contributes nothing (the host skips it explicitly),
      // and a nucleus sitting exactly on a pair's centre of charge is the
      // T = 0 limit of the Boys function -- finite, and the classic way this
      // integral goes wrong. Both are added to p1, so the answer must not move.
      {
        auto p = p1;
        p.push_back({ 0, test::random<double,3>(-1,+1) });
        p.push_back({ 0, center(basis[0]) });
        auto V4 = zeros(ijs.size(),N,N);
        compute(p, V4);
        for (size_t i = 0; i < V1.size(); ++i) {
          auto v = test::ReferenceValue(V1.data()[i]).at(i);
          CHECK(V4.data()[i] == v.epsilon(1e-10));
        }
      }

      // A charged nucleus exactly on a centre: finite, and the same thing the
      // host gets.
      {
        auto p = p1;
        p.push_back({ 5, center(basis[0]) });
        auto V5 = zeros(ijs.size(),N,N);
        compute(p, V5);
        auto host = zeros(ijs.size(),N,N);
        auto md_host = libintx::ao::integral_engine<2>(basis, basis);
        md_host->set(Nuclear::Operator::Parameters{p});
        md_host->compute(Nuclear,ijs,host.data());
        for (size_t i = 0; i < V5.size(); ++i) {
          CHECK(std::isfinite(V5.data()[i]));
          auto v = test::ReferenceValue(host.data()[i]).at(i);
          CHECK(V5.data()[i] == v.epsilon(1e-10));
        }
      }

    }
  }

}

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

// The two index orders against each other, and T == T^T.
//
// libintx::md::compute2 (src/libintx/ao/md/md2.cc:246-257) evaluates the host
// kinetic integral through a transposing accessor on the swapped, sign-flipped
// pair whenever B > A. The device kernel deliberately does NOT replicate that
// (gpu/kinetic/kinetic.cu says why -- on this skeleton the swap costs shared
// memory rather than saving it), but the failure mode the branch exists to
// produce is worth pinning down on both engines: get an accessor or the sign
// of R wrong and (f|s) is right while (s|f) is transposed nonsense. The
// reference sweep above walks both orders, so it would catch it too; this
// checks the relation directly, against no reference at all.
//
// Both cases run at every (A,B) up to LMAX -- and (f|s)/(s|f) is exactly what
// a green run at LIBINTX_MAX_L=2 has not executed.
TEST_CASE("libintx.gpu.md2.Kinetic.transpose") {

  namespace gpu = libintx::gpu;

  gpuStream_t stream = 0;
  const int n = 5;

  auto kinetic = [&](
    const Basis<Gaussian> &basis, const std::vector<Index2> &ijs, int NA, int NB)
  {
    auto T = zeros(ijs.size(),NA,NB);
    gpu::host::register_pointer(T.data(), T.size());
    auto md = libintx::gpu::integral_engine<2>(basis, basis, stream);
    md->compute(Kinetic,ijs,T.data());
    gpu::stream::synchronize(stream);
    gpu::host::unregister_pointer(T.data());
    return T;
  };

  for (int A = 0; A <= LMAX; ++A) {
    for (int B = 0; B <= LMAX; ++B) {

      if (!test::enabled(A,B)) continue;

      SUBCASE(str("(",A,"|",B,") == (",B,"|",A,")^T").c_str()) {

        // Two families of shells, deliberately of different contraction depth:
        // the (A|B) bin is K = 3*1 and the (B|A) bin K = 1*3, so a kernel that
        // confused the bra and the ket primitive loops would not survive this
        // either.
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

        auto Tab = kinetic(basis, ab, npure(A), npure(B));
        auto Tba = kinetic(basis, ba, npure(B), npure(A));

        for (size_t ij = 0; ij < ab.size(); ++ij) {
          for (int a = 0; a < npure(A); ++a) {
            for (int b = 0; b < npure(B); ++b) {
              auto ref = test::ReferenceValue(Tba(ij,b,a)).at(ij,a,b);
              CHECK(Tab(ij,a,b) == ref.epsilon(1e-10));
            }
          }
        }

      }
    }
  }

  // T == T^T within one bin: the same shells on both sides, every (i,j), so
  // the batch carries both triangles and the check is across the shell index
  // and the component index at once.
  for (int L = 0; L <= LMAX; ++L) {

    if (!test::enabled(L,L)) continue;

    SUBCASE(str("(",L,"|",L,") == T^T").c_str()) {

      Basis<Gaussian> basis;
      std::vector<Index2> ijs;
      for (int i = 0; i < n; ++i) basis.push_back(test::gaussian(L,3));
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
          ijs.push_back({i,j});
        }
      }

      int N = npure(L);
      auto T = kinetic(basis, ijs, N, N);

      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
          for (int a = 0; a < N; ++a) {
            for (int b = 0; b < N; ++b) {
              auto ref = test::ReferenceValue(T(j*n+i,b,a)).at(i,j,a,b);
              CHECK(T(i*n+j,a,b) == ref.epsilon(1e-10));
            }
          }
        }
      }

    }
  }

}

// The overlap gradient, dS/dA_x, through gpu::md::IntegralEngine<2>::compute1.
//
// There is no host derivative engine and no reference::compute1 to check
// against -- md::IntegralEngine<2>::compute1 throws -- so this case carries
// its own oracle: central finite differences of the *value* path, taken on
// libintx::md::reference::compute2<Overlap> so the engine appears on only one
// side of the comparison.
//
// A five-point stencil rather than the two-point one, because the tolerance is
// then set by the kernel and not by the oracle. Measured on a host run of the
// kernel at LIBINTX_MAX_L=3 over this same sweep, the oracle's own error is
// 1.4e-9 relative at h = 0.0025 -- against a 1e-7 tolerance here, and against
// the ~1e-6 an O(h^2) two-point difference would leave. That is the difference
// between a check that can see a wrong `2*alpha` and one that cannot: the same
// run says dropping the 2, dropping the `-i_x` lowering term, using the ket
// exponent, transposing the E lookup, or writing a component into the wrong
// slot each fail it by O(1).
namespace {

  /// The point-charge set, in the shape `Nuclear::Operator::Parameters` holds
  /// it. Entry `c` is nucleus `c`, and that index is the derivative buffer's
  /// slowest one -- see `ao::IntegralEngine<2>::compute1`.
  using NuclearParams = std::vector< std::tuple<int, std::array<double,3> > >;

  /// `g` with its centre displaced by `h` along axis `x`. The primitives are
  /// copied verbatim -- moving a shell does not renormalize it.
  Gaussian displaced(const Gaussian &g, int x, double h) {
    auto r = g.r;
    r[x] += h;
    std::vector<Gaussian::Primitive> prims(g.K);
    for (int k = 0; k < g.K; ++k) prims[k] = g.prims[k];
    return Gaussian(g.L, r, prims, g.pure);
  }

  /// The same for a nucleus: `p` with nucleus `c` displaced along axis `x`.
  ///
  /// This is the half of the oracle that separates `V` from `S` and `T`. A
  /// finite-difference test that displaces only the shell centres passes with
  /// the Hellmann-Feynman term entirely absent, which is the single most
  /// likely way to ship this operator's gradient wrong.
  NuclearParams displaced(const NuclearParams &p, int c, int x, double h) {
    auto q = p;
    std::get<1>(q[c])[x] += h;
    return q;
  }

  /// Five-point central difference:
  ///
  ///     df/dx ~ [ -f(2h) + 8*f(h) - 8*f(-h) + f(-2h) ] / (12h)
  ///
  /// rather than the two-point one, because the tolerance is then set by the
  /// kernel and not by the oracle -- see the Overlap.gradient case below.
  const double fd_h = 0.0025;
  const double fd_w[4] = { -1.0, +8.0, -8.0, +1.0 };
  const double fd_d[4] = { +2*fd_h, +fd_h, -fd_h, -2*fd_h };

  /// The solid-harmonic block of one shell pair, from the reference. `Params`
  /// is `None` for the operators that take none and the point charges for
  /// `Nuclear`.
  template<typename Op, typename Params>
  auto value_reference(Op, const Gaussian &a, const Gaussian &b, const Params &p) {
    auto pure = zeros(npure(a.L), npure(b.L));
    auto cart = zeros(ncart(a.L), ncart(b.L));
    libintx::md::reference::compute2<Op{}>(a, b, p, cart);
    libintx::pure::reference::transform(a.L, b.L, cart, pure);
    return pure;
  }

  /// `d/dA_x` of that block, by central differences on the bra centre.
  template<typename Op, typename Params>
  auto bra_gradient_reference(
    Op op, const Gaussian &a, const Gaussian &b, const Params &p, int x)
  {
    auto g = zeros(npure(a.L), npure(b.L));
    for (int s = 0; s < 4; ++s) {
      auto f = value_reference(op, displaced(a,x,fd_d[s]), b, p);
      for (int nb = 0; nb < npure(b.L); ++nb) {
        for (int na = 0; na < npure(a.L); ++na) {
          g(na,nb) += fd_w[s]*f(na,nb)/(12*fd_h);
        }
      }
    }
    return g;
  }

  /// `d/dR_C,x` of the nuclear-attraction block, by central differences on
  /// NUCLEUS `c`. `Nuclear` only: it is the one two-centre operator here whose
  /// operator moves.
  auto nuclear_gradient_reference(
    const Gaussian &a, const Gaussian &b, const NuclearParams &p, int c, int x)
  {
    auto g = zeros(npure(a.L), npure(b.L));
    for (int s = 0; s < 4; ++s) {
      auto f = value_reference(Nuclear, a, b, displaced(p,c,x,fd_d[s]));
      for (int nb = 0; nb < npure(b.L); ++nb) {
        for (int na = 0; na < npure(a.L); ++na) {
          g(na,nb) += fd_w[s]*f(na,nb)/(12*fd_h);
        }
      }
    }
    return g;
  }

  /// One bin through the 3-argument `compute1`, into registered host memory.
  template<typename Op>
  auto gradient(
    Op op, const Basis<Gaussian> &basis, const std::vector<Index2> &ijs,
    int A, int B, gpuStream_t stream)
  {
    namespace gpu = libintx::gpu;
    auto G = zeros(ijs.size(), npure(A), npure(B), 3);
    gpu::host::register_pointer(G.data(), G.size());
    auto md = libintx::gpu::integral_engine<2>(basis, basis, stream);
    md->compute1(op,ijs,G.data());
    gpu::stream::synchronize(stream);
    gpu::host::unregister_pointer(G.data());
    return G;
  }

  /// Both halves of `dV/dX` for one bin, through the 4-argument `compute1`.
  ///
  /// `dV` is `(ij, na, nb, x)` -- the bra derivative, the same shape every
  /// other one-electron gradient has. `dVC` is `(ij, na, nb, x, c)`: the
  /// Hellmann-Feynman term, one such block per nucleus, indexed in the order
  /// `set()` was given them.
  struct NuclearGradient {
    test::Tensor<double,4> dV;
    test::Tensor<double,5> dVC;
  };

  auto nuclear_gradient(
    const Basis<Gaussian> &basis, const std::vector<Index2> &ijs,
    int A, int B, const NuclearParams &p, gpuStream_t stream)
  {
    namespace gpu = libintx::gpu;
    NuclearGradient g = {
      zeros(ijs.size(), npure(A), npure(B), 3),
      zeros(ijs.size(), npure(A), npure(B), 3, (int)p.size())
    };
    gpu::host::register_pointer(g.dV.data(), g.dV.size());
    gpu::host::register_pointer(g.dVC.data(), g.dVC.size());
    auto md = libintx::gpu::integral_engine<2>(basis, basis, stream);
    md->set(Nuclear::Operator::Parameters{p});
    md->compute1(Nuclear,ijs,g.dV.data(),g.dVC.data());
    gpu::stream::synchronize(stream);
    gpu::host::unregister_pointer(g.dV.data());
    gpu::host::unregister_pointer(g.dVC.data());
    return g;
  }

  /// A small point-charge set for the gradient cases. Smaller than `params()`
  /// above on purpose: the oracle is `4*3*ncenters` reference evaluations per
  /// shell pair, so the nuclei count multiplies the whole sweep.
  NuclearParams gradient_params(int n = 3) {
    NuclearParams p;
    for (int i = 0; i < n; ++i) {
      p.push_back({ test::random<int>(1,20), test::random<double,3>(-1,+1) });
    }
    return p;
  }

  /// The (A|B) x {1,1}/{1,5}/{3,5} sweep against the finite differences above.
  ///
  /// Written once over the operator: only the derivative *kernels* differ
  /// between Overlap and Kinetic, while what a derivative has to satisfy does
  /// not, and a second copy would be a second thing to keep in step. `Nuclear`
  /// does not come through here -- its operator moves too, so it has its own
  /// cases below.
  template<typename Op>
  void gradient_sweep(Op op, gpuStream_t stream) {

    for (int A = 0; A <= LMAX; ++A) {
      for (int B = 0; B <= LMAX; ++B) {

        if (!test::enabled(A,B)) continue;

        SUBCASE(str("(",A,"|",B,")").c_str()) {
          for (auto K : Ks) {

            printf("(%i|%i) K={%i,%i}\n", A, B, K.first, K.second);

            auto [basis,ijs] = test::make_basis<2>({A,B}, {K.first,K.second}, 8);
            auto G = gradient(op, basis, ijs, A, B, stream);

            for (size_t ij = 0; ij < ijs.size(); ++ij) {
              auto [i,j] = ijs[ij];
              for (int x = 0; x < 3; ++x) {
                auto ref = bra_gradient_reference(op, basis[i], basis[j], None, x);
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
  /// dO/dA + dO/dB = 0 for a two-centre operator that does not itself move,
  /// and only the bra derivative is computed, so the relation is a tautology
  /// unless the ket derivative is obtained independently. It is: both
  /// operators here are symmetric, O(a,b) = O(b,a), so the bra derivative of
  /// the (B|A) bin at the swapped pair IS dO/dB of the (A|B) one, and
  ///
  ///     G_(A|B)[ij,na,nb,x] == -G_(B|A)[ji,nb,na,x]
  ///
  /// walks both index orders and both bin shapes to say so. The two shell
  /// families sit at different contraction depth, so a kernel that confused
  /// the bra and the ket primitive loops fails this as well. For Kinetic this
  /// is also the derivative analogue of `Kinetic.transpose` above -- the
  /// direct check on a host branch the device deliberately does not
  /// replicate, and what would catch a swap sneaking back in.
  template<typename Op>
  void gradient_translation(Op op, gpuStream_t stream) {

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

          auto Gab = gradient(op, basis, ab, A, B, stream);
          auto Gba = gradient(op, basis, ba, B, A, stream);

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

    // Both shells on one centre. dO/dA + dO/dB = 0 makes the *atom*
    // derivative vanish, which is the thing a caller's gradient scatter has to
    // reproduce by accumulating +V and -V into the same slot -- so this is a
    // statement about the caller as much as the kernel, and it is here because
    // getting it wrong is silent (CLAUDE.md, "A derivative scatter indexed by
    // atom must accumulate, not assign"). The kernel's own dO/dA for such a
    // pair is generally NOT zero above L = 0; the finite differences above are
    // what pin that.
    for (int L = 0; L <= LMAX; ++L) {

      if (!test::enabled(L,L)) continue;

      SUBCASE(str("(",L,"|",L,") on one centre").c_str()) {

        auto g = test::gaussian(L,3);
        Basis<Gaussian> basis;
        basis.push_back(g);
        basis.push_back(g);
        std::vector<Index2> ijs = { {0,1} };
        auto G = gradient(op, basis, ijs, L, L, stream);

        // Finite, and for L = 0 exactly zero: E^{1,0}_0 vanishes at zero
        // separation. Above L = 0 only the sum over the two centres does.
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

TEST_CASE("libintx.gpu.md2.Overlap.gradient") { gradient_sweep(Overlap, 0); }
TEST_CASE("libintx.gpu.md2.Kinetic.gradient") { gradient_sweep(Kinetic, 0); }

TEST_CASE("libintx.gpu.md2.Overlap.gradient.translation") {
  gradient_translation(Overlap, 0);
}
TEST_CASE("libintx.gpu.md2.Kinetic.gradient.translation") {
  gradient_translation(Kinetic, 0);
}

// The electron-nuclear gradient, dV/dX, through the 4-argument compute1.
//
// This operator has TWO derivative contributions and the case has to see both:
// the basis functions move (dV/dA, the same raising relation the overlap
// gradient uses) and the OPERATOR moves (dV/dR_C, the Hellmann-Feynman term,
// because 1/|r - R_C| depends on the nuclear position directly). So the
// central differences below displace the nuclei as well as the bra centre.
// Displacing only the shell centres would pass with the Hellmann-Feynman term
// entirely absent -- and that term is the dominant part of the force on any
// charged atom, so the result would be smooth, plausible and wrong.
//
// The oracle is the same five-point difference of the same
// libintx::md::reference::compute2 the value sweep uses, so the engine appears
// on only one side. The nuclei count multiplies the whole sweep (4*3*ncenters
// reference evaluations per pair per index set), which is why
// `gradient_params()` is three nuclei rather than the value sweep's nine.
TEST_CASE("libintx.gpu.md2.Nuclear.gradient") {

  gpuStream_t stream = 0;
  auto p = gradient_params();
  const int NC = (int)p.size();

  for (int A = 0; A <= LMAX; ++A) {
    for (int B = 0; B <= LMAX; ++B) {

      if (!test::enabled(A,B)) continue;

      SUBCASE(str("(",A,"|",B,")").c_str()) {
        for (auto K : Ks) {

          printf("(%i|%i) K={%i,%i}\n", A, B, K.first, K.second);

          auto [basis,ijs] = test::make_basis<2>({A,B}, {K.first,K.second}, 3);
          auto G = nuclear_gradient(basis, ijs, A, B, p, stream);

          for (size_t ij = 0; ij < ijs.size(); ++ij) {
            auto [i,j] = ijs[ij];
            for (int x = 0; x < 3; ++x) {
              // (1) the basis functions move.
              auto ref = bra_gradient_reference(Nuclear, basis[i], basis[j], p, x);
              for (int nb = 0; nb < npure(B); ++nb) {
                for (int na = 0; na < npure(A); ++na) {
                  auto v = test::ReferenceValue(ref(na,nb)).at(ij,na,nb,x);
                  CHECK(G.dV(ij,na,nb,x) == v.epsilon(1e-7));
                }
              }
              // (2) the operator moves, one nucleus at a time.
              for (int c = 0; c < NC; ++c) {
                auto refc = nuclear_gradient_reference(basis[i], basis[j], p, c, x);
                for (int nb = 0; nb < npure(B); ++nb) {
                  for (int na = 0; na < npure(A); ++na) {
                    auto v = test::ReferenceValue(refc(na,nb)).at(ij,na,nb,x,c);
                    CHECK(G.dVC(ij,na,nb,x,c) == v.epsilon(1e-7));
                  }
                }
              }
            }
          }

        }
      }
    }
  }

}

// Three-way translational invariance, elementwise -- and the check that pins
// the two contributions against each other, because it references nothing.
//
//     dV/dA + dV/dB + sum_C dV/dR_C = 0
//
// Unlike S and T this is a three-way sum: V does not depend on the centres
// only through r_a - r_b, so dV/dB is NOT -dV/dA and the two-term relation
// the overlap case checks is simply false here. dV/dB is obtained the same way
// it is there -- V(a,b) = V(b,a), so the bra derivative of the (B|A) bin at
// the swapped pair is dV/dB of this one:
//
//     dV_(A|B)[ij,na,nb,x] + dV_(B|A)[ji,nb,na,x] + sum_c dVC[ij,na,nb,x,c] == 0
//
// The sum is a cancellation, so the tolerance is relative to the size of what
// cancels rather than absolute.
TEST_CASE("libintx.gpu.md2.Nuclear.gradient.translation") {

  gpuStream_t stream = 0;
  const int n = 4;
  auto p = gradient_params();
  const int NC = (int)p.size();

  for (int A = 0; A <= LMAX; ++A) {
    for (int B = 0; B <= LMAX; ++B) {

      if (!test::enabled(A,B)) continue;

      SUBCASE(str("(",A,"|",B,") + (",B,"|",A,")^T + nuclei == 0").c_str()) {

        // Two shell families at different contraction depth, as in the overlap
        // case: the (A|B) bin is K = 3*1 and (B|A) is K = 1*3, so a kernel
        // that confused the bra and ket primitive loops fails this too.
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

        auto Gab = nuclear_gradient(basis, ab, A, B, p, stream);
        auto Gba = nuclear_gradient(basis, ba, B, A, p, stream);

        for (size_t ij = 0; ij < ab.size(); ++ij) {
          for (int x = 0; x < 3; ++x) {
            for (int nb = 0; nb < npure(B); ++nb) {
              for (int na = 0; na < npure(A); ++na) {
                double da = Gab.dV(ij,na,nb,x);
                double db = Gba.dV(ij,nb,na,x);
                double sum = da + db;
                double scale = std::max(std::fabs(da), std::fabs(db));
                for (int c = 0; c < NC; ++c) {
                  double dc = Gab.dVC(ij,na,nb,x,c);
                  sum += dc;
                  scale = std::max(scale, std::fabs(dc));
                }
                auto ref = test::ReferenceValue(0.0).at(ij,na,nb,x);
                CHECK(sum/std::max(scale,1.0) == ref.epsilon(1e-11));
              }
            }
          }
        }

      }
    }
  }

}

// What the (A|B) sweep cannot see about the derivative's parameter set. The
// value path has the same three cases in libintx.gpu.md2.Nuclear.parameters;
// the derivative needs its own, and the on-centre one is more delicate here
// than it is there.
TEST_CASE("libintx.gpu.md2.Nuclear.gradient.parameters") {

  gpuStream_t stream = 0;

  for (int L = 0; L <= LMAX; ++L) {

    if (!test::enabled(L,L)) continue;

    SUBCASE(str("(",L,"|",L,")").c_str()) {

      int N = npure(L);

      // A Z = 0 nucleus exerts no force and feels none: its whole
      // Hellmann-Feynman block is zero, and adding it changes nothing about
      // the rest of the gradient.
      {
        auto a = test::gaussian(L,3);
        auto b = test::gaussian(L,1);
        Basis<Gaussian> basis;
        basis.push_back(a);
        basis.push_back(b);
        std::vector<Index2> ijs = { {0,1} };
        auto p = gradient_params(2);
        auto G1 = nuclear_gradient(basis, ijs, L, L, p, stream);
        auto q = p;
        q.push_back({ 0, test::random<double,3>(-1,+1) });
        auto G2 = nuclear_gradient(basis, ijs, L, L, q, stream);
        for (int x = 0; x < 3; ++x) {
          for (int nb = 0; nb < N; ++nb) {
            for (int na = 0; na < N; ++na) {
              auto z = test::ReferenceValue(0.0).at(na,nb,x);
              CHECK(G2.dVC(0,na,nb,x,2) == z.epsilon(1e-12));
              auto v = test::ReferenceValue(G1.dV(0,na,nb,x)).at(na,nb,x);
              CHECK(G2.dV(0,na,nb,x) == v.epsilon(1e-10));
            }
          }
        }
      }

      // A charged nucleus sitting exactly on a shell centre: the T = 0 limit
      // of the Boys function, where a naive 1/sqrt(T) asymptotic blows up.
      // Finite, and equal to the finite difference -- which the value case
      // cannot check, because the derivative is where the odd Hermite terms
      // that vanish at T = 0 actually appear.
      {
        auto a = test::gaussian(L,3);
        auto b = test::gaussian(L,1);
        Basis<Gaussian> basis;
        basis.push_back(a);
        basis.push_back(b);
        std::vector<Index2> ijs = { {0,1} };
        NuclearParams p = {
          { 3, { a.r[0], a.r[1], a.r[2] } },
          { 2, test::random<double,3>(-1,+1) }
        };
        auto G = nuclear_gradient(basis, ijs, L, L, p, stream);
        for (int x = 0; x < 3; ++x) {
          auto ref = bra_gradient_reference(Nuclear, a, b, p, x);
          for (int nb = 0; nb < N; ++nb) {
            for (int na = 0; na < N; ++na) {
              CHECK(std::isfinite(G.dV(0,na,nb,x)));
              auto v = test::ReferenceValue(ref(na,nb)).at(na,nb,x);
              CHECK(G.dV(0,na,nb,x) == v.epsilon(1e-7));
            }
          }
          for (int c = 0; c < 2; ++c) {
            auto refc = nuclear_gradient_reference(a, b, p, c, x);
            for (int nb = 0; nb < N; ++nb) {
              for (int na = 0; na < N; ++na) {
                CHECK(std::isfinite(G.dVC(0,na,nb,x,c)));
                auto v = test::ReferenceValue(refc(na,nb)).at(na,nb,x,c);
                CHECK(G.dVC(0,na,nb,x,c) == v.epsilon(1e-7));
              }
            }
          }
        }
      }

      // Both shells AND the single nucleus on one centre. V cannot depend on
      // position at all there, so every one of the three derivative sets is
      // separately zero -- unlike the translational check above, this one does
      // not let a sign error between the two contributions hide in the sum.
      {
        auto g = test::gaussian(L,3);
        Basis<Gaussian> basis;
        basis.push_back(g);
        basis.push_back(g);
        std::vector<Index2> ijs = { {0,1} };
        NuclearParams p = { { 7, { g.r[0], g.r[1], g.r[2] } } };
        auto G = nuclear_gradient(basis, ijs, L, L, p, stream);
        for (int x = 0; x < 3; ++x) {
          for (int nb = 0; nb < N; ++nb) {
            for (int na = 0; na < N; ++na) {
              auto z = test::ReferenceValue(0.0).at(na,nb,x);
              CHECK(G.dV(0,na,nb,x) == z.epsilon(1e-10));
              CHECK(G.dVC(0,na,nb,x,0) == z.epsilon(1e-10));
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

  SUBCASE("every operator is dispatched") {
    // This subcase used to assert that `Operator::Coulomb` throws -- it was the
    // one `Operator` with no two-centre kernel and so the only thing that
    // reached `IntegralEngine<2>::compute<A,B>`'s (A|B) fallback table. It has
    // a kernel now (gpu/coulomb2/), so NOTHING reaches that table and there is
    // no operator left to assert a throw for. What replaces the assertion is
    // the statement it was standing in for: every `Operator` this engine can be
    // handed is dispatched to a kernel, and none of them silently returns the
    // buffer untouched.
    //
    // Each operator PR deleted a different CHECK_THROWS from here, which git
    // merges without complaint; a stale line asserting that an implemented
    // operator still throws is the way this subcase goes wrong. Read it by hand
    // after any merge.
    md->set(Nuclear::Operator::Parameters{params(Nuclear)});
    gpu::host::register_pointer(V.data(), V.size());
    for (auto op : {
           Operator::Overlap, Operator::Kinetic,
           Operator::Nuclear, Operator::Coulomb
         }) {
      std::fill(V.begin(), V.end(), 0.0);
      CHECK_NOTHROW(md->compute(op,ijs,V.data()));
      gpu::stream::synchronize(stream);
      // (s|s) is nonzero for all four of these operators, so an untouched
      // buffer is distinguishable from an answer.
      bool nonzero = false;
      for (double v : V) nonzero = nonzero || (v != 0.0);
      CHECK(nonzero);
    }
    gpu::host::unregister_pointer(V.data());
  }

  SUBCASE("nuclear without set()") {
    // Nuclear before any set() is an error, not a silent zero: the engine has
    // an empty device::vector of point charges and must say so rather than
    // read it.
    CHECK_THROWS(md->compute(Nuclear,ijs,V.data()));
    md->set(Nuclear::Operator::Parameters{params(Nuclear)});
    // ... and with the charges uploaded it runs. The output goes to registered
    // host memory, the same as every other compute in this file.
    gpu::host::register_pointer(V.data(), V.size());
    CHECK_NOTHROW(md->compute(Nuclear,ijs,V.data()));
    gpu::stream::synchronize(stream);
    gpu::host::unregister_pointer(V.data());
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
    CHECK_THROWS(md->compute1(Overlap,none,V.data()));
  }

  SUBCASE("which operators have a derivative kernel") {
    // Overlap and Kinetic answer the 3-argument overload; Nuclear has a kernel
    // but needs both buffers (next subcase); Coulomb has a VALUE kernel now but
    // no derivative one -- d(P|Q)/dX was deferred to whatever route the
    // derivative ERI batches settle on -- so it still belongs in this list even
    // though it is gone from the one above. A zero gradient is a
    // plausible-looking answer, and a caller assembling dE/dX out of one gets a
    // smooth, wrong force rather than an error.
    //
    // Each derivative PR edits this subcase, and git merges two patches each
    // deleting a different CHECK_THROWS without complaint -- leaving a stale
    // line asserting that an implemented operator still throws. Read it by
    // hand after any merge.
    std::vector<double> G(ijs.size()*npure(0)*npure(0)*3, 0.0);
    gpu::host::register_pointer(G.data(), G.size());
    CHECK_NOTHROW(md->compute1(Overlap,ijs,G.data()));
    CHECK_NOTHROW(md->compute1(Kinetic,ijs,G.data()));
    gpu::stream::synchronize(stream);
    gpu::host::unregister_pointer(G.data());
    CHECK_THROWS(md->compute1(Coulomb,ijs,G.data()));
    // And the host engine has no derivative path at all, on either overload.
    auto host = libintx::ao::integral_engine<2>(basis, basis);
    CHECK_THROWS(host->compute1(Overlap,ijs,G.data()));
    CHECK_THROWS(host->compute1(Nuclear,ijs,G.data(),G.data()));
  }

  SUBCASE("the nuclear derivative needs both buffers") {
    // Nuclear HAS a derivative kernel; what it does not have is a single
    // output buffer, because the Hellmann-Feynman term is indexed by nucleus.
    // The 3-argument overload therefore throws for it rather than write the
    // shell half and silently drop the dominant one -- the failure mode a
    // finite-difference test over shell centres alone cannot see.
    auto p = gradient_params(2);
    std::vector<double> dV(ijs.size()*npure(0)*npure(0)*3, 0.0);
    std::vector<double> dVC(dV.size()*p.size(), 0.0);
    md->set(Nuclear::Operator::Parameters{p});
    CHECK_THROWS(md->compute1(Nuclear,ijs,dV.data()));
    // ... and the other three have no nucleus-indexed term to put in dVC, so
    // the 4-argument overload throws for them.
    CHECK_THROWS(md->compute1(Overlap,ijs,dV.data(),dVC.data()));
    CHECK_THROWS(md->compute1(Kinetic,ijs,dV.data(),dVC.data()));
    CHECK_THROWS(md->compute1(Coulomb,ijs,dV.data(),dVC.data()));
    // With both buffers and the charges uploaded it runs.
    gpu::host::register_pointer(dV.data(), dV.size());
    gpu::host::register_pointer(dVC.data(), dVC.size());
    CHECK_NOTHROW(md->compute1(Nuclear,ijs,dV.data(),dVC.data()));
    gpu::stream::synchronize(stream);
    gpu::host::unregister_pointer(dV.data());
    gpu::host::unregister_pointer(dVC.data());
  }

  SUBCASE("nuclear derivative without set()") {
    // The same rule the value path has: no point charges is an error, not a
    // silent zero gradient.
    auto fresh = libintx::gpu::integral_engine<2>(basis, basis, stream);
    std::vector<double> dV(ijs.size()*npure(0)*npure(0)*3, 0.0);
    std::vector<double> dVC(dV.size()*2, 0.0);
    CHECK_THROWS(fresh->compute1(Nuclear,ijs,dV.data(),dVC.data()));
  }

}
