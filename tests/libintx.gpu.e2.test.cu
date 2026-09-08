// -*-c++-*-
//
// The device McMurchie-Davidson E coefficients against the host ones.
//
// E2 is the piece all three one-electron device kernels are built on -- and
// the Coulomb path's gpu::md::make_basis too -- so it is worth pinning down on
// its own, before any operator kernel exists to fail because of it.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "test.h"

#include "libintx/ao/md/hermite.h"
#include "libintx/config.h"
#include "libintx/gpu/api/api.h"
#include "libintx/gpu/api/thread_group.h"
#include "libintx/gpu/md/e2.h"

using namespace libintx;

namespace {

  // Dense [x][i][j][k] dump, so the host indexes it without repeating the
  // device class's strides.
  template<int A, int Bmax, int Pmax>
  __host__ __device__
  constexpr int e2_index(int i, int j, int k, int x) {
    return k + (Pmax+1)*(j + (Bmax+1)*(i + (A+1)*x));
  }

  template<typename Block, int A, int B, int DB>
  __global__ __launch_bounds__(Block::size())
  void e2_kernel(double a, double b, array<double,3> r, double *out) {
    using E2 = libintx::gpu::md::E2<A,B,DB>;
    auto block = Block();
    __shared__ E2 E;
    E.init(a, b, r, block);
    block.sync();
    if (block.thread_rank() != 0) return;
    for (int x = 0; x < 3; ++x) {
      for (int i = 0; i <= A; ++i) {
        for (int j = 0; j <= E2::Bmax; ++j) {
          for (int k = 0; k <= E2::Pmax; ++k) {
            out[e2_index<A,E2::Bmax,E2::Pmax>(i,j,k,x)] = E.value(i,j,k,x);
          }
        }
      }
    }
  }

  template<int A, int B, int DB>
  void e2_subcase() {

    namespace gpu = libintx::gpu;

    constexpr int Bmax = B + DB;
    constexpr int Pmax = A + Bmax;
    constexpr size_t N = 3*(A+1)*(Bmax+1)*(Pmax+1);

    printf("E2<%i,%i,%i> (Bmax=%i, Pmax=%i, %zu doubles)\n", A, B, DB, Bmax, Pmax, N);

    double a = test::random<double>(0.1, 4.0);
    double b = test::random<double>(0.1, 4.0);
    auto ra = test::random<double,3>(-1.0, +1.0);
    auto rb = test::random<double,3>(-1.0, +1.0);
    array<double,3> r = ra - rb;

    // A block wide enough for the recursion: E2::init parallelises over the
    // Hermite index and needs Block::size() >= A+Bmax+1.
    using Block = gpu::thread_block<32>;
    static_assert(Block::size() >= A+Bmax+1);

    gpu::device::vector<double> out(N);
    gpu::memset(out, 0);
    e2_kernel<Block,A,B,DB><<<1,Block()>>>(a, b, r, out.data());
    gpu::stream::synchronize();

    std::vector<double> v(N);
    gpu::memcpy(v.data(), out.data(), sizeof(double)*N);

    libintx::md::E2<double,A,Bmax,Pmax> ref(a, b, r);

    for (int x = 0; x < 3; ++x) {
      for (int i = 0; i <= A; ++i) {
        for (int j = 0; j <= Bmax; ++j) {
          for (int k = 0; k <= Pmax; ++k) {
            auto e = test::ReferenceValue(ref(i,j,k,x)).at(i,j,k,x);
            CHECK(v[e2_index<A,Bmax,Pmax>(i,j,k,x)] == e.epsilon(1e-10));
          }
        }
      }
    }

  }

}

TEST_CASE("libintx.gpu.e2") {
  foreach2(
    std::make_index_sequence<LMAX+1>(),
    std::make_index_sequence<LMAX+1>(),
    [](auto A, auto B) {
      // DB=0 is what the Coulomb path uses; DB=2 is the extra ket degree the
      // kinetic operator needs (host: libintx::md::E2<T,A,B+2,0>).
      e2_subcase<A.value,B.value,0>();
      e2_subcase<A.value,B.value,2>();
    }
  );
}
