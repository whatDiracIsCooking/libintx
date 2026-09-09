// -*-c++-*-

// this must come first to resolve HIP device asserts
#include "libintx/gpu/api/runtime.h"

#include "libintx/gpu/md/basis.h"
#include "libintx/gpu/md/e2.h"
#include "libintx/gpu/api/api.h"
#include "libintx/gpu/api/thread_group.h"
#include "libintx/ao/md/hermite.h"
#include "libintx/pure.transform.h"
#include "libintx/config.h"
#include "libintx/utility.h"
#include "libintx/math.h"

#include <functional>

namespace libintx::gpu::md {

  namespace cart = cartesian;
  namespace herm = hermite;

  // 2*LMAX+1, not 2*LMAX: a derivative batch's Hermite index reaches
  // A+B+1 (see make_basis1). One extra degree in a compile-time table.
  __device__
  constexpr auto orbitals = hermite::orbitals2<2*LMAX+1>;

  // Gaussian2 and E2 used to live here, file-private. Both are shared with the
  // one-electron device engine now: Gaussian2 in gpu/md/basis.h next to the
  // other device basis PODs, E2 in gpu/md/e2.h.

  /// One coefficient of the pair's Hermite expansion, for Hermite index `op`
  /// and the Cartesian pair `(oa,ob)`.
  ///
  ///  - `Deriv < 0`: the value, `E^{ab}_p = prod_y E^{a_y b_y}_{p_y}`.
  ///  - `Deriv == 0`: `d/dA_x` of it, `2a E^{(a+1_x)b}_p - i_x E^{(a-1_x)b}_p`
  ///    along axis `x` and the plain value along the other two.
  ///  - `Deriv == 1`: the same with the roles of the two shells swapped.
  ///
  /// `E2::value` is zero outside `k <= i+j`, so the raised-index term is the
  /// only one that reaches the top Hermite degree and no bounds test is
  /// needed; the lowering term is guarded only to keep `i-1 = -1` out of the
  /// index arithmetic.
  template<int Deriv, typename E, typename Oa, typename Ob, typename Op>
  __device__ LIBINTX_ALWAYS_INLINE
  double coefficient(E &e, double a, double b, int x, Oa oa, Ob ob, Op op) {
    if constexpr (Deriv < 0) {
      double v = 1;
      for (int i = 0; i < 3; ++i) v *= e.value(oa[i], ob[i], op[i], i);
      return v;
    }
    else {
      double v;
      if constexpr (Deriv == 0) {
        v = 2*a*e.value(oa[x]+1, ob[x], op[x], x);
        if (oa[x]) v -= oa[x]*e.value(oa[x]-1, ob[x], op[x], x);
      }
      else {
        v = 2*b*e.value(oa[x], ob[x]+1, op[x], x);
        if (ob[x]) v -= ob[x]*e.value(oa[x], ob[x]-1, op[x], x);
      }
      for (int i = 0; i < 3; ++i) {
        if (i == x) continue;
        v *= e.value(oa[i], ob[i], op[i], i);
      }
      return v;
    }
  }

  /// @tparam Deriv -1 for a value batch, 0/1 for the derivative with respect
  ///         to the first/second centre of the pair (see `coefficient`).
  ///         A derivative batch spans one more Hermite degree.
  template<typename ThreadBlock, int A, int B, bool Pure, int Deriv = -1>
  __global__ __launch_bounds__(ThreadBlock::size())
  void make_basis(const Gaussian2 *gbasis, double *H, size_t stride, size_t k_stride, int x) {

    namespace cart = cartesian;

    auto thread_block = ThreadBlock();

    constexpr int DimX = ThreadBlock::x;
    constexpr int DimY = ThreadBlock::y;
    // extra bra/ket degree the derivative relation raises E by
    constexpr int DA = (Deriv == 0);
    constexpr int DB = (Deriv == 1);
    constexpr int NP = nherm2(A+B+(Deriv >= 0));

    __shared__
    union shmem {
      Gaussian2 ab;
      __device__ shmem() {}
    } shmem;

    //__shared__ Gaussian2 ab;
    auto &ab = shmem.ab;

    memcpy1(&gbasis[blockIdx.x], &ab, thread_block);
    thread_block.sync();

    __shared__ array<double,3> AB;
    if (thread_block.thread_rank() == 0) {
      AB = ab.r.first - ab.r.second;
    }

    for (int ki = 0, k = 0; ki < ab.first.K; ++ki) {
      for (int kj = 0; kj < ab.second.K; ++kj, ++k) {

        __shared__ E2<A+DA,B,DB> E;
        __shared__ double* Hk;
        __shared__ double a, b;

        {
          __shared__ Hermite h;
          if (thread_block.thread_rank() == 0) {
            Hk = H + blockIdx.x*stride + k*k_stride;
            auto& [ai,Ci] = ab.first.prims[ki];
            auto& [aj,Cj] = ab.second.prims[kj];
            // P = (AB| overlap
            double Kab = std::exp(-(ai*aj)/(ai+aj)*norm(AB));
            //double sij = (ij.first == ij.second ? 1 : 2);
            // shmem values
            a = ai;
            b = aj;
            h.exp = (a+b);
            h.C = Ci*Cj*Kab;
            h.r = center_of_charge(a, ab.r.first, b, ab.r.second);
            h.inv_2_exp = 1/math::pow<A+B>(2*(a+b));
            //printf("%i: %f \n", blockIdx.x, h.exp);
          }
          thread_block.sync();
          E.init(a, b, AB, thread_block);
          memcpy1(&h, Hermite::hdata(Hk), thread_block);
        }
        thread_block.sync();

        if constexpr (Pure) {

          constexpr int NA = npure(A);
          constexpr int NB = npure(B);

          static_assert(ncart(A) <= DimX);
          static_assert(npure(B) <= DimX);

          __shared__ double h[NB*ncart(A)*DimY];

          for (int batch = 0; batch < (NP+DimY-1)/DimY; ++batch) {

            int ip = batch*DimY + threadIdx.y;
            int np = min(DimY,NP-batch*DimY);

#define h(i,j,p) h[(j) + (i)*NB + (p)*NB*ncart(A)]

            // [a'b'p] -> [a'bp]
            if (threadIdx.x < ncart(A) && ip < NP) {
              int i = threadIdx.x;
              double v[ncart(B)] = {};
              for (int j = 0; j < ncart(B); ++j) {
                auto oa = orbitals[cart::index(A)+i];
                auto ob = orbitals[cart::index(B)+j];
                auto op = orbitals[ip];
                v[j] = coefficient<Deriv>(E, a, b, x, oa, ob, op);
              }
              pure::cartesian_to_pure<B>(
                [&](auto j) { return v[index(j)]; },
                [&](auto j, auto v) { h(i,index(j),threadIdx.y) = v; }
              );
            }
            thread_block.sync();

            // [a'bp] -> [abp]
            double v[ncart(A)] = {};
            if (threadIdx.x < NB && ip < NP) {
              int j = threadIdx.x;
              for (int i = 0; i < ncart(A); ++i) {
                v[i] = h(i,j,threadIdx.y);
              }
            }
            thread_block.sync();
#undef h

#define h(i,j,p) h[(i) + (j)*NA + (p)*NB*NA]
            if (threadIdx.x < NB && ip < NP) {
              int j = threadIdx.x;
              pure::cartesian_to_pure<A>(
                [&](auto i) { return v[index(i)]; },
                [&](auto i, auto v) {
                  //printf("%i,%i,%i %f\n", (int)index(i), (int)j, (int)threadIdx.y, v);
                  h(index(i),j,threadIdx.y) = v;
                }
              );
            }
            thread_block.sync();
#undef h

            memcpy(NA*NB*np, h, Hermite::gdata(Hk)+batch*DimY*NA*NB, thread_block);
            thread_block.sync();

          }

        }

        if constexpr (!Pure) {
          for (int ip = threadIdx.z; ip < NP; ip += blockDim.z) {
            for (int i = threadIdx.y; i < ncart(A); i += blockDim.y) {
              int j = threadIdx.x;
              int idx = j;
              idx += i*ncart(B);
              idx += ip*ncart(B)*ncart(A);
              auto oa = orbitals[cart::index(A)+i];
              auto ob = orbitals[cart::index(B)+j];
              auto op = orbitals[ip];
              Hermite::gdata(Hk)[idx] = coefficient<Deriv>(E, a, b, x, oa, ob, op);
            }
          }
        }

      }
    }

  }

  /// @tparam Deriv -1 for a value batch, 0/1 for a derivative batch with
  ///         respect to the pair's first/second centre.
  /// @param x Cartesian component; ignored when `Deriv < 0`.
  template<int A, int B, int Deriv = -1>
  Basis2 make_basis(
    const std::vector<Gaussian2> &ab,
    int x,
    device::vector<double> &H, // Hermite data buffer
    gpuStream_t stream)
  {

    // A derivative batch spans one more Hermite degree than its shell pair;
    // everything else about it -- the header, the basis-function extent, the
    // solid-harmonic transform -- is the value batch's.
    constexpr int dL = (Deriv >= 0);
    constexpr uint NP = nherm2(A+B+dL);
    constexpr int align = Basis2::alignment;

    // auto idx = pairs.at(0);
    auto a = ab[0].first;
    auto b = ab[0].second;
    int K = a.K*b.K;
    int N = ab.size();
    int n_aligned = (N+(align-N%align));
    size_t extent = Hermite::extent(a,b,dL);
    size_t k_stride = extent*n_aligned;

    bool pure = (a.pure && b.pure);
    H.resize(
      k_stride*K +
      // pure_transform data; a derivative batch has none, see basis.h
      (dL ? 0 : npure(A)*npure(B)*ncart(A+B))
    );
    double *pure_transform_ptr = (dL ? nullptr : H.data() + K*k_stride);

    dim3 grid = { (unsigned int)N };

    if (pure) {
      constexpr bool Pure = true;
      constexpr uint NX = std::max(ncart(A),npure(B));
      using Block = thread_block<NX, std::min(NP,128/NX)>;
      //ssert(false);
      //printf("BLOCK<%i,%i,%i>\n", Block::x, Block::y, Block::z);
      make_basis<Block,A,B,Pure,Deriv><<<grid,Block(),0,stream>>>(ab.data(), H.data(), extent, k_stride, x);
      if constexpr (!dL) {
        constexpr libintx::md::pure_transform<A,B> pure_transform;
        gpu::memcpy(
          pure_transform_ptr,
          pure_transform.data,
          sizeof(pure_transform.data)
        );
      }
    }
    else {
      constexpr bool Pure = false;
      constexpr uint NA = ncart(A);
      constexpr uint NB = ncart(B);
      constexpr uint MaxThreads = std::min<uint>(128,NA*NB*NP);
      constexpr uint NX = NB;
      constexpr uint NY = std::min<uint>(MaxThreads/NX,NA);
      constexpr uint NZ = (NY != NA) ? 1 : std::min<uint>(MaxThreads/(NX*NY),64);
      static_assert(NZ);
      static_assert(NY == ncart(A) || NZ == 1);
      //printf("BLOCK<%i,%i,%i>\n", NX, NY, NZ);
      using Block = thread_block<NX,NY,NZ>;
      make_basis<Block,A,B,Pure,Deriv><<<grid,Block()>>>(ab.data(), H.data(), extent, k_stride, x);
      pure_transform_ptr = nullptr;
    }

    return Basis2 {
      .first = a,
      .second = b,
      .N = N,
      .K = K,
      .data = H.data(),
      .k_stride = k_stride,
      .pure_transform = pure_transform_ptr,
      .dL = dL
    };

  }

  Basis2 make_basis(
    const Basis<Gaussian> &A,
    const Basis<Gaussian> &B,
    const std::vector<Index2> &pairs,
    device::vector<double> &H,
    gpuStream_t stream)
  {

    std::vector<Gaussian2> ab;
    ab.reserve(pairs.size());
    for (auto [i,j] : pairs) {
      Gaussian2 g = {
        A[i], B[j],
        { center(A[i]), center(B[j]) }
      };
      ab.push_back(g);
    }

    gpu::host::register_pointer(ab.data(), ab.size());

    auto a = ab[0].first;
    auto b = ab[0].second;

    using F = std::function<
      Basis2(
        const std::vector<Gaussian2> &ab,
        int x,
        device::vector<double> &H,
        gpuStream_t stream
      )>;

    // Still (LMAX+1)^2, indexed by the pair's own angular momenta: Route A
    // never builds an L+1 shell, so nothing here grows with the derivative.
    static auto make_basis = make_array<F,LMAX+1,LMAX+1>(
      [](auto ... args) -> F {
        return &md::make_basis<args...>;
      }
    );

    auto basis = make_basis[a.L][b.L](ab, 0, H, stream);

    gpu::stream::synchronize(stream);
    gpu::host::unregister_pointer(ab.data());

    return basis;

  }

  Basis2 make_basis1(
    const Basis<Gaussian> &A,
    const Basis<Gaussian> &B,
    const std::vector<Index2> &pairs,
    int centre,
    int x,
    device::vector<double> &H,
    gpuStream_t stream)
  {

    libintx_assert(centre == 0 || centre == 1);
    libintx_assert(x >= 0 && x < 3);

    std::vector<Gaussian2> ab;
    ab.reserve(pairs.size());
    for (auto [i,j] : pairs) {
      Gaussian2 g = {
        A[i], B[j],
        { center(A[i]), center(B[j]) }
      };
      ab.push_back(g);
    }

    gpu::host::register_pointer(ab.data(), ab.size());

    auto a = ab[0].first;
    auto b = ab[0].second;

    using F = std::function<
      Basis2(
        const std::vector<Gaussian2> &ab,
        int x,
        device::vector<double> &H,
        gpuStream_t stream
      )>;

    static auto first = make_array<F,LMAX+1,LMAX+1>(
      [](auto ... args) -> F { return &md::make_basis<args...,0>; }
    );
    static auto second = make_array<F,LMAX+1,LMAX+1>(
      [](auto ... args) -> F { return &md::make_basis<args...,1>; }
    );

    auto &table = (centre == 0 ? first : second);
    auto basis = table[a.L][b.L](ab, x, H, stream);

    gpu::stream::synchronize(stream);
    gpu::host::unregister_pointer(ab.data());

    return basis;

  }

  Basis1 make_basis(
    const Basis<Gaussian> &A,
    const std::vector<Index1> &idx,
    device::vector<Hermite> &H,
    gpuStream_t stream)
  {

    libintx_assert(!A.empty());
    libintx_assert(!idx.empty());

    int L = A[idx.front()].L;
    int K = A[idx.front()].K;
    int N = idx.size();

    for (auto i : idx) {
      libintx_assert(A[i].K == K);
      libintx_assert(A[i].L == L);
    }

    std::vector<Hermite> a;
    a.reserve(K*idx.size());
    for (int k = 0; k < K; ++k) {
      for (auto i : idx) {
        auto &r = center(A[i]);
        auto &g = A[i].prims;
        auto e = g[k].a;
        auto C = g[k].C;
        a.push_back( { e, C, r, 1.0/(2*e) } );
      }
    }
    H.assign(a.data(), a.size());

    return Basis1{L,K,N,H.data()};

  }

}
