#ifndef LIBINTX_GPU_ERI_FORMAT_H
#define LIBINTX_GPU_ERI_FORMAT_H

#if !defined(__CUDACC__) && !defined(__HIPCC__)
#error "libintx/gpu/eri/format.h is device code -- include it from a .cu"
#endif

// this must come first to resolve HIP device asserts
#include "libintx/gpu/api/runtime.h"

#include "libintx/gpu/eri.h"
#include "libintx/gpu/md/engine.h"
#include "libintx/gpu/api/api.h"
#include "libintx/kengine/md/driver.h"
#include "libintx/forward.h"
#include "libintx/shell.h"

#include <algorithm>
#include <cmath>
#include <vector>

/// The half of the full-ERI build that both formats share.
///
/// The two formats are the same tensor written with different axes paired into
/// the row index, so everything except one index expression is common: the
/// shell-pair binning, the batching, the eight-fold orbit and the scatter loop
/// are written once here and each .cu instantiates them with its own Format
/// policy. Keeping them literally the same code is what makes the J/K
/// cross-check in the tests meaningful rather than a check of two independent
/// transcriptions.
namespace libintx::gpu::eri {

  /// Entry p of the eight-fold permutation orbit of a real ERI, mapping ERI
  /// slot @p n to tensor axis. Same table, same order and same meaning as
  /// kengine::md::PERMUTATIONS -- repeated as a function so it is usable from
  /// device code, with the static_assert below to keep the two from drifting.
  LIBINTX_GPU_ENABLED
  constexpr int permutation(int p, int n) {
    constexpr int P[8][4] = {
      {0,1,2,3}, {1,0,2,3}, {0,1,3,2}, {1,0,3,2},
      {2,3,0,1}, {3,2,0,1}, {2,3,1,0}, {3,2,1,0},
    };
    return P[p][n];
  }

  constexpr bool permutations_agree() {
    for (int p = 0; p < 8; ++p) {
      for (int n = 0; n < 4; ++n) {
        if (permutation(p,n) != kengine::md::PERMUTATIONS[p][n]) return false;
      }
    }
    return true;
  }

  static_assert(
    permutations_agree(),
    "libintx::gpu::eri and libintx::kengine::md must enumerate the ERI "
    "permutation orbit in the same order"
  );

  /// Where the integral (g0 g1 | g2 g3) lands in the format matrix, given the
  /// four *global* basis-function indices.
  ///
  /// J format: rows pair the bra, columns the ket, which is the ERI's own
  /// pairing -- so J[mu,nu] = sum (mu nu|lambda sigma) D[lambda,sigma] is a
  /// GEMV over contiguous rows.
  struct JFormat {
    LIBINTX_GPU_ENABLED
    static size_t index(size_t nbf, int g0, int g1, int g2, int g3) {
      return (g0*nbf + g1)*(nbf*nbf) + (g2*nbf + g3);
    }
  };

  /// Scatter one computed (ab|cd) batch into the format matrix.
  ///
  /// One thread per bra pair: `ij` is the fastest-varying index of the engine's
  /// output, so consecutive threads read consecutive doubles. Each thread walks
  /// the whole (na,nb,nc,nd) block of its quartet, reads each value once, and
  /// writes it to every distinct member of its permutation orbit.
  ///
  /// Every element of G is written exactly once over the whole build, so these
  /// are plain stores and no atomics are needed. That rests on two things: the
  /// driver visits each unordered pair-of-pairs once (the class-pair loop, and
  /// the triangle test below when both sides are one class), and the orbit is
  /// deduplicated on the permuted shell tuple so a quartet never writes the
  /// same destination twice.
  template<typename Format>
  __global__
  void scatter_kernel(
    const double *V,
    int M, int N,
    int NA, int NB, int NC, int ND,
    const int *bra, const int *ket,
    const int *shell_offset,
    size_t bra_offset, size_t ket_offset,
    bool same_class,
    size_t nbf,
    double *G)
  {

    const int ij = blockIdx.x*blockDim.x + threadIdx.x;
    if (ij >= M) return;

    const size_t stride[4] = {
      (size_t)M,
      (size_t)M*NA,
      (size_t)M*NA*NB,
      (size_t)M*NA*NB*NC
    };
    const size_t kl_stride = (size_t)M*NA*NB*NC*ND;

    for (int kl = blockIdx.y; kl < N; kl += gridDim.y) {

      // One class on both sides is computed as a full rectangle, so each
      // quartet appears twice in the batch; keep the lower triangle in the
      // engine's global pair order and let the orbit put back the rest. Same
      // rule as kengine::md::digest, and it has to stay the same one: relax it
      // and every element of G is written twice.
      if (same_class && (bra_offset + ij) < (ket_offset + kl)) continue;

      const int s[4] = { bra[2*ij], bra[2*ij+1], ket[2*kl], ket[2*kl+1] };

      // Enumerate the orbit and drop the members that repeat a shell tuple:
      // a==b, c==d and (ab)==(cd) each collapse a different subset of it. Every
      // survivor carries weight one -- there are no degeneracy factors here,
      // which is precisely why the orbit is enumerated rather than counted.
      int orbit[8];
      int norbit = 0;
      for (int t = 0; t < 8; ++t) {
        bool seen = false;
        for (int u = 0; u < norbit && !seen; ++u) {
          seen = true;
          for (int n = 0; n < 4; ++n) {
            if (s[permutation(t,n)] != s[permutation(orbit[u],n)]) {
              seen = false;
              break;
            }
          }
        }
        if (!seen) orbit[norbit++] = t;
      }

      const size_t base = (size_t)ij + (size_t)kl*kl_stride;

      for (int x3 = 0; x3 < ND; ++x3) {
        for (int x2 = 0; x2 < NC; ++x2) {
          for (int x1 = 0; x1 < NB; ++x1) {
            for (int x0 = 0; x0 < NA; ++x0) {

              const int x[4] = { x0, x1, x2, x3 };
              const double v = V[
                base +
                x0*stride[0] + x1*stride[1] + x2*stride[2] + x3*stride[3]
              ];

              for (int t = 0; t < norbit; ++t) {
                // Orbit member t reads ERI slot n off tensor axis
                // permutation(orbit[t],n), so that slot's global basis
                // function index is the axis' shell offset plus the axis'
                // index. This is the same mapping kengine::md::digest uses,
                // written from the axis side rather than the slot side.
                int g[4];
                for (int n = 0; n < 4; ++n) {
                  const int axis = permutation(orbit[t], n);
                  g[n] = shell_offset[s[axis]] + x[axis];
                }
                G[Format::index(nbf, g[0], g[1], g[2], g[3])] = v;
              }

            }
          }
        }
      }

    }

  }

  /// Doubles in one integral batch. Device memory, so this can be far larger
  /// than the K engine's pinned host buffer -- but it shares the card with G,
  /// which is already nbf^4.
  inline constexpr size_t default_max_batch = 8*1024*1024;

  /// The host half: bin the basis into pair classes, walk the class pairs, and
  /// scatter each computed batch into @p G.
  ///
  /// The structure is kengine::md::build minus the density and the screening.
  /// Neither belongs here: the point of this format is a complete tensor, and
  /// a screened-out quartet would leave a zero indistinguishable from a real
  /// one. The pair binning itself is shared with the K engine, since the
  /// three-way (L, solid-harmonic, contraction-degree) constraint on a batch
  /// is the engine's, not the caller's.
  template<typename Format>
  void build(
    const Basis<Gaussian> &basis,
    double *G,
    gpuStream_t stream,
    size_t max_batch = default_max_batch)
  {

    const size_t nbf = basis.nbf();
    const size_t nshells = basis.size();
    if (!nshells || !nbf) return;

    // Zero first. Every element is written exactly once below, so a zero that
    // survives is a hole -- which the tests can see, where uninitialised
    // memory would only sometimes show up.
    device_memory_t::memset(
      G, 0, format_size(basis)*sizeof(double), stream
    );

    // Nothing screens, so every pair is in play and every bound is 1.
    auto classes = kengine::md::make_pair_classes(
      basis, [](int, int) { return 1.0f; }
    );

    std::vector<int> offsets(nshells);
    for (size_t i = 0; i < nshells; ++i) {
      offsets[i] = basis.range(i).begin();
    }
    device::vector<int> shell_offset;
    shell_offset.assign(offsets.data(), offsets.size());

    md::IntegralEngine<4> engine(basis, basis, stream);

    device::vector<double> V;
    device::vector<int> bra_shells, ket_shells;
    std::vector<Index2> bra_index, ket_index;
    std::vector<int> bra_flat, ket_flat;

    constexpr int block_size = 128;

    for (size_t cb = 0; cb < classes.size(); ++cb) {
      for (size_t ck = 0; ck <= cb; ++ck) {

        const auto &B = classes[cb];
        const auto &Q = classes[ck];

        const size_t block = (
          (size_t)B.nbf_first*B.nbf_second*Q.nbf_first*Q.nbf_second
        );
        // Bound a batch by the size of the integral buffer rather than by the
        // pair count: a (dd|dd) quartet is 1296 doubles where an (ss|ss) one
        // is a single double.
        const size_t quartets = std::max<size_t>(1, max_batch/block);
        const size_t nbra = std::max<size_t>(
          1, (size_t)std::sqrt((double)quartets)
        );
        const size_t nket = std::max<size_t>(1, quartets/nbra);
        const bool same_class = (cb == ck);

        for (size_t i0 = 0; i0 < B.pairs.size(); i0 += nbra) {
          const size_t i1 = std::min(i0 + nbra, B.pairs.size());

          bra_index.clear();
          bra_flat.clear();
          for (size_t i = i0; i < i1; ++i) {
            const auto &p = B.pairs[i];
            bra_index.push_back(Index2{p.first, p.second});
            bra_flat.push_back(p.first);
            bra_flat.push_back(p.second);
          }

          for (size_t j0 = 0; j0 < Q.pairs.size(); j0 += nket) {
            const size_t j1 = std::min(j0 + nket, Q.pairs.size());
            // With one class on both sides only the lower triangle is
            // scattered, so a chunk entirely above the diagonal is all waste.
            if (same_class && i1 <= j0) continue;

            ket_index.clear();
            ket_flat.clear();
            for (size_t j = j0; j < j1; ++j) {
              const auto &p = Q.pairs[j];
              ket_index.push_back(Index2{p.first, p.second});
              ket_flat.push_back(p.first);
              ket_flat.push_back(p.second);
            }

            const size_t M = bra_index.size(), N = ket_index.size();
            V.resize(M*N*block);
            const std::array<size_t,2> dims = {
              M*(size_t)B.nbf_first*B.nbf_second,
              (size_t)Q.nbf_first*Q.nbf_second*N
            };
            engine.compute(
              libintx::Coulomb, bra_index, ket_index,
              BraKet<const double*>{nullptr,nullptr},
              V.data(), dims
            );

            // Stream-ordered so the arrays are in place before the scatter
            // reads them. An async copy out of pageable memory has staged the
            // source by the time it returns, so reusing the vectors on the
            // next iteration is safe.
            bra_shells.resize(bra_flat.size());
            ket_shells.resize(ket_flat.size());
            gpu::memcpy(
              bra_shells.data(), bra_flat.data(),
              sizeof(int)*bra_flat.size(), stream
            );
            gpu::memcpy(
              ket_shells.data(), ket_flat.data(),
              sizeof(int)*ket_flat.size(), stream
            );

            dim3 grid(
              (unsigned)((M + block_size - 1)/block_size),
              (unsigned)std::min<size_t>(N, 65535)
            );
            scatter_kernel<Format><<<grid, block_size, 0, stream>>>(
              V.data(),
              (int)M, (int)N,
              B.nbf_first, B.nbf_second, Q.nbf_first, Q.nbf_second,
              bra_shells.data(), ket_shells.data(), shell_offset.data(),
              i0, j0, same_class,
              nbf, G
            );

          }
        }
      }
    }

    stream::synchronize(stream);

  }

}

#endif /* LIBINTX_GPU_ERI_FORMAT_H */
