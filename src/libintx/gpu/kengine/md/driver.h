#ifndef LIBINTX_GPU_KENGINE_MD_DRIVER_H
#define LIBINTX_GPU_KENGINE_MD_DRIVER_H

#include "libintx/kengine.h"
#include "libintx/shell.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <memory>
#include <vector>

/// The engine-agnostic half of the integral-direct K build. Both the host
/// (libintx::md::IntegralEngine<4>) and the device
/// (libintx::gpu::md::IntegralEngine<4>) four-centre MD engines expose the
/// same compute(Operator, bra, ket, norms, V, dims) contract, so the shell
/// pair bookkeeping, the screening and the digest live here once and each back
/// end supplies only its engine type and its integral buffer.
///
/// The file sits under gpu/ next to the device K engine, mirroring where the
/// J engine's implementation lives, but nothing in it is device code: it is
/// ordinary host C++ that the host engine (src/libintx/ao/md/kengine.cc)
/// includes unchanged. The density-fitted counterpart is df.h in this
/// directory, and the two share the pair binning and the tile plumbing below.
namespace libintx::kengine::md {

  /// A canonical shell pair. `first`/`second` are shell indices into the
  /// basis, ordered so that L(first) >= L(second): the MD kernel table is
  /// keyed on the pair's total angular momentum and every batch handed to the
  /// engine must be one (La,Lb) class, so that ordering is what makes a class
  /// a class. `norm` is the Schwarz bound sqrt(max |(pq|pq)|), or 1 when
  /// nothing is screening.
  struct ShellPair {
    int first, second;
    float norm = 1;
  };

  /// One angular-momentum class: the pairs that can go to the engine in a
  /// single batch, plus their block dimensions.
  struct PairClass {
    int nbf_first = 0, nbf_second = 0;
    float max_norm = 0;
    std::vector<ShellPair> pairs;
  };

  /// The eight index permutations that leave a real ERI (ab|cd) invariant:
  /// a<->b, c<->d and bra<->ket. Entry p maps loop slot n to tensor axis
  /// PERMUTATIONS[p][n] -- see digest() for why the permutation, and not just
  /// the permuted shell tuple, has to be carried around.
  inline constexpr int PERMUTATIONS[8][4] = {
    {0,1,2,3}, {1,0,2,3}, {0,1,3,2}, {1,0,3,2},
    {2,3,0,1}, {3,2,0,1}, {2,3,1,0}, {3,2,1,0},
  };

  /// Schwarz bounds sqrt(max |(ij|ij)|), indexed by shell pair and symmetric,
  /// with the threshold that decides what is too small to compute. This is the
  /// screening object both back ends hand to the engine: once built it is
  /// plain host data, so a set of bounds computed on the device is equally
  /// usable by the host engine and the other way round.
  struct SchwarzScreening : libintx::KEngine::Screening {

    SchwarzScreening(size_t n, float threshold)
      : n_(n), threshold_(threshold), g_(n*n, 0.0f) {}

    float max2(int i, int j) const override { return g_[(size_t)i*n_ + j]; }
    float max() const override { return max_; }

    bool skip(float v) const override {
      return (threshold_ > 0 && v < threshold_);
    }

    void set(int i, int j, float v) {
      g_[(size_t)i*n_ + j] = v;
      g_[(size_t)j*n_ + i] = v;
      max_ = std::max(max_, v);
    }

  private:
    size_t n_;
    float threshold_;
    float max_ = 0;
    std::vector<float> g_;
  };

  /// Evaluate the Schwarz bounds with the given four-centre engine.
  ///
  /// One diagonal quartet per call. A batch would have to be square in the
  /// shell pairs to keep its diagonal -- |pairs| times the integral work for
  /// the same |pairs| numbers -- so the per-call overhead is the cheaper of
  /// the two, and this runs once per geometry rather than once per SCF
  /// iteration.
  template<typename Engine, typename Buffer>
  std::shared_ptr<const libintx::KEngine::Screening> schwarz_screening(
    const Basis<Gaussian> &basis,
    Engine &engine,
    Buffer &buffer,
    float threshold)
  {
    auto s = std::make_shared<SchwarzScreening>(basis.size(), threshold);
    for (size_t p = 0; p < basis.size(); ++p) {
      for (size_t q = 0; q <= p; ++q) {
        int i = (int)p, j = (int)q;
        if (basis[i].L < basis[j].L) std::swap(i,j);
        const size_t na = (size_t)libintx::nbf(basis[i]);
        const size_t nb = (size_t)libintx::nbf(basis[j]);
        std::vector<Index2> ij = { Index2{i,j} };
        double *v = buffer.resize(na*nb*na*nb);
        engine.compute(
          libintx::Coulomb, ij, ij, BraKet<const double*>{nullptr,nullptr},
          v, std::array<size_t,2>{na*nb, na*nb}
        );
        buffer.synchronize();
        // v is a (1*na*nb) x (na*nb*1) column-major block, so the Schwarz
        // diagonal (mu nu | mu nu) is v[m + na*nb*m].
        double g = 0;
        for (size_t m = 0; m < na*nb; ++m) {
          g = std::max(g, std::fabs(v[m + na*nb*m]));
        }
        s->set(i, j, (float)std::sqrt(g));
      }
    }
    return s;
  }

  /// Split a basis into canonical (p >= q) shell pairs grouped by angular
  /// momentum class. `norm2(i,j)` supplies each pair's Schwarz bound.
  template<typename Norm2>
  std::vector<PairClass> make_pair_classes(
    const Basis<Gaussian> &basis,
    Norm2 &&norm2)
  {
    // Key on (L, pure, K) of both slots. L and pure fix the block dimensions,
    // and the contraction degree has to match too: md/basis.cc packs a batch
    // as one flat array of K_ab primitive pairs and asserts a.K*b.K == K, so a
    // batch mixing a 3-primitive and a 1-primitive shell aborts rather than
    // padding. This is the same three-way binning the pair batching in a
    // direct SCF does -- angular momentum, magnitude, contraction degree.
    std::map< std::array<int,5>, PairClass > classes;
    for (size_t p = 0; p < basis.size(); ++p) {
      for (size_t q = 0; q <= p; ++q) {
        int i = (int)p, j = (int)q;
        // Higher angular momentum first; the MD kernels are written for
        // A >= B, and the digest carries the real shell indices anyway.
        if (basis[i].L < basis[j].L) std::swap(i,j);
        const auto &a = basis[i];
        const auto &b = basis[j];
        std::array<int,5> key = { a.L, a.pure, b.L, b.pure, a.K*b.K };
        auto &c = classes[key];
        c.nbf_first = libintx::nbf(a);
        c.nbf_second = libintx::nbf(b);
        float n = (float)norm2(i,j);
        c.max_norm = std::max(c.max_norm, n);
        c.pairs.push_back({i,j,n});
      }
    }
    std::vector<PairClass> v;
    v.reserve(classes.size());
    for (auto &kv : classes) v.push_back(std::move(kv.second));
    return v;
  }

  /// The integral batch on the host: a plain buffer, nothing to synchronise.
  /// Both host K engines -- the integral-direct one and the density-fitted one
  /// in df.h -- hand this to build() where the device engines hand a pinned,
  /// stream-synchronising one.
  struct HostBuffer {
    std::vector<double> data;
    double* resize(size_t n) {
      data.assign(n, 0.0);
      return data.data();
    }
    void synchronize() {}
  };

  /// Dense row-major nbf x nbf scratch: the engine's private copy of D and its
  /// accumulator for K. Both are full matrices -- the tile callbacks are the
  /// boundary with the caller's storage, not the engine's internal layout.
  struct Matrix {
    size_t n = 0;
    std::vector<double> data;
    explicit Matrix(size_t n = 0) : n(n), data(n*n, 0.0) {}
    double& operator()(size_t i, size_t j) { return data[i*n + j]; }
    double operator()(size_t i, size_t j) const { return data[i*n + j]; }
  };

  /// Read D shell block by shell block through the caller's TileIn. A tile the
  /// caller declines is left at zero.
  inline Matrix gather_density(
    const Basis<Gaussian> &basis,
    const KEngine::TileIn &D)
  {
    Matrix d(basis.nbf());
    std::vector<double> block;
    for (size_t i = 0; i < basis.size(); ++i) {
      auto ri = basis.range(i);
      for (size_t j = 0; j < basis.size(); ++j) {
        auto rj = basis.range(j);
        size_t ni = (size_t)ri.size(), nj = (size_t)rj.size();
        block.assign(ni*nj, 0.0);
        KEngine::TileIndex ti{(size_t)ri.begin(), (size_t)ri.end()};
        KEngine::TileIndex tj{(size_t)rj.begin(), (size_t)rj.end()};
        if (!D(ti, tj, block.data())) continue;
        for (size_t p = 0; p < ni; ++p) {
          for (size_t q = 0; q < nj; ++q) {
            d(ri.begin()+p, rj.begin()+q) = block[p*nj + q];
          }
        }
      }
    }
    return d;
  }

  /// Per-shell-block max|D|, the density half of the screening bound.
  inline std::vector<float> density_block_max(
    const Basis<Gaussian> &basis,
    const Matrix &d)
  {
    std::vector<float> m(basis.size()*basis.size(), 0.0f);
    for (size_t i = 0; i < basis.size(); ++i) {
      auto ri = basis.range(i);
      for (size_t j = 0; j < basis.size(); ++j) {
        auto rj = basis.range(j);
        float v = 0;
        for (int p = ri.begin(); p < ri.end(); ++p) {
          for (int q = rj.begin(); q < rj.end(); ++q) {
            v = std::max(v, (float)std::fabs(d(p,q)));
          }
        }
        m[i*basis.size() + j] = v;
      }
    }
    return m;
  }

  /// Hand the accumulated K back, one shell block per tile. A TileOut called
  /// with a null pointer is the query form: skip a block the caller declines
  /// rather than packing it.
  inline void scatter_exchange(
    const Basis<Gaussian> &basis,
    const Matrix &k,
    const KEngine::TileOut &K)
  {
    std::vector<double> block;
    for (size_t i = 0; i < basis.size(); ++i) {
      auto ri = basis.range(i);
      for (size_t j = 0; j < basis.size(); ++j) {
        auto rj = basis.range(j);
        KEngine::TileIndex ti{(size_t)ri.begin(), (size_t)ri.end()};
        KEngine::TileIndex tj{(size_t)rj.begin(), (size_t)rj.end()};
        if (!K(ti, tj, nullptr)) continue;
        size_t ni = (size_t)ri.size(), nj = (size_t)rj.size();
        block.resize(ni*nj);
        for (size_t p = 0; p < ni; ++p) {
          for (size_t q = 0; q < nj; ++q) {
            block[p*nj + q] = k(ri.begin()+p, rj.begin()+q);
          }
        }
        K(ti, tj, block.data());
      }
    }
  }

  /// Digest one computed batch of (ab|cd) into K.
  ///
  /// V is the engine's output for `bra.size()` bra pairs by `ket.size()` ket
  /// pairs, laid out exactly as the MD engines write it: a column-major
  /// (M*NA*NB) x (NC*ND*N) matrix, so element (ij, na, nb, nc, nd, kl) sits at
  /// ij + M*(na + NA*(nb + NB*(nc + NC*(nd + ND*kl)))).
  ///
  /// K[mu,nu] = sum (mu lambda | nu sigma) D[lambda,sigma] runs over *every*
  /// index tuple, so a unique quartet contributes once per distinct member of
  /// its eight-fold permutation orbit. Rather than carry per-case degeneracy
  /// factors -- which is where a K digest normally goes wrong, since a==b,
  /// c==d and (ab)==(cd) each collapse a different subset of the orbit -- the
  /// orbit is enumerated and deduplicated on the permuted shell tuple, and
  /// every survivor contributes with weight one. The permutation index is what
  /// says where in V each survivor's value lives.
  template<typename DensityMax>
  void digest(
    const Basis<Gaussian> &basis,
    const std::vector<ShellPair> &bra, const PairClass &bra_class, size_t bra_offset,
    const std::vector<ShellPair> &ket, const PairClass &ket_class, size_t ket_offset,
    bool same_class,
    const double *V,
    const Matrix &d,
    DensityMax &&dmax,
    const KEngine::Screening *screening,
    Matrix &k)
  {
    const size_t M = bra.size(), N = ket.size();
    const size_t NA = (size_t)bra_class.nbf_first, NB = (size_t)bra_class.nbf_second;
    const size_t NC = (size_t)ket_class.nbf_first, ND = (size_t)ket_class.nbf_second;
    const size_t stride[4] = { M, M*NA, M*NA*NB, M*NA*NB*NC };
    const size_t nbf[4] = { NA, NB, NC, ND };
    const size_t kl_stride = M*NA*NB*NC*ND;

    for (size_t kl = 0; kl < N; ++kl) {
      const auto &q = ket[kl];
      for (size_t ij = 0; ij < M; ++ij) {
        // One class on both sides is computed as a full rectangle, so each
        // quartet appears twice in the batch; keep the lower triangle in the
        // engine's global pair order and let the orbit restore the rest.
        if (same_class && (bra_offset + ij) < (ket_offset + kl)) continue;
        const auto &p = bra[ij];

        const int s[4] = { p.first, p.second, q.first, q.second };
        if (screening) {
          // K couples the bra indices to the ket indices, so the density
          // factor is the largest block over the cross terms, not D[ab] alone.
          float dm = std::max({
            dmax(s[0],s[1]), dmax(s[0],s[2]), dmax(s[0],s[3]),
            dmax(s[1],s[2]), dmax(s[1],s[3]), dmax(s[2],s[3])
          });
          if (screening->skip(p.norm*q.norm*dm)) continue;
        }

        int orbit[8];
        int norbit = 0;
        for (int t = 0; t < 8; ++t) {
          const int *pi = PERMUTATIONS[t];
          bool seen = false;
          for (int u = 0; u < norbit && !seen; ++u) {
            const int *pu = PERMUTATIONS[orbit[u]];
            seen = (s[pi[0]] == s[pu[0]] && s[pi[1]] == s[pu[1]] &&
                    s[pi[2]] == s[pu[2]] && s[pi[3]] == s[pu[3]]);
          }
          if (!seen) orbit[norbit++] = t;
        }

        const double *v = V + ij + kl*kl_stride;

        for (int t = 0; t < norbit; ++t) {
          const int *pi = PERMUTATIONS[orbit[t]];
          const int mu = basis.range(s[pi[0]]).begin();
          const int lambda = basis.range(s[pi[1]]).begin();
          const int nu = basis.range(s[pi[2]]).begin();
          const int sigma = basis.range(s[pi[3]]).begin();
          for (size_t n0 = 0; n0 < nbf[pi[0]]; ++n0) {
            for (size_t n1 = 0; n1 < nbf[pi[1]]; ++n1) {
              for (size_t n2 = 0; n2 < nbf[pi[2]]; ++n2) {
                for (size_t n3 = 0; n3 < nbf[pi[3]]; ++n3) {
                  size_t off = (
                    n0*stride[pi[0]] + n1*stride[pi[1]] +
                    n2*stride[pi[2]] + n3*stride[pi[3]]
                  );
                  k(mu+n0, nu+n2) += v[off]*d(lambda+n1, sigma+n3);
                }
              }
            }
          }
        }

      }
    }
  }

  /// The K build itself, parameterised on the four-centre engine.
  ///
  /// `Engine` needs
  ///   compute(Operator, const std::vector<Index2>&, const std::vector<Index2>&,
  ///           BraKet<const double*>, double*, const std::array<size_t,2>&)
  ///
  /// `Buffer` owns the integral batch and bridges the two back ends: `resize(n)`
  /// returns a pointer the engine may write n doubles to, and `synchronize()`
  /// is called before the digest reads it -- a no-op on the host, a stream
  /// synchronise (and, on the device path, the host-pointer registration a
  /// device write into host memory needs) on the GPU.
  template<typename Engine, typename Buffer>
  void build(
    const Basis<Gaussian> &basis,
    const std::vector<PairClass> &classes,
    Engine &engine,
    Buffer &buffer,
    const Matrix &d,
    const KEngine::Screening *screening,
    size_t max_batch,
    Matrix &k)
  {
    const auto dblock = density_block_max(basis, d);
    const size_t nshells = basis.size();
    auto dmax = [&](int i, int j) { return dblock[(size_t)i*nshells + j]; };
    const float dmax_global = (
      dblock.empty() ? 0.0f : *std::max_element(dblock.begin(), dblock.end())
    );

    std::vector<Index2> bra_index, ket_index;
    std::vector<double> bra_norms, ket_norms;
    std::vector<ShellPair> bra, ket;

    for (size_t cb = 0; cb < classes.size(); ++cb) {
      for (size_t ck = 0; ck <= cb; ++ck) {
        const auto &B = classes[cb];
        const auto &Q = classes[ck];
        if (screening && screening->skip(B.max_norm*Q.max_norm*dmax_global)) {
          continue;
        }

        const size_t block = (
          (size_t)B.nbf_first*B.nbf_second*Q.nbf_first*Q.nbf_second
        );
        // Bound a batch by the size of the integral buffer rather than by the
        // pair count: a (dd|dd) batch is 1296 doubles per quartet where an
        // (ss|ss) batch is one.
        const size_t quartets = std::max<size_t>(1, max_batch/block);
        const size_t nbra = std::max<size_t>(
          1, (size_t)std::sqrt((double)quartets)
        );
        const size_t nket = std::max<size_t>(1, quartets/nbra);
        const bool same_class = (cb == ck);

        for (size_t i0 = 0; i0 < B.pairs.size(); i0 += nbra) {
          size_t i1 = std::min(i0 + nbra, B.pairs.size());
          bra.assign(B.pairs.begin()+i0, B.pairs.begin()+i1);
          bra_index.clear(); bra_norms.clear();
          for (auto &p : bra) {
            bra_index.push_back(Index2{p.first, p.second});
            bra_norms.push_back(p.norm);
          }

          for (size_t j0 = 0; j0 < Q.pairs.size(); j0 += nket) {
            size_t j1 = std::min(j0 + nket, Q.pairs.size());
            // With one class on both sides only the lower triangle is
            // digested, so a chunk entirely above the diagonal is all waste.
            if (same_class && i1 <= j0) continue;

            ket.assign(Q.pairs.begin()+j0, Q.pairs.begin()+j1);
            ket_index.clear(); ket_norms.clear();
            for (auto &p : ket) {
              ket_index.push_back(Index2{p.first, p.second});
              ket_norms.push_back(p.norm);
            }

            const size_t M = bra.size(), N = ket.size();
            double *V = buffer.resize(M*N*block);
            std::array<size_t,2> dims = {
              M*(size_t)B.nbf_first*B.nbf_second,
              (size_t)Q.nbf_first*Q.nbf_second*N
            };
            BraKet<const double*> norms = {
              screening ? bra_norms.data() : nullptr,
              screening ? ket_norms.data() : nullptr
            };
            engine.compute(
              libintx::Coulomb, bra_index, ket_index, norms, V, dims
            );
            buffer.synchronize();

            digest(
              basis, bra, B, i0, ket, Q, j0, same_class,
              V, d, dmax, screening, k
            );
          }
        }
      }
    }
  }

}

#endif /* LIBINTX_GPU_KENGINE_MD_DRIVER_H */
