#include "libintx/ao/md/engine.h"
#include "libintx/ao/md/hermite.h"
#include "libintx/ao/md/r1.h"
#include "libintx/boys/chebyshev.h"
#include "libintx/math.h"
#include "libintx/simd.h"

#include "libintx/config.h"
#include "libintx/utility.h"

#include <stdexcept>

namespace libintx::md {

  template<typename S>
  auto params2(const Nuclear::Operator::Parameters &params) {
    const auto &v = params.centers;
    libintx_assert(v.size());
    if constexpr (std::is_scalar_v<S>) {
      return v;
    }
    else {
      constexpr int N = S::size();
      std::vector< std::tuple<S, std::array<S,3> > > s((v.size() + N - 1)/N);
      s.back() = { S(0), { S(0), S(0), S(0) } };
      for (size_t i = 0; i < v.size(); ++i) {
        std::get<0>(s[i/N])[i%N] = std::get<0>(v[i]);
        for (int k = 0; k < 3; ++k) {
          std::get<1>(s[i/N])[k][i%N] = std::get<1>(v[i])[k];
        }
      }
      return s;
    }
  }

  template<int A, int B, typename T>
  void cartesian_to_pure(const T (&V)[ncart(A)][ncart(B)], auto &&AB) {
    constexpr pure::Transform<A> pure_transform_a;
    constexpr pure::Transform<B> pure_transform_b;
    constexpr int NA = npure(A);
    T U[ncart(B)][NA] = {};
libintx_unroll(13)
    for (int lm = 0; lm < NA; ++lm) {
      constexpr int NA = ncart(A);
      constexpr int NB = ncart(B);
      T Ub[NB] = {};
libintx_unroll(28)
      for (int ia = 0; ia < NA; ++ia) {
        auto c = pure_transform_a.data[ia][lm];
        if (!c) continue;
        for (int ib = 0; ib < NB; ++ib) {
          Ub[ib] += c*V[ia][ib];
        }
      }
      for (int ib = 0; ib < NB; ++ib) {
        U[ib][lm] = Ub[ib];
      }
    }
    constexpr int NB = npure(B);
libintx_unroll(13)
    for (int lm = 0; lm < NB; ++lm) {
      T V[NA] = {};
      constexpr int NB = ncart(B);
libintx_unroll(28)
      for (int ib = 0; ib < NB; ++ib) {
        auto c = pure_transform_b.data[ib][lm];
        if (!c) continue;
        for (int ia = 0; ia < NA; ++ia) {
          V[ia] += c*U[ib][ia];
        }
      }
      for (int ia = 0; ia < NA; ++ia) {
        AB(ia,lm) = V[ia];
      }
    } // ib
  }


  template<typename T, int A, int B>
  void overlap(T a1, T a2, auto &&R, T C, auto &&U) {
    using std::sqrt;
    using cartesian::orbitals;
    using cartesian::index;
    E2<T,A,B,0> E(a1,a2,R);
    C *= sqrt(math::pow<3>(math::pi/(a1+a2)));
libintx_unroll(28)
    for (auto &a : orbitals<A>()) {
libintx_unroll(28)
      for (auto &b : orbitals<B>()) {
        U[index(a)][index(b)] += C*E(a, b, Orbital{{0,0,0}});

      }
    }
  }

  template<typename T, int A, int B>
  void kinetic(T a1, T a2, auto &&R, T C, auto &&U) {
    using std::sqrt;
    using cartesian::orbitals;
    using cartesian::index;
    E2<T,A,B+2,0> E(a1,a2,R);
    C *= sqrt(math::pow<3>(math::pi/(a1+a2)));
libintx_unroll(28)
    for (auto [l1,m1,n1] : orbitals<A>()) {
libintx_unroll(28)
      for (auto [l2,m2,n2] : orbitals<B>()) {
        T t0 = E.x(l1,l2)*E.y(m1,m2)*E.z(n1,n2);
        T t1 = (
          E.x(l1,l2+2)*E.y(m1,m2)*E.z(n1,n2) +
          E.x(l1,l2)*E.y(m1,m2+2)*E.z(n1,n2) +
          E.x(l1,l2)*E.y(m1,m2)*E.z(n1,n2+2)
        );
        T t2 = {};
        int l = l2*(l2-1);
        int m = m2*(m2-1);
        int n = n2*(n2-1);
        if (l) t2 += l*E.x(l1,l2-2)*E.y(m1,m2)*E.z(n1,n2);
        if (m) t2 += m*E.x(l1,l2)*E.y(m1,m2-2)*E.z(n1,n2);
        if (n) t2 += n*E.x(l1,l2)*E.y(m1,m2)*E.z(n1,n2-2);
        U(index(l1,m1,n1),index(l2,m2,n2)) += C*(a2*(2*B+3)*t0 - 2*a2*a2*t1 - 0.5*t2);
      }
    }
  }

  /// `dS/dA_x` for one primitive pair -- the host counterpart of
  /// `gpu/overlap/overlap.cu`'s `OverlapD1`, written against this file's own
  /// `E2<T,A,B,P>` and `orbitals<A>()` rather than transcribed from it, so
  /// host/device agreement is evidence rather than a tautology.
  ///
  /// Differentiating a Cartesian primitive with respect to its own centre
  /// raises and lowers one Cartesian index,
  ///
  ///     d/dA_x (a|  =  2*alpha*(a+1_x|  -  i_x*(a-1_x|
  ///
  /// and the overlap factorises axis by axis, so the derivative is the same
  /// product with one axis replaced. The raised coefficient comes out of
  /// `E2<T,A+1,B,0>`: **nothing here shifts a shell**, so the contraction
  /// coefficients stay the parent's and `gto::normalized`'s L-dependent factor
  /// never enters. `C` carries `K_ab`, which depends on the bra centre too --
  /// that is not a missing term, because the raising relation is an identity
  /// on the primitive function and `E`'s own recursion carries the whole
  /// A-dependence.
  ///
  /// `U[x][ia][ib]`, three components; only the bra derivative, since
  /// `dS/dB = -dS/dA` elementwise.
  template<typename T, int A, int B>
  void overlap1(T a1, T a2, auto &&R, T C, auto &&U) {
    using std::sqrt;
    using cartesian::orbitals;
    using cartesian::index;
    E2<T,A+1,B,0> E(a1,a2,R);
    C *= sqrt(math::pow<3>(math::pi/(a1+a2)));
libintx_unroll(28)
    for (auto &a : orbitals<A>()) {
libintx_unroll(28)
      for (auto &b : orbitals<B>()) {
        T e[3];
        for (int y = 0; y < 3; ++y) {
          e[y] = E(a[y], b[y], 0, y);
        }
        for (int x = 0; x < 3; ++x) {
          T d = 2*a1*E(a[x]+1, b[x], 0, x);
          if (a[x]) d -= a[x]*E(a[x]-1, b[x], 0, x);
          U[x][index(a)][index(b)] += C*d*e[(x+1)%3]*e[(x+2)%3];
        }
      }
    }
  }

  /// `dT/dA_x` for one primitive pair.
  ///
  /// `d/dA_x` acts on the bra Cartesian index alone, so the three-term
  /// structure of `kinetic` above is untouched and every factor on axis `x` is
  /// replaced by its derivative. Two things that are easy to get wrong:
  ///
  ///  - **`2*B+3` does not move.** It is the KET shell's angular momentum,
  ///    which raising the bra index does not change.
  ///  - **Every exponent in the body is still the ket's.** `a1` appears
  ///    exactly once, in the raising relation.
  ///
  /// **The transpose branch `compute2` uses for the kinetic *value* is not
  /// replicated here, deliberately.** That branch evaluates `(A|B)` as
  /// `kinetic<B,A>` on the swapped, sign-flipped pair when `B > A`; a
  /// derivative makes the trade unavailable, because the raised index is no
  /// longer symmetric between bra and ket and a transposing accessor would
  /// have to know which centre was differentiated. The device kernel does not
  /// replicate it either, for its own reasons (CLAUDE.md).
  template<typename T, int A, int B>
  void kinetic1(T a1, T a2, auto &&R, T C, auto &&U) {
    using std::sqrt;
    using cartesian::orbitals;
    using cartesian::index;
    E2<T,A+1,B+2,0> E(a1,a2,R);
    C *= sqrt(math::pow<3>(math::pi/(a1+a2)));
libintx_unroll(28)
    for (auto &a : orbitals<A>()) {
libintx_unroll(28)
      for (auto &b : orbitals<B>()) {
        // Per axis, the three ket degrees the body reads -- j, j+2, j-2 -- as
        // the plain coefficient and as its bra derivative. A negative degree
        // is left at zero, which is not an approximation: it only ever
        // multiplies the j*(j-1) coefficient, which is zero there.
        T e[3][3] = {};
        T d[3][3] = {};
        for (int y = 0; y < 3; ++y) {
          for (int k = 0; k < 3; ++k) {
            int j = (k == 0 ? b[y] : (k == 1 ? b[y]+2 : b[y]-2));
            if (j < 0) continue;
            e[y][k] = E(a[y], j, 0, y);
            T v = 2*a1*E(a[y]+1, j, 0, y);
            if (a[y]) v -= a[y]*E(a[y]-1, j, 0, y);
            d[y][k] = v;
          }
        }
        int c2[3] = { b[0]*(b[0]-1), b[1]*(b[1]-1), b[2]*(b[2]-1) };
        for (int x = 0; x < 3; ++x) {
          auto g = [&](int y, int k) { return (y == x ? d[y][k] : e[y][k]); };
          T t0 = g(0,0)*g(1,0)*g(2,0);
          T t1 = (
            g(0,1)*g(1,0)*g(2,0) +
            g(0,0)*g(1,1)*g(2,0) +
            g(0,0)*g(1,0)*g(2,1)
          );
          T t2 = (
            c2[0]*g(0,2)*g(1,0)*g(2,0) +
            c2[1]*g(0,0)*g(1,2)*g(2,0) +
            c2[2]*g(0,0)*g(1,0)*g(2,2)
          );
          U[x][index(a)][index(b)] += C*(a2*(2*B+3)*t0 - 2*a2*a2*t1 - 0.5*t2);
        }
      }
    }
  }

  template<int A, int B, typename T, typename Z>
  void nuclear(
    double a1, auto &&r1, double a2, auto &&r2,
    const std::vector< std::tuple<Z,std::array<T,3> > > &Cs,
    double C, auto &&U)
  {

    using std::sqrt;
    using cartesian::orbitals;
    using cartesian::index;

    auto &boys = libintx::boys::chebyshev<2*LMAX,A+B>();

    assert(!Cs.empty());

    constexpr int NP = nherm2(A+B);

    T p = a1 + a2;
    array<T,3> P;
    for (int i = 0; i < 3; ++i) {
      P[i] = center_of_charge(a1, r1, a2, r2)[i];
    }
    T R[NP] = {};
    for (size_t i = 0; i < Cs.size(); ++i) {

      auto& [Zi,Ci] = Cs[i];
      auto PC = P-Ci;

      if constexpr (!is_simd_v<T>) {
        if (!Zi) continue;
      }

      T s[A+B+1] = { };//, a2, p, a1*a2, 1, 1 };
      auto x = p*norm(PC);
      boys.template compute<A+B>(x, s);
      T pi = 1;
      libintx_unroll(13)
      for (int i = 0; i <= A+B; ++i) {
        s[i] *= pi;
        //printf("F_[%i](%f) = %f\n", i, (double)x[3], (double)s[i][3]);
        pi *= -2*p;
      }

      auto V = [&,Zi=Zi](auto &&r) {
        R[r.index] += -Zi*r.value;
      };
      namespace r1 = libintx::md::r1;
      r1::visit<A+B>(V, PC, s);
    }

    double R1[NP] = {};
    if constexpr (std::is_scalar_v<T>) {
      for (size_t i = 0; i < NP; ++i) {
        R1[i] = R[i];
      }
    }
    else {
      for (size_t i = 0; i < NP; ++i) {
        for (size_t k = 0; k < T::size(); ++k) {
          R1[i] += R[i][k];
        }
        //printf("R[i] = %f\n", double(R1[i]));
      }
    }

    E2<double,A,B,A+B> E(a1,a2,r1-r2);
    C *= 2*math::pi/(a1+a2);

    foreach2(
      std::make_index_sequence<ncart(A)>{},
      std::make_index_sequence<ncart(B)>{},
      [&](auto ia, auto ib) {
        constexpr auto a = orbitals<A>()[ia];
        constexpr auto b = orbitals<B>()[ib];
        constexpr auto ab = a+b;
        constexpr auto PX = ab.lmn[0];
        constexpr auto PY = ab.lmn[1];
        constexpr auto PZ = ab.lmn[2];
        double u = 0;
        auto *Ex = E.px(a,b);
        auto *Ey = E.py(a,b);
        auto *Ez = E.pz(a,b);
        libintx_unroll(13)
        for (uint8_t iz = 0; iz <= PZ; ++iz) {
          libintx_unroll(13)
          for (uint8_t iy = 0; iy <= PY; ++iy) {
            double Eyz = Ey[iy]*Ez[iz];
            libintx_unroll(13)
            for (uint8_t ix = 0; ix <= PX; ++ix) {
              auto p = Orbital{ix,iy,iz};
              u += R1[hermite::index2(p)]*Ex[ix]*Eyz;
            }
          }
        }
        U[ia][ib] += C*u;
      }
    );

  }

  template<int A, int B, Operator Op, typename T, int KMAX>
  void compute2(
    const auto &params,
    const array<T,3> &r1,
    const array<T,3> &r2,
    std::pair<int,int> K,
    const gto::Primitive<T> (&g1)[KMAX],
    const gto::Primitive<T> (&g2)[KMAX],
    auto &&V)
  {
    T U[ncart(A)][ncart(B)] = {};
    array<T,3> R = r1-r2;
    T r = norm(R);
    for (int k1 = 0; k1 < K.first; ++k1) {
      for (int k2 = 0; k2 < K.second; ++k2) {
        auto [ei,Ci] = g1[k1];
        auto [ej,Cj] = g2[k2];
        using std::exp;
        T Kab = exp(-ei*ej/(ei+ej)*r);
        auto C = Kab*Ci*Cj;
        if constexpr (Op == Operator::Overlap) {
          overlap<T,A,B>(ei, ej, R, C, U);
        }
        if constexpr (Op == Operator::Kinetic) {
          constexpr bool Transpose = (B > A);
          auto Uij = [&](auto i, auto j) ->auto& {
            return (Transpose ? U[j][i] : U[i][j]);;
          };
          if (!Transpose) {
            kinetic<T,A,B>(ei, ej, R, C, Uij);
          }
          else {
            kinetic<T,B,A>(ej, ei, -R, C, Uij);
          }
        }
        if constexpr (Op == Operator::Nuclear) {
          static_assert(!is_simd_v<T>);
          nuclear<A,B>(ei, r1, ej, r2, params, C, U);
        }
      }
    }
    cartesian_to_pure<A,B>(U,V);
  }

  //using Gaussian2 = std::tuple<Gaussian,Gaussian>;
  using Gaussian2 = std::tuple<const Gaussian&, const Gaussian&>;

  /// `compute2`'s derivative twin: three Cartesian components, one solid-
  /// harmonic pass each.
  ///
  /// Only `Overlap` and `Kinetic` come through here. `Nuclear` has a second
  /// contribution indexed by nucleus and does not fit this shape -- see
  /// `ao::IntegralEngine<2>::compute1`'s 4-argument overload.
  template<int A, int B, Operator Op, typename T, int KMAX>
  void compute2_1(
    const array<T,3> &r1,
    const array<T,3> &r2,
    std::pair<int,int> K,
    const gto::Primitive<T> (&g1)[KMAX],
    const gto::Primitive<T> (&g2)[KMAX],
    auto &&V)
  {
    static_assert(Op == Operator::Overlap || Op == Operator::Kinetic);
    T U[3][ncart(A)][ncart(B)] = {};
    array<T,3> R = r1-r2;
    T r = norm(R);
    for (int k1 = 0; k1 < K.first; ++k1) {
      for (int k2 = 0; k2 < K.second; ++k2) {
        auto [ei,Ci] = g1[k1];
        auto [ej,Cj] = g2[k2];
        using std::exp;
        T Kab = exp(-ei*ej/(ei+ej)*r);
        auto C = Kab*Ci*Cj;
        if constexpr (Op == Operator::Overlap) {
          overlap1<T,A,B>(ei, ej, R, C, U);
        }
        if constexpr (Op == Operator::Kinetic) {
          kinetic1<T,A,B>(ei, ej, R, C, U);
        }
      }
    }
    // The solid-harmonic transform is linear with constant coefficients, so it
    // commutes with d/dA_x: each component transforms exactly as a value does.
    for (int x = 0; x < 3; ++x) {
      cartesian_to_pure<A,B>(
        U[x],
        [&](auto ia, auto lm) -> auto& { return V(ia,lm,x); }
      );
    }
  }

  template<int A, int B, Operator Op, typename T>
  void compute2_1(
    const std::vector<Gaussian2> &abs,
    T* __restrict__ V, int ldV)
  {

    constexpr size_t N = simd::size<T,1>;
    constexpr int NA = npure(A);
    constexpr int NB = npure(B);

    for (size_t i = 0; i < abs.size(); i += N) {

      size_t Ni = std::min(N, abs.size()-i);

      const auto a = [&](size_t idx) -> const Gaussian* {
        if (idx >= Ni) return nullptr;
        return &std::get<0>(abs[i + idx]);
      };

      const auto b = [&](size_t idx) -> const Gaussian* {
        if (idx >= Ni) return nullptr;
        return &std::get<1>(abs[i + idx]);
      };

      std::pair<int,int> K = { 0, 0 };
      for (size_t j = 0; j < N; ++j) {
        K.first = std::max(K.first, (a(j) ? a(j)->K : 0));
        K.second = std::max(K.second, (b(j) ? b(j)->K : 0));
      }
      assert(K.first && K.second);

      const auto &r1 = gto::pack_centers<T>(a);
      const auto &r2 = gto::pack_centers<T>(b);

      const auto &g1 = gto::pack_primitives<T>(a);
      const auto &g2 = gto::pack_primitives<T>(b);

      // The value layout with the Cartesian component as one more, slowest
      // index -- ao::IntegralEngine<2>::compute1's contract.
      auto f = [i,V,ldV](auto ia, auto ib, int x) -> auto& {
        return V[i/N + (ia + ib*NA + x*NA*NB)*ldV];
      };

      compute2_1<A,B,Op,T>(r1,r2,K,g1.data,g2.data,f);

    }
  }

  template<int A, int B, Operator Op, typename Params, typename T>
  void compute2(
    const Params& params,
    const std::vector<Gaussian2> &abs,
    T* __restrict__ V, int ldV)
  {

    constexpr size_t N = simd::size<T,1>;

    for (size_t i = 0; i < abs.size(); i += N) {

      size_t Ni = std::min(N, abs.size()-i);

      const auto a = [&](size_t idx) -> const Gaussian* {
        if (idx >= Ni) return nullptr;
        return &std::get<0>(abs[i + idx]);
      };

      const auto b = [&](size_t idx) -> const Gaussian* {
        if (idx >= Ni) return nullptr;
        return &std::get<1>(abs[i + idx]);
      };

      std::pair<int,int> K = { 0, 0 };
      for (size_t j = 0; j < N; ++j) {
        K.first = std::max(K.first, (a(j) ? a(j)->K : 0));
        K.second = std::max(K.second, (b(j) ? b(j)->K : 0));
      }
      assert(K.first && K.second);

      const auto &r1 = gto::pack_centers<T>(a);
      const auto &r2 = gto::pack_centers<T>(b);

      const auto &g1 = gto::pack_primitives<T>(a);
      const auto &g2 = gto::pack_primitives<T>(b);

      auto f = [i,V,ldV](auto ia, auto ib) ->auto& {
        return V[i/N + (ia + ib*npure(A))*ldV];
      };

      compute2<A,B,Op,T>(params,r1,r2,K,g1.data,g2.data,f);

    }
  }

  template<typename T, Operator Op, typename Params>
  void IntegralEngine<2>::compute(const Params &params, const std::vector<Index2> &ijs, const Visitor &V) {

    using Kernel = std::function<void(
      const Params&,
      const std::vector<Gaussian2>&,
      T* __restrict__, int ldV
    )>;

    static auto kernel_array = make_array<Kernel,LMAX+1,LMAX+1>(
      [](auto a, auto b) {
        return Kernel(&md::compute2<a,b,Op,Params,T>);
      }
    );

    auto [i,j] = ijs.front();
    const auto &a = this->bra_[i];
    const auto &b = this->ket_[j];
    auto kernel = kernel_array[a.L][b.L];

    constexpr int N = simd::size<T,1>;
    int Batch = (this->Batch ? this->Batch : N);

#pragma omp parallel num_threads(this->num_threads)
    {

      // int K = nprim(a)*nprim(b);
      int ldV = (Batch+N-1)/N;
      auto V_batch = std::make_unique<T[]>(npure(a.L,b.L)*ldV);

      std::vector<Gaussian2> batch;
      batch.reserve(Batch);

#pragma omp for schedule(dynamic,1)
      for (size_t ij = 0; ij < ijs.size(); ij += Batch) {
        size_t nij = std::min<size_t>(ijs.size()-ij,Batch);
        batch.clear();
        for (size_t k = 0; k < nij; ++k) {
          const auto& [i,j] = ijs[ij+k];
          batch.emplace_back(std::tie(bra_[i], ket_[j]));
        }
        kernel(params, batch, V_batch.get(), ldV);
        V(nij,ij,reinterpret_cast<double*>(V_batch.get()),N*ldV);
      }

    }

  }

  void IntegralEngine<2>::compute(Operator op, const std::vector<Index2> &ij, const Visitor &V) {
#ifdef LIBINTX_SIMD_DOUBLE
    using S = LIBINTX_SIMD_DOUBLE;
#else
    using S = double;
#endif
    if (op == Overlap) {
      this->compute<S,Overlap>(Overlap::Operator::Parameters{},ij,V);
    }
    if (op == Kinetic) {
      this->compute<S,Kinetic>(Kinetic::Operator::Parameters{},ij,V);
    }
    if (op == Nuclear) {
      const auto &params = params2<S>(std::get<Nuclear::Operator::Parameters>(this->params_));
      this->compute<double,Nuclear>(params,ij,V);
    }
  }

  void IntegralEngine<2>::compute(Operator op, const std::vector<Index2> &ijs, double *V) {
    int na = nbf(bra_[ijs[0].first]);
    int nb = nbf(bra_[ijs[0].second]);
    size_t ldV = ijs.size();
    auto v = [&](size_t batch, size_t idx, const double *U, size_t ldU) {
      for (int iab = 0; iab < na*nb; ++iab) {
        auto *dst = V + iab*ldV + idx;
        auto *src = U + iab*ldU;
        std::copy_n(src, batch, dst);
      }
    };
    this->compute(op, ijs, v);
  }

  template<typename T, Operator Op>
  void IntegralEngine<2>::compute1(const std::vector<Index2> &ijs, const Visitor &V) {

    using Kernel = std::function<void(
      const std::vector<Gaussian2>&, T* __restrict__, int ldV
    )>;

    static auto kernel_array = make_array<Kernel,LMAX+1,LMAX+1>(
      [](auto a, auto b) {
        return Kernel(&md::compute2_1<a,b,Op,T>);
      }
    );

    auto [i,j] = ijs.front();
    const auto &a = this->bra_[i];
    const auto &b = this->ket_[j];
    auto kernel = kernel_array[a.L][b.L];

    constexpr int N = simd::size<T,1>;
    int Batch = (this->Batch ? this->Batch : N);

#pragma omp parallel num_threads(this->num_threads)
    {

      int ldV = (Batch+N-1)/N;
      auto V_batch = std::make_unique<T[]>(3*npure(a.L,b.L)*ldV);

      std::vector<Gaussian2> batch;
      batch.reserve(Batch);

#pragma omp for schedule(dynamic,1)
      for (size_t ij = 0; ij < ijs.size(); ij += Batch) {
        size_t nij = std::min<size_t>(ijs.size()-ij,Batch);
        batch.clear();
        for (size_t k = 0; k < nij; ++k) {
          const auto& [i,j] = ijs[ij+k];
          batch.emplace_back(std::tie(bra_[i], ket_[j]));
        }
        kernel(batch, V_batch.get(), ldV);
        V(nij,ij,reinterpret_cast<double*>(V_batch.get()),N*ldV);
      }

    }

  }

  void IntegralEngine<2>::compute1(Operator op, const std::vector<Index2> &ijs, double *V) {

    libintx_assert(!ijs.empty());

    // Nuclear has a host derivative kernel too, but its Hellmann-Feynman term
    // is indexed by nucleus and does not fit this buffer -- the same split the
    // device engine makes, and for the same reason: writing the shell half and
    // dropping the rest is smooth, plausible and wrong by the dominant part of
    // the force on a charged atom.
    if (op == Nuclear) {
      throw std::runtime_error(
        "libintx::md::IntegralEngine<2>::compute1: Operator::Nuclear has a"
        " second, nucleus-indexed derivative term (Hellmann-Feynman) that does"
        " not fit this buffer; use the compute1(op,ijs,dV,dVC) overload"
      );
    }

    if (op != Overlap && op != Kinetic) {
      throw std::runtime_error(
        str(
          "libintx::md::IntegralEngine<2>::compute1: no host derivative kernel"
          " for operator ", (int)op
        )
      );
    }

#ifdef LIBINTX_SIMD_DOUBLE
    using S = LIBINTX_SIMD_DOUBLE;
#else
    using S = double;
#endif

    int na = nbf(bra_[ijs[0].first]);
    int nb = nbf(ket_[ijs[0].second]);
    size_t ldV = ijs.size();
    auto v = [&](size_t batch, size_t idx, const double *U, size_t ldU) {
      for (int iab = 0; iab < 3*na*nb; ++iab) {
        auto *dst = V + iab*ldV + idx;
        auto *src = U + iab*ldU;
        std::copy_n(src, batch, dst);
      }
    };

    if (op == Overlap) this->compute1<S,Overlap>(ijs,v);
    if (op == Kinetic) this->compute1<S,Kinetic>(ijs,v);

  }

  void IntegralEngine<2>::compute1(
    Operator op, const std::vector<Index2> &ijs, double *dV, double *dVC)
  {
    (void)ijs;
    (void)dV;
    (void)dVC;
    throw std::runtime_error(
      str(
        "libintx::md::IntegralEngine<2>::compute1: no host derivative kernel"
        " for operator ", (int)op
      )
    );
  }

  IntegralEngine<2>::IntegralEngine(const Basis<Gaussian> &bra, const Basis<Gaussian> &ket)
    : bra_(bra), ket_(ket)
  {
    libintx_assert(!bra_.empty());
    libintx_assert(!ket_.empty());
  }

  IntegralEngine<2>::~IntegralEngine() {}

} // libintx::md

template<>
std::unique_ptr< libintx::ao::IntegralEngine<2> > libintx::ao::integral_engine(
  const Basis<Gaussian> &bra,
  const Basis<Gaussian> &ket)
{
  return std::make_unique< libintx::md::IntegralEngine<2> >(bra,ket);
}
