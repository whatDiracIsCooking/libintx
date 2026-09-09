#ifndef LIBINTX_MD_REFERENCE_H
#define LIBINTX_MD_REFERENCE_H

#include "libintx/boys/reference.h"
#include "libintx/orbital.h"
#include "libintx/forward.h"

#include <utility>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace libintx::md::reference {

  const double sqrt_4_pi5 = std::sqrt(4*std::pow(M_PI,5));

  using double3 = double[3];
  using libintx::Orbital;

  constexpr auto orbital(int idx) {
    constexpr auto orbitals = hermite::orbitals2<max(XMAX,2*LMAX)+2*LMAX>;
    assert((size_t)idx < orbitals.size());
    return orbitals[idx];
  }

  LIBINTX_GPU_ENABLED
  inline double E(int i, int j, int k, double a, double b, double R);

  // The bare three-term McMurchie-Davidson recursion. Every step recurses back
  // through E() below, so the memo table there short-circuits the shared
  // subtrees; on the device, where there is no table, this is the whole of E.
  LIBINTX_GPU_ENABLED
  inline double E_recurrence(int i, int j, int k, double a, double b, double R) {
    //printf("E(%i,%i,%i)\n", i, j, k);
    auto p = a + b;
    assert(p);
    auto q = ((a ? a : 1)*(b ? b : 1))/p;
    assert(q);
    if (!a || !b) R = 0;
    if (k > (i + j)) return 0;
    if ((k < 0) || (i < 0) || (j < 0)) return 0;
    if (i == 0 && j == 0 && k == 0) {
      return 1;//std::exp(-q*R*R); // K_AB
    }
    if (i) {
      // decrement index i
      assert(i && a);
      return (
        (1/(2*p))*E(i-1,j,k-1,a,b,R) -
        (q*R/a)*E(i-1,j,k,a,b,R)    +
        (k+1)*E(i-1,j,k+1,a,b,R)
      );
    }
    else {
      // decrement index j
      assert(j && b);
      return (
        (1/(2*p))*E(i,j-1,k-1,a,b,R) +
        (q*R/b)*E(i,j-1,k,a,b,R) +
        (k+1)*E(i,j-1,k+1,a,b,R)
      );
    }
  }

#ifndef __CUDA_ARCH__

  namespace detail {

    // E is a pure function of its arguments and is called with the same ones
    // an enormous number of times: the bare recursion above costs ~1,800
    // nested calls for i=j=3 summed over k, and it is driven once per
    // Cartesian axis, per Hermite index, per primitive quartet. That recursion
    // -- the test oracle, not libintx -- is where libintx.md4.test's runtime
    // at LIBINTX_MAX_L=3 goes (issue #15). Memoising a pure function cannot
    // change a value, only stop recomputing one.
    //
    // The (a,b,R) triple is constant through any one top-level call's whole
    // recursion tree, but callers rotate between a handful of triples -- three
    // Cartesian axes of the bra and three of the ket. So the cache is a small
    // set of fixed-size (i,j,k) tables, each tagged with the triple it holds,
    // recycled least-recently-used. Fixed size is the point: it stays at tens
    // of kB over an hours-long sweep instead of growing without limit, and a
    // hit costs an array index rather than a hash of six fields.
    //
    // thread_local because md::IntegralEngine takes an OpenMP num_threads, so
    // nothing guarantees the reference is only ever reached from one thread.
    //
    // The whole table is host-only: E is LIBINTX_GPU_ENABLED, and a static
    // table in a __device__ function does not compile. Under nvcc's device
    // pass __CUDA_ARCH__ is defined and none of this exists, leaving the plain
    // recursion the device has always used.

    constexpr int E_MAX_IJ = max(LMAX,XMAX) + 2; // +1 covers Kinetic's A+/-dx
    constexpr int E_MAX_K  = 2*E_MAX_IJ + 1;
    constexpr int E_TABLE  = (E_MAX_IJ+1)*(E_MAX_IJ+1)*(E_MAX_K+1);
    constexpr int E_SLOTS  = 8;                  // 6 triples are live at once

    // Tag on the bit pattern rather than the value, so that -0.0 and 0.0 land
    // in separate slots (same answer, one extra miss) instead of comparing
    // equal in one place and unequal in another.
    inline uint64_t E_bits(double x) {
      uint64_t u;
      std::memcpy(&u, &x, sizeof(u));
      return u;
    }

    struct E_slot {
      uint64_t a = 0, b = 0, R = 0;
      bool live = false;
      unsigned long stamp = 0;
      bool cached[E_TABLE] = {};
      double value[E_TABLE] = {};
    };

    inline E_slot& E_slot_for(double a, double b, double R) {
      static thread_local E_slot slots[E_SLOTS];
      static thread_local unsigned long tick = 0;
      auto ka = E_bits(a), kb = E_bits(b), kR = E_bits(R);
      ++tick;
      E_slot *lru = slots;
      for (auto &s : slots) {
        if (s.live && s.a == ka && s.b == kb && s.R == kR) {
          s.stamp = tick;
          return s;
        }
        if (s.stamp < lru->stamp) lru = &s;
      }
      // Miss: recycle the least recently used slot for this triple.
      lru->a = ka;
      lru->b = kb;
      lru->R = kR;
      lru->live = true;
      lru->stamp = tick;
      std::memset(lru->cached, 0, sizeof(lru->cached));
      return *lru;
    }

  }

#endif // __CUDA_ARCH__

  LIBINTX_GPU_ENABLED
  inline double E(int i, int j, int k, double a, double b, double R) {
#ifndef __CUDA_ARCH__
    if (i >= 0 && i <= detail::E_MAX_IJ &&
        j >= 0 && j <= detail::E_MAX_IJ &&
        k >= 0 && k <= detail::E_MAX_K)
    {
      const int idx = (i*(detail::E_MAX_IJ+1) + j)*(detail::E_MAX_K+1) + k;
      {
        detail::E_slot &s = detail::E_slot_for(a,b,R);
        if (s.cached[idx]) return s.value[idx];
      }
      // Do not hold that reference across the recursion: a nested call can in
      // principle land on a different triple (E_recurrence rewrites R to 0
      // when a or b is zero) and recycle the slot, so look it up again.
      const double v = E_recurrence(i,j,k,a,b,R);
      detail::E_slot &s = detail::E_slot_for(a,b,R);
      s.cached[idx] = true;
      s.value[idx] = v;
      return v;
    }
#endif
    return E_recurrence(i,j,k,a,b,R);
  }

  double R(int t, int u, int v, int n, const auto &s, const double3 &r) {
    if (!t && !u && !v) return s[n];
    if (!t && !u) {
      double value = 0.0;
      if (v > 1)
        value += (v-1)*R(t,u,v-2,n+1,s,r);
      value += r[2]*R(t,u,v-1,n+1,s,r);
      return value;
    }
    if (!t) {
      double value = 0.0;
      if (u > 1)
        value += (u-1)*R(t,u-2,v,n+1,s,r);
      value += r[1]*R(t,u-1,v,n+1,s,r);
      return value;
    }
    {
      double value = 0.0;
      if (t > 1)
        value += (t-1)*R(t-2,u,v,n+1,s,r);
      value += r[0]*R(t-1,u,v,n+1,s,r);
      return value;
    }
  }

  void compute_r1(int L, double alpha, const auto &PQ, auto &r1) {

    std::vector<double> s(L+1);

    for (int m = 0; m <= L; ++m) {
      double Fm = boys::Reference().compute(alpha*norm(PQ), m);
      s[m] = Fm*pow(-2*alpha,m);
    }

    for (int i = 0; i < nherm2(L); ++i) {
      auto [x,y,z] = reference::orbital(i);
      r1[i] = md::reference::R(x, y, z, 0, s.data(), PQ.data);
    }

  }

  template<Operator>
  struct Integral;

  template<>
  struct Integral<Overlap> {
    template<typename Orbital = Orbital>
    static double compute(Orbital i, double ai, auto ri, Orbital j, double aj, auto rj, struct None = {}) {
      auto r = ri-rj;
      double v = 1;
      v *= std::pow(math::pi/(ai+aj),1.5);
      for (int ix = 0; ix < 3; ++ix) {
        v *= std::exp(-ai*aj*r[ix]*r[ix]/(ai+aj));
        v *= E(i[ix],j[ix],0,ai,aj,r[ix]);
      }
      return v;
    }
  };

  template<>
  struct Integral<Kinetic> {
    static double compute(auto A, double a1, auto r1, auto B, double a2, auto r2, struct None = {}) {
      Integral<Overlap> overlap;
      double t = 0;
      for (int ix = 0; ix < 3; ++ix) {
        int ax = A[ix];
        int bx = B[ix];
        int abx = ax*bx;
        Orbital dx = {0,0,0};
        dx[ix] = 1;
        t += 2*a1*a2*overlap.compute(A+dx,a1,r1,B+dx,a2,r2);
        if (bx) t -= a1*bx*overlap.compute(A+dx,a1,r1,B-dx,a2,r2);
        if (ax) t -= a2*ax*overlap.compute(A-dx,a1,r1,B+dx,a2,r2);
        if (abx) t += 0.5*abx*overlap.compute(A-dx,a1,r1,B-dx,a2,r2);
      }
      return t;
    }
  };

  template<>
  struct Integral<Nuclear> {
    static double compute(auto A, double a1, auto r1, auto B, double a2, auto r2, const auto &Cs) {
      assert(!Cs.empty());
      int L = A.L()+B.L();
      auto P = center_of_charge(a1, r1, a2, r2);
      double v = 0;
      for (auto [Z,C] : Cs) {
        std::vector<double> R(nherm2(L));
        compute_r1(L, a1+a2, P-C, R);
        for (int ip = 0; ip < nherm2(L); ++ip) {
          auto p =  reference::orbital(ip);
          double e = 1;
          for (int i = 0; i < 3; ++i) {
            e *= E(A[i], B[i], p[i], a1, a2, r1[i]-r2[i]);
          }
          v += -Z*e*R[ip];
        }
      }
      v *= std::exp(-(a1*a2)/(a1+a2)*norm(r1,r2));
      v *= 2*M_PI/(a1+a2);
      // printf(
      //   "%f, np.array([%f,%f,%f]), a2=%f, r2=np.array([%f,%f,%f]) v=%f\n",
      //   a1, r1[0], r1[1], r1[2],
      //   a2, r2[0], r2[1], r2[2],
      //   v
      // );
      return v;
    }
  };

  template<Operator Op>
  inline void compute2(const auto &A, const auto &B, const auto& Params, auto &&V) {
    using cartesian::orbitals;
    using cartesian::index;
    Integral<Op> integral;
    for (auto a : orbitals(A)) {
      for (auto b : orbitals(B)) {
        double v = 0;
        for (int kb = 0; kb < B.K; ++kb) {
          for (int ka = 0; ka < A.K; ++ka) {
            double C = 1;
            C *= primitive(A,ka).C;
            C *= primitive(B,kb).C;
            v += C*integral.compute(
              a,exp(A,ka),center(A),
              b,exp(B,kb),center(B),
              Params
            );
          }
        }
        V(index(a),index(b)) = v;
      }
    }
  }

  void compute_p_q(int Bra, int Ket, const auto &r1, auto &pq) {
    for (int ip = 0; ip < nherm2(Bra); ++ip) {
      for (int iq = 0; iq < nherm2(Ket); ++iq) {
        auto p = reference::orbital(ip);
        auto q = reference::orbital(iq);
        pq(ip,iq) = r1[hermite::index2(p+q)];
      }
    }
  }

  void compute_p_cd(const auto &Ak, const auto &Bk, const auto &C, const auto &D, auto &pCD) {

    auto &[A,ka] = Ak;
    auto &[B,kb] = Bk;

    int Bra = A.L + B.L;
    int Ket = C.L + D.L;

    auto AB = center(A) - center(B);
    auto CD = center(C) - center(D);

    for (int kd = 0; kd < D.K; ++kd) {
      for (int kc = 0; kc < C.K; ++kc) {

        std::vector<double> r1(nherm2(Bra+Ket));
        {
          double a = exp(A,ka);
          double b = exp(B,kb);
          double c = exp(C,kc);
          double d = exp(D,kd);
          auto P = center_of_charge(a, center(A), b, center(B));
          auto Q = center_of_charge(c, center(C), d, center(D));
          double p = a+b;
          double q = c+d;
          double pq = p*q;
          double K = 1;
          K *= 1/sqrt(pq*pq*(p+q));
          K *= primitive(A,ka).C*primitive(B,kb).C;
          K *= primitive(C,kc).C*primitive(D,kd).C;
          K *= std::exp(-(a*b)/p*norm(AB));
          K *= std::exp(-(c*d)/q*norm(CD));
          //K *= 2*std::pow(M_PI,2.5);
          double alpha = (p*q)/(p+q);
          compute_r1(Bra+Ket, alpha, P-Q, r1);
          for (size_t i = 0; i < r1.size(); ++i) {
            r1[i] *= K;
            //printf("ref: r[%i]=%f\n", i, r1[i]);
          }
        }

        for (auto d : cartesian::orbitals(D.L)) {
          for (auto c : cartesian::orbitals(C.L)) {
            for (int iq = 0; iq < nherm2(Ket); ++iq) {
              auto q =  reference::orbital(iq);
              double e = 1;
              for (int i = 0; i < 3; ++i) {
                e *= E(c[i], d[i], q[i], exp(C,kc), exp(D,kd), CD[i]);
              }
              for (int ip = 0; ip < nherm2(Bra); ++ip) {
                auto p = reference::orbital(ip);
                auto q = reference::orbital(iq);
                double phase = (q.L()%2 == 0 ? +1 : -1);
                double r = r1[hermite::index2(p+q)];
                //printf("ref: r=%f\n", K*r);
                pCD(ip,index(c),index(d)) += e*r*phase;
              }
            }
          } // c
        } // d

      } // kc
    } // kd

  }

  void compute(const auto &A, const auto &B, const auto &C, const auto &D, auto &ABCD) {

    int Bra = A.L + B.L;
    int Ket = C.L + D.L;

    auto AB = center(A) - center(B);
    auto CD = center(C) - center(D);

    for (int kd = 0; kd < D.K; ++kd) {
      for (int kc = 0; kc < C.K; ++kc) {
        for (int kb = 0; kb < B.K; ++kb) {
          for (int ka = 0; ka < A.K; ++ka) {

            std::vector<double> r1(nherm2(Bra+Ket));
            {
              double a = exp(A,ka);
              double b = exp(B,kb);
              double c = exp(C,kc);
              double d = exp(D,kd);
              auto P = center_of_charge(a, center(A), b, center(B));
              auto Q = center_of_charge(c, center(C), d, center(D));
              double p = a+b;
              double q = c+d;
              double pq = p*q;
              double K = 1;
              K *= 1/sqrt(pq*pq*(p+q));
              K *= primitive(A,ka).C*primitive(B,kb).C;
              K *= primitive(C,kc).C*primitive(D,kd).C;
              K *= std::exp(-(a*b)/p*norm(AB));
              K *= std::exp(-(c*d)/q*norm(CD));
              //K *= 2*std::pow(M_PI,2.5);
              double alpha = (p*q)/(p+q);
              compute_r1(Bra+Ket, alpha, P-Q, r1);
              for (size_t i = 0; i < r1.size(); ++i) {
                r1[i] *= K;
                //printf("ref: r[%i]=%f\n", i, r1[i]);
              }
            }

            for (auto d : cartesian::orbitals(D.L)) {
              for (auto c : cartesian::orbitals(C.L)) {
                for (auto b : cartesian::orbitals(B.L)) {
                  for (auto a : cartesian::orbitals(A.L)) {
                    double v = 0;
                    for (int iq = 0; iq < nherm2(Ket); ++iq) {
                      auto q =  reference::orbital(iq);
                      double Eq = 1;
                      for (int ix = 0; ix < 3; ++ix) {
                        Eq *= E(c[ix], d[ix], q[ix], exp(C,kc), exp(D,kd), CD[ix]);
                      }
                      double phase = (q.L()%2 == 0 ? +1 : -1);
                      for (int ip = 0; ip < nherm2(Bra); ++ip) {
                        auto p = reference::orbital(ip);
                        double Ep = 1;
                        for (int ix = 0; ix < 3; ++ix) {
                          Ep *= E(a[ix], b[ix], p[ix], exp(A,ka), exp(B,kb), AB[ix]);
                        }
                        double r = r1[hermite::index2(p+q)];
                        v += Ep*Eq*r*phase;
                      } // p
                    } // q
                    ABCD(index(a),index(b),index(c),index(d)) += sqrt_4_pi5*v;
                  } // a
                } // b
              } // c
            } // d

          } // ka
        }// kb
      } // kc
    } // kd

  }

}

#endif /* LIBINTX_MD_REFERENCE_H */
