#include "libintx/gpu/onebody/basis.h"
#include "libintx/utility.h"

namespace libintx::gpu::md {

  GaussianPairs make_basis(
    const Basis<Gaussian> &A,
    const Basis<Gaussian> &B,
    const std::vector<Index2> &pairs,
    Gaussian2Buffer &ab,
    gpuStream_t stream)
  {

    libintx_assert(!A.empty());
    libintx_assert(!B.empty());
    libintx_assert(!pairs.empty());

    const auto &a0 = A[pairs.front().first];
    const auto &b0 = B[pairs.front().second];
    const int K = a0.K*b0.K;

    std::vector<Gaussian2> h;
    h.reserve(pairs.size());

    for (auto [i,j] : pairs) {
      const auto &a = A[i];
      const auto &b = B[j];
      // A batch is one bin -- angular momentum, the solid-harmonic flag and
      // the contraction degree all have to agree, because the batch is one
      // flat array of K primitive pairs and nothing pads.
      libintx_assert(a.L == a0.L);
      libintx_assert(b.L == b0.L);
      libintx_assert(a.pure == a0.pure);
      libintx_assert(b.pure == b0.pure);
      libintx_assert(a.K*b.K == K);
      h.push_back(Gaussian2{ a, b, { center(a), center(b) } });
    }

    // Same host-to-device pattern as gpu::md::make_basis: pin the staging
    // buffer, copy on the caller's stream, then synchronize before the local
    // vector goes out of scope.
    const size_t bytes = sizeof(Gaussian2)*h.size();
    gpu::host::register_pointer(h.data(), h.size());
    ab.resize(bytes);
    gpu::memcpy(ab.data(), h.data(), bytes, stream);
    gpu::stream::synchronize(stream);
    gpu::host::unregister_pointer(h.data());

    return GaussianPairs {
      .first = a0,
      .second = b0,
      .N = (int)pairs.size(),
      .K = K,
      .data = reinterpret_cast<const Gaussian2*>(ab.data())
    };

  }

  void make_centers(
    const Nuclear::Operator::Parameters &params,
    device::vector<NuclearCenter> &centers,
    gpuStream_t stream)
  {

    libintx_assert(!params.centers.empty());

    std::vector<NuclearCenter> h;
    h.reserve(params.centers.size());
    for (const auto& [Z,r] : params.centers) {
      h.push_back(NuclearCenter{ (double)Z, { r[0], r[1], r[2] } });
    }

    gpu::host::register_pointer(h.data(), h.size());
    centers.resize(h.size());
    gpu::memcpy(centers.data(), h.data(), sizeof(NuclearCenter)*h.size(), stream);
    gpu::stream::synchronize(stream);
    gpu::host::unregister_pointer(h.data());

  }

}
