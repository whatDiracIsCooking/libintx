#include "libintx/gpu/kengine.h"
#include "libintx/gpu/md/engine.h"
#include "libintx/gpu/api/api.h"
#include "libintx/kengine/md/driver.h"

namespace libintx::gpu::md {

  namespace {

    /// The integral batch on the device path.
    ///
    /// The MD device engine writes its result through a plain `double*`, and
    /// a device write into host memory needs that memory registered -- so the
    /// buffer owns the registration, re-doing it whenever a resize moves the
    /// allocation, and drops it in the destructor. `synchronize` is the stream
    /// wait the digest needs before it reads what the kernel wrote.
    struct DeviceBuffer {

      explicit DeviceBuffer(gpuStream_t stream) : stream_(stream) {}

      ~DeviceBuffer() { unregister(); }

      DeviceBuffer(const DeviceBuffer&) = delete;
      DeviceBuffer& operator=(const DeviceBuffer&) = delete;

      double* resize(size_t n) {
        if (n > data_.capacity()) {
          unregister();
          data_.reserve(n);
        }
        data_.assign(n, 0.0);
        if (registered_ != data_.data()) {
          unregister();
          if (!data_.empty()) {
            gpu::host::register_pointer(data_.data(), data_.capacity());
            registered_ = data_.data();
          }
        }
        return data_.data();
      }

      void synchronize() { gpu::stream::synchronize(stream_); }

    private:
      void unregister() {
        if (!registered_) return;
        gpu::host::unregister_pointer(registered_);
        registered_ = nullptr;
      }

      gpuStream_t stream_;
      std::vector<double> data_;
      double *registered_ = nullptr;
    };

    struct KEngine : libintx::KEngine {

      KEngine(
        const Basis<Gaussian> &basis,
        std::shared_ptr<const Screening> screening,
        gpuStream_t stream)
        : basis_(basis), screening_(screening), stream_(stream)
      {
        auto norm2 = [&](int i, int j) -> float {
          return (screening_ ? screening_->max2(i,j) : 1.0f);
        };
        classes_ = kengine::md::make_pair_classes(basis_, norm2);
      }

      void K(const TileIn &D, const TileOut &K, const AllSum &allsum) override {
        namespace kmd = kengine::md;
        auto d = kmd::gather_density(basis_, D);
        kmd::Matrix k(basis_.nbf());

        IntegralEngine<4> engine(basis_, basis_, stream_);
        DeviceBuffer buffer(stream_);
        kmd::build(
          basis_, classes_, engine, buffer, d, screening_.get(), max_batch, k
        );

        if (allsum) allsum(k.data.data(), k.data.size());
        kmd::scatter_exchange(basis_, k, K);
      }

      /// Doubles in one integral batch. Smaller than the host engine's: this
      /// buffer is pinned host memory the device writes through, and pinning
      /// is a scarcer resource than plain heap.
      size_t max_batch = 1024*1024;

    private:
      Basis<Gaussian> basis_;
      std::shared_ptr<const Screening> screening_;
      gpuStream_t stream_;
      std::vector<kengine::md::PairClass> classes_;

    };

  } // namespace

  std::unique_ptr<libintx::KEngine> make_kengine(
    const Basis<Gaussian> &basis,
    std::shared_ptr<const libintx::KEngine::Screening> screening,
    gpuStream_t stream)
  {
    return std::make_unique<KEngine>(basis, screening, stream);
  }

  std::shared_ptr<const libintx::KEngine::Screening> make_schwarz_screening(
    const Basis<Gaussian> &basis,
    float threshold,
    gpuStream_t stream)
  {
    IntegralEngine<4> engine(basis, basis, stream);
    DeviceBuffer buffer(stream);
    return kengine::md::schwarz_screening(basis, engine, buffer, threshold);
  }

} // libintx::gpu::md

std::unique_ptr<libintx::KEngine> libintx::gpu::make_kengine(
  const Basis<Gaussian> &basis,
  std::shared_ptr<const libintx::KEngine::Screening> screening,
  gpuStream_t stream)
{
  return md::make_kengine(basis, screening, stream);
}

std::shared_ptr<const libintx::KEngine::Screening>
libintx::gpu::make_schwarz_screening(
  const Basis<Gaussian> &basis,
  float threshold,
  gpuStream_t stream)
{
  return md::make_schwarz_screening(basis, threshold, stream);
}
