#ifndef LIBINTX_GPU_KENGINE_MD_BUFFER_H
#define LIBINTX_GPU_KENGINE_MD_BUFFER_H

#include "libintx/gpu/api/api.h"

#include <vector>

namespace libintx::gpu::md {

  /// The integral batch on the device path, the Buffer both device K engines
  /// (integral-direct and density-fitted) hand to their driver.
  ///
  /// The MD device engines write their result through a plain `double*`, and
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

}

#endif /* LIBINTX_GPU_KENGINE_MD_BUFFER_H */
