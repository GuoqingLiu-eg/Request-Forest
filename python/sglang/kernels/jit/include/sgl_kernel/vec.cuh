







#pragma once
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <cstddef>
#include <cstdint>

namespace sglang {

namespace device {

namespace details {


template <std::size_t N>
struct uint_trait {};

template <>
struct uint_trait<1> {
  using type = uint8_t;
};

template <>
struct uint_trait<2> {
  using type = uint16_t;
};

template <>
struct uint_trait<4> {
  using type = uint32_t;
};

template <>
struct uint_trait<8> {
  using type = uint64_t;
};


template <typename T>
using sized_int = typename uint_trait<sizeof(T)>::type;

}


template <typename T, std::size_t N>
struct alignas(sizeof(T) * N) AlignedStorage {
  T data[N];
};



















template <typename T, std::size_t N>
struct AlignedVector {
 private:
  static_assert(
      (N > 0 && (N & (N - 1)) == 0) && sizeof(T) * N <= kMaxVecBytes,
      "CUDA vector size exceeds arch limit: max 16 bytes on pre-Blackwell/AMD, "
      "32 bytes on Blackwell or greater");
  using element_t = typename details::sized_int<T>;
  using storage_t = AlignedStorage<element_t, N>;

 public:

  SGL_DEVICE void load(const void* ptr, int64_t offset = 0) {
    m_storage = reinterpret_cast<const storage_t*>(ptr)[offset];
  }

  SGL_DEVICE void store(void* ptr, int64_t offset = 0) const {
    reinterpret_cast<storage_t*>(ptr)[offset] = m_storage;
  }

  SGL_DEVICE void fill(T value) {
    const auto store_value = *reinterpret_cast<element_t*>(&value);
#pragma unroll
    for (std::size_t i = 0; i < N; ++i) {
      m_storage.data[i] = store_value;
    }
  }

  SGL_DEVICE auto operator[](std::size_t idx) -> T& {
    return reinterpret_cast<T*>(&m_storage)[idx];
  }
  SGL_DEVICE auto operator[](std::size_t idx) const -> T {
    return reinterpret_cast<const T*>(&m_storage)[idx];
  }
  SGL_DEVICE auto data() -> T* {
    return reinterpret_cast<T*>(&m_storage);
  }
  SGL_DEVICE auto data() const -> const T* {
    return reinterpret_cast<const T*>(&m_storage);
  }

 private:
  storage_t m_storage;
};




template <bool kFP32Acc = true, typename T2, size_t N, size_t M>
SGL_DEVICE auto reduce_vec(device::AlignedVector<T2, N> (&vec)[M]) -> device::AlignedVector<T2, N> {
  static_assert(DTypeTrait<T2>::kVecSize == 2, "reduce_vec only supports 2-element vectors for now");
  static_assert(M > 0, "reduce_vec requires at least one vector to reduce");
  if constexpr (kFP32Acc) {
    fp32x2_t acc[N];
#pragma unroll
    for (size_t i = 0; i < M; ++i) {
#pragma unroll
      for (size_t j = 0; j < N; ++j) {
        const auto [x, y] = cast<fp32x2_t>(vec[i][j]);
        acc[j].x = i == 0 ? x : acc[j].x + x;
        acc[j].y = i == 0 ? y : acc[j].y + y;
      }
    }
    device::AlignedVector<T2, N> out_vec;
#pragma unroll
    for (size_t j = 0; j < N; ++j) {
      out_vec[j] = cast<T2>(acc[j]);
    }
    return out_vec;
  } else {
    using SumOp = ReductionTrait<ReductionOp::SUM, T2>;
    device::AlignedVector<T2, N> out_vec = vec[0];
#pragma unroll
    for (size_t i = 1; i < M; ++i) {
#pragma unroll
      for (size_t j = 0; j < N; ++j) {
        out_vec[j] = SumOp::apply(out_vec[j], vec[i][j]);
      }
    }
    return out_vec;
  }
}

}

}
