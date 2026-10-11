#pragma once
#include <sgl_kernel/ffi.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/distributed/ptx.cuh>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/optional.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>
#include <vector>

namespace sglang {

namespace device::distributed {

inline constexpr uint32_t kMaxWorldSize = 16;

struct Counter {
 public:
  Counter(const Counter&) = delete;
  SGL_DEVICE uint32_t get() const {
    return m_counter;
  }
  SGL_DEVICE void set(uint32_t val) {
    m_counter = val;
  }
  SGL_DEVICE uint32_t inc(uint32_t val) {
    return ::atomicAdd(&m_counter, val);
  }

 private:
  uint32_t m_counter;
};



struct alignas(128) Semaphore {
 public:
  Semaphore(const Semaphore&) = delete;
  SGL_DEVICE Counter* counter_ptr() {
    return &m_counter;
  }
  SGL_DEVICE uint32_t get_relaxed() const {
    return ptx::load_relaxed_sys(&m_flag);
  }
  SGL_DEVICE void put_relaxed() {
    ptx::red_add_relaxed_sys(&m_flag, 1);
  }
  SGL_DEVICE void put_relaxed_multicast() {
    ptx::multimem_red_add_relaxed(&m_flag, 1);
  }
  SGL_DEVICE uint32_t get_acquire() const {
    return ptx::load_acquire_sys(&m_flag);
  }
  SGL_DEVICE void put_release() {
    ptx::red_add_release_sys(&m_flag, 1);
  }
  SGL_DEVICE void put_release_multicast() {
    ptx::multimem_red_add_release(&m_flag, 1);
  }

 private:
  uint32_t m_flag;
  Counter m_counter;
};
static_assert(sizeof(Semaphore) == 128, "must match _SEMAPHORE_BYTES in custom_all_reduce_v2.py");




template <uint32_t kWorldSize>
struct PushWorkSpace {
  std::array<uint8_t*, kWorldSize> workspaces;
  Counter* counter;
  uint8_t* mc_workspace;
  uint32_t slot_bytes;
};



template <uint32_t kWorldSize>
struct PullWorkSpace {
  std::array<Semaphore*, kWorldSize> semaphores;
  std::array<uint8_t*, kWorldSize> workspaces;
  Semaphore* mc_semaphore;
  uint8_t* mc_workspace;
};





template <typename T>
struct FloatTrait {};

template <>
struct FloatTrait<bf16_t> {
  using type = uint16_t;
  static constexpr uint16_t kNegZero = 0x8000u;
};

template <>
struct FloatTrait<fp16_t> {
  using type = uint16_t;
  static constexpr uint16_t kNegZero = 0x8000u;
};

template <>
struct FloatTrait<float> {
  using type = uint32_t;
  static constexpr uint32_t kNegZero = 0x80000000u;
};

template <typename T, uint32_t N, uint32_t kAtom = sizeof(T)>
struct LamportTrait {
  static_assert(kAtom >= sizeof(T) && (kAtom == 2 || kAtom == 4 || kAtom == 8));
  static_assert(kAtom % sizeof(T) == 0 && N % (kAtom / sizeof(T)) == 0);
  using Packed = std::conditional_t<kAtom == 2, uint16_t, std::conditional_t<kAtom == 4, uint32_t, uint64_t>>;
  static constexpr Packed kNegZero = FloatTrait<T>::kNegZero;
  static constexpr uint32_t kNumPacked = N / (kAtom / sizeof(T));

  SGL_DEVICE static void clear_pos_zero(void* val) {
    const auto ptr = static_cast<Packed*>(val);
#pragma unroll
    for (uint32_t i = 0; i < kNumPacked; ++i) {
      if (ptr[i] == 0) ptr[i] = kNegZero;
    }
  }

  SGL_DEVICE static bool has_pos_zero(const void* val) {
    const auto ptr = static_cast<const Packed*>(val);
    bool result = false;
#pragma unroll
    for (uint32_t i = 0; i < kNumPacked; ++i) {
      result |= ptr[i] == 0;
    }
    return result;
  }

  SGL_DEVICE static void fill_pos_zero(void* val) {
    const auto ptr = static_cast<Packed*>(val);
#pragma unroll
    for (uint32_t i = 0; i < kNumPacked; ++i) {
      ptr[i] = 0;
    }
  }
};

template <uint32_t kWorldSize>
struct Barrier {
 public:
  SGL_DEVICE Barrier(Semaphore* const* semaphores, uint32_t rank, uint32_t num_arrives)
      : m_counter(0), m_rank(rank), m_semaphores(semaphores) {
    const auto counter = semaphores[rank][blockIdx.x].counter_ptr();
    const auto signal = num_arrives * kWorldSize;
    m_counter = threadIdx.x == rank ? counter->inc(signal) : 0;
  }

  template <bool kNeedFence>
  SGL_DEVICE void arrive(uint32_t n) const {
    if (const auto tx = threadIdx.x; tx < kWorldSize) {
      const auto bx = blockIdx.x;
      const auto semaphore = &m_semaphores[tx][bx];
      const auto current = m_counter + n * kWorldSize;
      if constexpr (kNeedFence) {
        semaphore->put_release();
        if (tx == m_rank) {
          while (semaphore->get_acquire() - current < kWorldSize)
            ;
        }
      } else {
        semaphore->put_relaxed();
        if (tx == m_rank) {
          while (semaphore->get_relaxed() - current < kWorldSize)
            ;
        }
      }
    }
  }

  SGL_DEVICE void arrive_relaxed(uint32_t n) const {
    return this->arrive<false>(n);
  }

  SGL_DEVICE void arrive_rel_acq(uint32_t n) const {
    return this->arrive<true>(n);
  }

 private:
  uint32_t m_counter;
  uint32_t m_rank;
  Semaphore* const* m_semaphores;
};




template <uint32_t kWorldSize>
struct PushEpoch {
 public:
  SGL_DEVICE PushEpoch(Counter* counter, uint8_t* const* workspaces, uint32_t slot_bytes)
      : m_counter(counter), m_workspaces(workspaces), m_slot_bytes(slot_bytes), m_epoch(m_counter[blockIdx.x].get()) {}

  SGL_DEVICE PushEpoch(const PushWorkSpace<kWorldSize>& ws)
      : PushEpoch(ws.counter, ws.workspaces.data(), ws.slot_bytes) {}


  SGL_DEVICE void* slot_ptr(uint32_t dst, uint32_t src = 0) const {
    const auto epoch_stride_bytes = (m_epoch & 1) * m_slot_bytes * kWorldSize;
    return m_workspaces[dst] + src * m_slot_bytes + epoch_stride_bytes;
  }

  SGL_DEVICE uint32_t slot_offset(uint32_t src = 0) const {
    const auto epoch_stride_bytes = (m_epoch & 1) * m_slot_bytes * kWorldSize;
    return src * m_slot_bytes + epoch_stride_bytes;
  }

  SGL_DEVICE void flip() const {
    if (threadIdx.x == 0) m_counter[blockIdx.x].set(m_epoch ^ 1);
  }

  SGL_DEVICE void unsafe_flip_at(uint32_t bx) const {
    m_counter[bx].set(m_epoch ^ 1);
  }

  SGL_DEVICE void unsafe_flip_range(uint32_t start, uint32_t finish) const {
    for (uint32_t idx = start + threadIdx.x; idx < finish; idx += blockDim.x) {
      m_counter[idx].set(m_epoch ^ 1);
    }
  }

 private:
  Counter* m_counter;
  uint8_t* const* m_workspaces;
  uint32_t m_slot_bytes;
  uint32_t m_epoch;
};










struct McBarrier {
 public:
  SGL_DEVICE McBarrier(Semaphore* local, Semaphore* mc, uint32_t world_size, uint32_t num_arrives)
      : m_counter(0), m_world_size(world_size), m_local(local), m_mc(mc) {
    if (threadIdx.x == 0) {
      m_counter = local[blockIdx.x].counter_ptr()->inc(num_arrives * world_size);
    }
  }

  template <bool kNeedFence>
  SGL_DEVICE void arrive(uint32_t n) const {
    return arrive_at<kNeedFence>(m_local, m_mc, m_world_size, m_counter + n * m_world_size);
  }

  SGL_DEVICE void arrive_relaxed(uint32_t n) const {
    return this->arrive<false>(n);
  }

  SGL_DEVICE void arrive_rel_acq(uint32_t n) const {
    return this->arrive<true>(n);
  }




  SGL_DEVICE uint32_t window() const {
    return m_counter;
  }



  template <bool kNeedFence>
  SGL_DEVICE static void arrive_at(Semaphore* local, Semaphore* mc, uint32_t world_size, uint32_t window) {
    if (threadIdx.x != 0) return;
    const auto bx = blockIdx.x;
    const auto semaphore = &local[bx];
    const auto mc_semaphore = &mc[bx];
    if constexpr (kNeedFence) {
      mc_semaphore->put_release_multicast();
      while (semaphore->get_acquire() - window < world_size)
        ;
    } else {
      mc_semaphore->put_relaxed_multicast();
      while (semaphore->get_relaxed() - window < world_size)
        ;
    }
  }

 private:
  uint32_t m_counter;
  uint32_t m_world_size;
  Semaphore* m_local;
  Semaphore* m_mc;
};

}

namespace host::distributed {

using device::distributed::Counter, device::distributed::Semaphore;
using device::distributed::PullWorkSpace, device::distributed::PushWorkSpace;
using TensorView = tvm::ffi::TensorView;
template <typename T>
using Optional = tvm::ffi::Optional<T>;

inline constexpr uint32_t kMaxWorldSize = device::distributed::kMaxWorldSize;



inline uint8_t* offset_mc(uint8_t* base, int64_t offset) {
  return base != nullptr ? base + offset : nullptr;
}



struct BasePlane {
  BasePlane(const BasePlane&) = delete;
  BasePlane(uint32_t rank, uint32_t world_size);
  uint32_t rank;
  uint32_t world_size;
};













struct PushPlaneObj : public tvm::ffi::Object, BasePlane {
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("sgl.distributed.PushPlane", PushPlaneObj, tvm::ffi::Object);


  PushPlaneObj(
      uint32_t rank,
      uint32_t world_size,
      std::vector<TensorView> workspaces,
      TensorView counter,
      intptr_t mc_workspace_ptr);

  uint32_t num_blocks;
  int64_t slot_bytes;
  Counter* counter;
  std::array<uint8_t*, kMaxWorldSize> workspaces;
  uint8_t* mc_workspace;

  template <uint32_t N>
  PushWorkSpace<N> get_workspace(int64_t size, int64_t offset = 0) const {
    CHECK_HOST(N == world_size) << "Plane holds " << world_size << " ranks, asked for " << N;
    CHECK_HOST(size >= 0 && offset >= 0 && offset + size <= slot_bytes)
        << "slice [" << offset << ", " << offset + size << ") escapes the " << slot_bytes << "-byte push slot";


    CHECK_HOST(2 * N * slot_bytes <= std::numeric_limits<uint32_t>::max())
        << 2 * N * slot_bytes << " bytes of push plane exceeds the 32-bit offset range";
    PushWorkSpace<N> ws{{}, counter, offset_mc(mc_workspace, offset), static_cast<uint32_t>(slot_bytes)};
    for (uint32_t i = 0; i < N; ++i) {
      ws.workspaces[i] = workspaces[i] + offset;
    }
    return ws;
  }
};



















struct PullPlaneObj : public tvm::ffi::Object, BasePlane {
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("sgl.distributed.PullPlane", PullPlaneObj, tvm::ffi::Object);


  PullPlaneObj(
      uint32_t rank,
      uint32_t world_size,
      std::vector<TensorView> workspaces,
      std::vector<TensorView> semaphores,
      intptr_t mc_workspace_ptr,
      intptr_t mc_semaphore_ptr);

  uint32_t num_blocks;
  int64_t num_bytes;
  std::array<Semaphore*, kMaxWorldSize> semaphores;
  std::array<uint8_t*, kMaxWorldSize> workspaces;
  Semaphore* mc_semaphore;
  uint8_t* mc_workspace;

  template <uint32_t N>
  PullWorkSpace<N> get_workspace(int64_t size, int64_t offset = 0) const {
    CHECK_HOST(N == world_size) << "Plane holds " << world_size << " ranks, asked for " << N;
    CHECK_HOST(size >= 0 && offset >= 0 && offset + size <= num_bytes)
        << "slice [" << offset << ", " << offset + size << ") escapes the " << num_bytes << "-byte pull workspace";
    PullWorkSpace<N> ws{{}, {}, mc_semaphore, offset_mc(mc_workspace, offset)};
    for (uint32_t i = 0; i < N; ++i) {
      ws.semaphores[i] = semaphores[i];
      ws.workspaces[i] = workspaces[i] + offset;
    }
    return ws;
  }
};

SGLANG_REGISTER_FFI_REFERENCE_CLASS(PushPlaneRef, PushPlaneObj);
SGLANG_REGISTER_FFI_REFERENCE_CLASS(PullPlaneRef, PullPlaneObj);









struct CommunicatorObj : public tvm::ffi::Object {
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("sgl.distributed.Communicator", CommunicatorObj, tvm::ffi::Object);
  static constexpr bool _type_mutable = true;




  CommunicatorObj(Optional<PushPlaneRef> push, Optional<PullPlaneRef> pull);

  uint32_t get_rank() const {
    return m_push.has_value() ? m_push.value()->rank : m_pull.value()->rank;
  }
  uint32_t get_world_size() const {
    return m_push.has_value() ? m_push.value()->world_size : m_pull.value()->world_size;
  }

  bool has_push() const {
    return m_push.has_value();
  }
  bool has_pull() const {
    return m_pull.has_value();
  }


  Optional<PushPlaneRef> get_push() const {
    return m_push;
  }
  Optional<PullPlaneRef> get_pull() const {
    return m_pull;
  }
  const PushPlaneObj& get_push_obj() const {
    CHECK_HOST(m_push.has_value()) << "This communicator has no push plane";
    return *m_push.value().get();
  }
  const PullPlaneObj& get_pull_obj() const {
    CHECK_HOST(m_pull.has_value()) << "This communicator has no pull plane";
    return *m_pull.value().get();
  }




  void set_pull_blocks(std::optional<uint32_t> num_blocks) {
    m_pull_blocks = num_blocks;
  }
  void set_pull_multicast_blocks(std::optional<uint32_t> num_blocks) {
    m_pull_multicast_blocks = num_blocks;
  }
  uint32_t get_pull_blocks() const {
    const auto capacity = get_pull_obj().num_blocks;
    CHECK_HOST(capacity > 0) << "This pull plane has no semaphores to barrier on";
    return m_pull_blocks.has_value() ? std::min(*m_pull_blocks, capacity) : capacity;
  }
  uint32_t get_pull_multicast_blocks() const {
    const auto blocks = get_pull_blocks();
    return m_pull_multicast_blocks.has_value() ? std::min(*m_pull_multicast_blocks, blocks) : blocks;
  }

 private:
  std::optional<uint32_t> m_pull_blocks;
  std::optional<uint32_t> m_pull_multicast_blocks;
  Optional<PushPlaneRef> m_push;
  Optional<PullPlaneRef> m_pull;
};

SGLANG_REGISTER_FFI_REFERENCE_CLASS(CommunicatorRef, CommunicatorObj);

}

}
