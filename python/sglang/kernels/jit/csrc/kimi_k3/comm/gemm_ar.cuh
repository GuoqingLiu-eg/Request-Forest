










#include <sgl_kernel/mbarrier.cuh>
#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/distributed/communicator.cuh>

#include <cute/arch/cluster_sm90.hpp>
#include <cutlass/cuda_host_adapter.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>








namespace sglang {

namespace device::ptx {









struct bf16 {
  using packed2_t = uint32_t;
};

template <typename Dst>
static __device__ __forceinline__ typename Dst::packed2_t cvt_pack_f32x2_to(float a, float b);

template <>
__device__ __forceinline__ uint32_t cvt_pack_f32x2_to<bf16>(float a, float b) {
  uint32_t d;
  asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(a), "f"(b));
  return d;
}






static SGL_DEVICE void ldmatrix_x4_b16(uint32_t row_addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(row_addr));
}

static SGL_DEVICE void ldmatrix_x2_b16(uint32_t row_addr, uint32_t& r0, uint32_t& r1) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared::cta.b16 {%0, %1}, [%2];" : "=r"(r0), "=r"(r1) : "r"(row_addr));
}






static __device__ __forceinline__ void
mma_m16n8k16_bf16f32(float4& d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
      : "+f"(d.x), "+f"(d.y), "+f"(d.z), "+f"(d.w)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}




enum class Major : uint8_t {
  K = 0,
  MN = 1,
};

enum class F16Type : uint8_t { F16 = 0, BF16 = 1 };
enum class DType : uint8_t { F16 = 0, F32 = 1, S32 = 2 };








__host__ __device__ static __forceinline__ constexpr uint64_t
mma_smem_desc(uint32_t matrix_addr, uint32_t lbo, uint32_t sbo, uint32_t base_offset, int swizzle_bytes) {
  auto enc = [](uint32_t x) -> uint64_t { return (uint64_t)((x & 0x3FFFFu) >> 4); };
  uint8_t code = (swizzle_bytes == 128) ? 2u : (swizzle_bytes == 64) ? 4u : (swizzle_bytes == 32) ? 6u : 0u;
  uint64_t d = 0;
  d |= enc(matrix_addr);
  d |= enc(lbo) << 16;
  d |= enc(sbo) << 32;
  d |= uint64_t(1u) << 46;
  d |= uint64_t(base_offset & 0x7u) << 49;
  d |= uint64_t(code & 0x7u) << 61;
  return d;
}




template <typename T, int BLOCK_K, int SWIZZLE_BYTES>
__host__ __device__ static __forceinline__ constexpr uint64_t
mma_smem_desc_k_major(uint32_t addr, uint32_t base_offset = 0) {
  constexpr int K_BYTES = BLOCK_K * int(sizeof(T));
  static_assert(SWIZZLE_BYTES == K_BYTES, "K-major requires swizzle bytes == BLOCK_K * sizeof(T)");
  return mma_smem_desc(addr,         0u,         8u * uint32_t(K_BYTES), base_offset, SWIZZLE_BYTES);
}


__host__ __device__ static __forceinline__ constexpr uint32_t mma_inst_desc_f16(
    uint32_t M,
    uint32_t N,
    F16Type a_type = F16Type::BF16,
    F16Type b_type = F16Type::BF16,
    DType d_type = DType::F32,
    Major a_major = Major::K,
    Major b_major = Major::K,
    bool negate_a = false,
    bool negate_b = false) {
  uint32_t d = 0;
  d |= (static_cast<uint32_t>(d_type) & 0x3u) << 4;
  d |= (static_cast<uint32_t>(a_type) & 0x7u) << 7;
  d |= (static_cast<uint32_t>(b_type) & 0x7u) << 10;
  if (negate_a) d |= 1u << 13;
  if (negate_b) d |= 1u << 14;
  d |= (static_cast<uint32_t>(a_major) & 0x1u) << 15;
  d |= (static_cast<uint32_t>(b_major) & 0x1u) << 16;
  d |= ((N >> 3) & 0x3Fu) << 17;
  d |= ((M >> 4) & 0x1Fu) << 24;
  return d;
}











static SGL_DEVICE void mbar_arrive_cluster_release(uint64_t* bar, uint32_t cta_rank) {
  const uint32_t mapped = cute::set_block_rank(to_shared(bar), cta_rank);
  asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0], 1;" ::"r"(mapped));
}





static SGL_DEVICE void cluster_sync_rel_acq() {
  asm volatile("barrier.cluster.arrive.release.aligned;");
  asm volatile("barrier.cluster.wait.acquire.aligned;");
}



static SGL_DEVICE bool elect_one() {
  uint32_t pred;
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "elect.sync _|p, 0xffffffff;\n\t"
      "selp.b32 %0, 1, 0, p;\n\t}\n"
      : "=r"(pred));
  return pred != 0;
}

static SGL_DEVICE uint32_t cluster_cta_rank() {
  uint32_t rank;
  asm("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  return rank;
}














static SGL_DEVICE void tcgen05_alloc(uint32_t smem_addr_for_taddr, uint32_t n_cols) {
  asm volatile(
      "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_addr_for_taddr), "r"(n_cols));
}

static SGL_DEVICE void tcgen05_dealloc(uint32_t taddr, uint32_t n_cols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(n_cols));
}

static SGL_DEVICE void tcgen05_relinquish() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
}




static SGL_DEVICE void tcgen05_ld_32x32b_x8(
    uint32_t taddr,
    uint32_t& r0,
    uint32_t& r1,
    uint32_t& r2,
    uint32_t& r3,
    uint32_t& r4,
    uint32_t& r5,
    uint32_t& r6,
    uint32_t& r7) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
      " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3), "=r"(r4), "=r"(r5), "=r"(r6), "=r"(r7)
      : "r"(taddr));
}

static SGL_DEVICE void tcgen05_ld_32x32b_x8(uint32_t taddr, uint32_t* dst) {
  tcgen05_ld_32x32b_x8(taddr, dst[0], dst[1], dst[2], dst[3], dst[4], dst[5], dst[6], dst[7]);
}






static SGL_DEVICE void tcgen05_wait_ld() {
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}



static SGL_DEVICE void tcgen05_commit_arrive(uint64_t* bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" ::"r"(to_shared(bar)));
}

static SGL_DEVICE void tcgen05_fence_after_thread_sync() {
  asm volatile("tcgen05.fence::after_thread_sync;");
}




static SGL_DEVICE void
tcgen05_mma_f16(uint32_t d, uint64_t desc_a, uint64_t desc_b, uint32_t inst_desc_high, uint32_t scale_c) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n" ::"r"(d),
      "l"(desc_a),
      "l"(desc_b),
      "r"(inst_desc_high),
      "r"(scale_c));
}















static SGL_DEVICE void prefetch_tensormap(const void* tmap) {
  asm volatile("prefetch.tensormap [%0];" ::"l"(tmap) : "memory");
}


static SGL_DEVICE void
cp_async_bulk_tensor_2d_load(uint32_t dst_smem, const CUtensorMap* tmap, int32_t x, int32_t y, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3}], [%4];" ::"r"(dst_smem),
      "l"(tmap),
      "r"(x),
      "r"(y),
      "r"(to_shared(bar))
      : "memory");
}











static SGL_DEVICE void cp_async_bulk_tensor_2d_load_multicast_cg1(
    uint32_t dst_smem,
    const CUtensorMap* tmap,
    int32_t x,
    int32_t y,
    uint64_t* bar,
    uint16_t multicast_mask = 0b11,
    uint64_t cache_hint = 0x0ULL) {
  const uint32_t mbar_addr = to_shared(bar);
  asm volatile(
      "cp.async.bulk.tensor.2d.cta_group::1.shared::cluster.global"
      ".mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
      " [%0], [%1, {%4, %5}], [%2], %3, %6;" ::"r"(dst_smem),
      "l"(tmap),
      "r"(mbar_addr),
      "h"(multicast_mask),
      "r"(x),
      "r"(y),
      "l"(cache_hint)
      : "memory");
}

}

namespace swz {








__host__ __device__ inline uint32_t smem_col_128b_bf16(uint32_t r, uint32_t c) {
  return c ^ ((r & 7u) << 3);
}

}

namespace tmap {













inline CUtensorMap encode_tiled_2d(
    void* global_ptr,
    CUtensorMapDataType dtype,
    uint64_t global_rows,
    uint64_t global_cols,
    uint64_t row_stride_bytes,
    uint32_t box_rows,
    uint32_t box_cols,
    CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE,
    CUtensorMapL2promotion promo = CU_TENSOR_MAP_L2_PROMOTION_NONE) {
  cuuint64_t global_dim[2] = {global_cols, global_rows};
  cuuint64_t global_strides[1] = {row_stride_bytes};
  cuuint32_t box_dim[2] = {box_cols, box_rows};
  cuuint32_t element_strides[2] = {1, 1};

  CUtensorMap m{};
  const CUresult res = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
      &m,
      dtype,
               2,
      global_ptr,
      global_dim,
      global_strides,
      box_dim,
      element_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      swizzle,
      promo,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (res != CUDA_SUCCESS) {
    std::fprintf(
        stderr,
        "gemm_ar: cuTensorMapEncodeTiled failed (%d) at %s:%d — rows=%llu cols=%llu "
        "row_stride=%llu box=%ux%u swizzle=%d\n",
        int(res),
        __FILE__,
        __LINE__,
        (unsigned long long)global_rows,
        (unsigned long long)global_cols,
        (unsigned long long)row_stride_bytes,
        box_rows,
        box_cols,
        int(swizzle));
    std::abort();
  }
  return m;
}

}

namespace dense_gemm_mainloop {







template <int CTA_GROUP, int GROUP_N>
__device__ __forceinline__ int2 group_n_swizzle(int linear, int crank, int cluster_grid_m, int grid_n) {
  if constexpr (CTA_GROUP == 1) {
    const int num_blocks_per_group = cluster_grid_m * GROUP_N;
    const int group_idx = linear / num_blocks_per_group;
    const int first_n = group_idx * GROUP_N;
    const int in_group = linear - group_idx * num_blocks_per_group;
    const int num_n_in_group = grid_n - first_n < GROUP_N ? grid_n - first_n : GROUP_N;
    const int bid_m = in_group / num_n_in_group;
    const int bid_n = first_n + (in_group % num_n_in_group);
    (void)crank;
    return {bid_m, bid_n};
  } else {
    const int c = linear / CTA_GROUP;
    const int r = linear & (CTA_GROUP - 1);
    const int num_blocks_per_group = cluster_grid_m * GROUP_N;
    const int group_idx = c / num_blocks_per_group;
    const int first_n = group_idx * GROUP_N;
    const int in_group = c - group_idx * num_blocks_per_group;
    const int num_n_in_group = grid_n - first_n < GROUP_N ? grid_n - first_n : GROUP_N;
    const int cluster_bid_m = in_group / num_n_in_group;
    const int cluster_bid_n = first_n + (in_group % num_n_in_group);
    const int bid_m = cluster_bid_m * CTA_GROUP + r;
    const int bid_n = cluster_bid_n;
    (void)crank;
    return {bid_m, bid_n};
  }
}

}

namespace oproj_ar {


#ifndef OPROJ_N
#define OPROJ_N 7168
#endif
constexpr int kN = OPROJ_N;
static_assert(
    kN % 256 == 0 && kN >= 256,
    "N must be a multiple of 256 (member tile table: 128-row "
    "strips + BN up to 256); relaxing this needs a BN-table edit");
constexpr int kBK = 64;
constexpr int kBNRows = 48;
constexpr int kTilesMax = 6;
constexpr int kCWarps = 6;
constexpr int kThreads = (kCWarps + 1) * 32;
constexpr int kMMax = 512;
constexpr int kRing = 64;

enum class Comm { kNone, kMc, kPeer, kMcPull, kTwoShot, kTwoShotPeer };











constexpr size_t kSlotBytes1 = size_t(kMMax) * kN * 2;
constexpr __host__ __device__ size_t slot_off(int parity, int src, int R) {
  return (size_t(parity) * R + src) * kSlotBytes1;
}
constexpr __host__ __device__ size_t pull_off(int parity, int R) {
  return (2 * size_t(R) + parity) * kSlotBytes1;
}
constexpr __host__ __device__ size_t flags_off(int R) {
  const size_t end = (2 * size_t(R) + 2) * kSlotBytes1;
  return (end + (size_t(2) << 20)) & ~((size_t(2) << 20) - 1);
}






constexpr int kMaxCTA = 256;
constexpr int kFams = 7;








constexpr __host__ __device__ size_t done_off(int R) {
  return flags_off(R) + size_t(kFams) * 512;
}
constexpr __host__ __device__ size_t region_bytes(int R) {
  return done_off(R) + size_t(kFams) * kRing * kMaxCTA * 4 + 512;
}







SGL_DEVICE uint32_t bf2_u32(float2 f) {
  const __nv_bfloat162 p = __float22bfloat162_rn(f);
  return *reinterpret_cast<const uint32_t*>(&p);
}



template <int R>
struct Params {
  uint8_t* mc_base;
  uint8_t* uc_base[R];
  uint32_t* gather;
  __nv_bfloat16* out;
  const __nv_bfloat16* partial_in;






  uint32_t* epoch_base;
  int my_rank;
  int fam;
};



struct Strip {
  int t0, nt;
  static SGL_DEVICE Strip make(int cta, int ncta) {
    const int kT = kN / 8;
    const int base = kT / ncta, rem = kT % ncta;
    Strip s;
    if (cta < rem) {
      s.nt = base + 1;
      s.t0 = cta * (base + 1);
    } else {
      s.nt = base;
      s.t0 = rem * (base + 1) + (cta - rem) * base;
    }
    return s;
  }
};
















template <int M, int K, int R, Comm COMM, bool GEMM_ON, int S, int CH, int C>
__global__ void __launch_bounds__(kThreads) oproj_ar_kernel(
    const __grid_constant__ CUtensorMap w_map,
    const __grid_constant__ CUtensorMap x_map,
    const __grid_constant__ Params<R> prm) {
  constexpr int Mp = (M + 15) & ~15;
  constexpr int MT = Mp / 16;
  constexpr int KSTEPS = K / (kBK * CH);
  constexpr int kBBytes = kBNRows * kBK * 2, kABytes = Mp * kBK * 2;
  constexpr int kStB = CH * kBBytes, kStA = CH * kABytes;
  static_assert(K % (kBK * CH) == 0);

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const Strip strip = Strip::make(blockIdx.x, gridDim.x);
  __shared__ uint32_t cta_epoch;
  if (tid == 0) cta_epoch = atomicAdd(prm.epoch_base + prm.fam, 1u) / uint32_t(gridDim.x);
  __syncthreads();
  const uint32_t epoch = cta_epoch;
  const int parity = int(epoch & 1);
  const int ring = int(epoch % kRing);

  const size_t foff = flags_off(R) + size_t(prm.fam) * 512;
  const size_t doff2 = done_off(R) + size_t(prm.fam) * kRing * kMaxCTA * 4;
  uint32_t* const flag_local = reinterpret_cast<uint32_t*>(prm.uc_base[prm.my_rank] + foff) + ring;
  uint32_t* const done_local = reinterpret_cast<uint32_t*>(prm.uc_base[prm.my_rank] + doff2) + size_t(blockIdx.x);
  uint32_t* const gather_fam = prm.gather + size_t(prm.fam) * 2 * kRing;
  const uint32_t wrap = epoch / kRing + 1;
  const uint32_t flag_target = wrap * R;
  const uint32_t gath_target = wrap * gridDim.x;

  float4 acc[MT];
#pragma unroll
  for (int i = 0; i < MT; ++i)
    acc[i] = make_float4(0.f, 0.f, 0.f, 0.f);

  extern __shared__ __align__(1024) uint8_t smem[];
  uint8_t* b_st = smem;
  uint8_t* a_st = b_st + size_t(S) * kStB;
  uint64_t* fullb = reinterpret_cast<uint64_t*>(a_st + size_t(S) * kStA);
  uint64_t* emptyb = fullb + S;

  const uint32_t crank = C > 1 ? device::ptx::cluster_cta_rank() : 0;
  {
    if (tid == 0) {
      device::ptx::prefetch_tensormap(&w_map);
      device::ptx::prefetch_tensormap(&x_map);
#pragma unroll
      for (int s = 0; s < S; ++s) {
        device::ptx::mbar_init(fullb + s, 1);
        device::ptx::mbar_init(emptyb + s, crank == 0 ? kCWarps * C : kCWarps);
      }
    }
    __syncthreads();
    if constexpr (C > 1) device::ptx::cluster_sync_rel_acq();

    if (warp == kCWarps) {




      const int phase = ((int(blockIdx.x) / C) * KSTEPS) / (int(gridDim.x) / C);
      if (lane == 0) {
        for (int j = 0; j < KSTEPS; ++j) {
          const int slot = j % S;
          const int jj = (j + phase) % KSTEPS;
          if (j >= S) device::ptx::mbar_wait_parity(emptyb + slot, ((j - S) / S) & 1);
          device::ptx::mbar_arrive_expect_tx(fullb + slot, kStB + kStA);
#pragma unroll
          for (int c = 0; c < CH; ++c) {
            device::ptx::cp_async_bulk_tensor_2d_load(
                device::ptx::to_shared(b_st + size_t(slot) * kStB + c * kBBytes),
                &w_map,
                (jj * CH + c) * kBK,
                strip.t0 * 8,
                fullb + slot);
            if constexpr (C > 1) {
              if (crank == 0)
                device::ptx::cp_async_bulk_tensor_2d_load_multicast_cg1(
                    device::ptx::to_shared(a_st + size_t(slot) * kStA + c * kABytes),
                    &x_map,
                    (jj * CH + c) * kBK,
                    0,
                    fullb + slot,
                    uint16_t((1u << C) - 1));
            } else {
              device::ptx::cp_async_bulk_tensor_2d_load(
                  device::ptx::to_shared(a_st + size_t(slot) * kStA + c * kABytes),
                  &x_map,
                  (jj * CH + c) * kBK,
                  0,
                  fullb + slot);
            }
          }
        }
      }
    } else {



      const int b_row = (lane & 7) + warp * 8;
      const int b_ka = (lane >> 3) & 1;
      const int a_row = lane & 15, a_ka = lane >> 4;
      for (int s = 0; s < KSTEPS; ++s) {
        const int slot = s % S;
        device::ptx::mbar_wait_parity(fullb + slot, (s / S) & 1);
#pragma unroll
        for (int c = 0; c < CH; ++c) {
          const uint32_t b_base = device::ptx::to_shared(b_st + size_t(slot) * kStB + c * kBBytes);
#pragma unroll
          for (int k16 = 0; k16 < kBK / 16; ++k16) {
            uint32_t b0, b1;
            device::ptx::ldmatrix_x2_b16(
                b_base + uint32_t(b_row) * (kBK * 2) + swz::smem_col_128b_bf16(b_row, (k16 * 2 + b_ka) * 8) * 2,
                b0,
                b1);
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
              uint32_t a0, a1, a2, a3;
              device::ptx::ldmatrix_x4_b16(
                  device::ptx::to_shared(
                      a_st + size_t(slot) * kStA + c * kABytes + uint32_t(mt * 16 + a_row) * (kBK * 2) +
                      swz::smem_col_128b_bf16(a_row, (k16 * 2 + a_ka) * 8) * 2),
                  a0,
                  a1,
                  a2,
                  a3);
              device::ptx::mma_m16n8k16_bf16f32(acc[mt], a0, a1, a2, a3, b0, b1);
            }
          }
        }
        if (lane == 0) {
          device::ptx::mbar_arrive(emptyb + slot);
          if constexpr (C > 1)
            if (crank != 0) device::ptx::mbar_arrive_cluster_release(emptyb + slot, 0);
        }
      }
    }
  }


  __syncthreads();
  device::PDLTriggerSecondary<true>();



  device::PDLWaitPrimary<true>();
  {



    if (tid == 0 && epoch >= 2) {
      const uint32_t e2 = epoch - 2;
      const uint32_t tgt = (e2 / kRing + 1) * R;
      while (device::ptx::load_acquire_sys(done_local + size_t(e2 % kRing) * kMaxCTA) < tgt) {
      }
    }
    __syncthreads();
  }
  const int tig = lane & 3, grp = lane >> 2;
  const int n0 = (strip.t0 + warp) * 8;
  const bool own = warp < strip.nt;


  auto slot_tm = [&](uint8_t* base, size_t sb, int t, int m) { return base + sb + (size_t(t) * M + m) * 16; };

  {
    const size_t sb = slot_off(parity, prm.my_rank, R);
    if (own) {
      const int t = strip.t0 + warp;
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int m = mt * 16 + grp + half * 8;
          if (m < M) {
            const uint32_t v =
                half ? bf2_u32(make_float2(acc[mt].z, acc[mt].w)) : bf2_u32(make_float2(acc[mt].x, acc[mt].y));


            const size_t boff = size_t(m) * 16 + tig * 4;
            {
#pragma unroll
              for (int r = 0; r < R; ++r)
                *reinterpret_cast<uint32_t*>(slot_tm(prm.uc_base[r], sb, t, 0) + boff) = v;
            }
          }
        }
      }
    }
  }


  __syncthreads();
  if (tid == 0) {
    const uint32_t old = device::ptx::atomic_add_acq_rel_gpu(gather_fam + ring, 1);
    if (old + 1 == gath_target) {



      device::ptx::fence_release_sys();
      {
#pragma unroll
        for (int r = 0; r < R; ++r)
          device::ptx::red_add_relaxed_sys(reinterpret_cast<uint32_t*>(prm.uc_base[r] + foff) + ring, 1);
      }
    }
    while (device::ptx::load_acquire_sys(flag_local) < flag_target) {
    }
  }
  __syncthreads();


  const int units = strip.nt * M;
  for (int u = tid; u < units; u += kThreads) {
    const int t = u / M, m = u % M, c0 = (strip.t0 + t) * 8;
    const size_t soff = (size_t(strip.t0 + t) * M + m) * 16;
    uint4 res;
    {
      float2 s[4] = {{0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}};
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const uint4 v = *reinterpret_cast<const uint4*>(prm.uc_base[prm.my_rank] + slot_off(parity, r, R) + soff);
        const uint32_t w4[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w4[j]));
          s[j].x += f.x;
          s[j].y += f.y;
        }
      }
      res = make_uint4(bf2_u32(s[0]), bf2_u32(s[1]), bf2_u32(s[2]), bf2_u32(s[3]));
    }
    *reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(prm.out) + (size_t(m) * kN + c0) * 2) = res;
  }






  __syncthreads();
  if (tid == 0) {
    const size_t off = size_t(ring) * kMaxCTA + blockIdx.x;
    {
#pragma unroll
      for (int r = 0; r < R; ++r)
        device::ptx::red_add_relaxed_sys(reinterpret_cast<uint32_t*>(prm.uc_base[r] + doff2) + off, 1);
    }
  }
}

constexpr int kD3BM = 128, kD3BK = 64;
constexpr __host__ __device__ int d3_ns(int bn) {




  const int stage = kD3BM * kD3BK * 2 + bn * kD3BK * 2;
  const int ns = 220 * 1024 / stage;
  return ns > 12 ? 12 : ns;
}
constexpr int kD3GroupN = 16, kD3KPer = kD3BK / 16;
constexpr int kD3Threads = 8 * 32;
constexpr int kD3ABytes = kD3BM * kD3BK * 2;






constexpr __host__ __device__ int d3_bn(int Mp) {
  return Mp <= 128 ? 64 : (Mp == 256 ? 128 : 256);
}



SGL_DEVICE size_t d3_slot_off(int t, int nb, int w, int l, int nblk_per_tile) {
  return ((size_t(t) * nblk_per_tile + nb) * 4u + w) * 32u * 16u + size_t(l) * 16u;
}






template <int M, int K, int R, Comm COMM>
__global__ void __launch_bounds__(kD3Threads) oproj_dense_ar_kernel(
    const __grid_constant__ CUtensorMap x_tmap,
    const __grid_constant__ CUtensorMap w_tmap,
    const __grid_constant__ Params<R> prm) {
  constexpr bool SWAP = (M <= 64);


  constexpr bool k2S = (COMM == Comm::kTwoShot || COMM == Comm::kTwoShotPeer);




  constexpr int Mp = SWAP ? (M < 16 ? 16 : ((M + 15) & ~15)) : (M + kD3BM - 1) / kD3BM * kD3BM;
  constexpr int kD3BN = SWAP ? (Mp <= 32 ? 16 : 32) : d3_bn(Mp);
  constexpr int kD3BBytes = kD3BN * kD3BK * 2;
  constexpr int kGridM = SWAP ? kN / kD3BM : Mp / kD3BM;
  constexpr int kGridN = SWAP ? Mp / kD3BN : kN / kD3BN;
  constexpr int kTiles = kGridM * kGridN;
  constexpr int kIters = K / kD3BK;
  constexpr int kNBlk = kD3BN / 8;

  constexpr int kD3NS = d3_ns(kD3BN);

  constexpr int kTmemCols = 2 * kD3BN <= 32    ? 32
                            : 2 * kD3BN <= 64  ? 64
                            : 2 * kD3BN <= 128 ? 128
                            : 2 * kD3BN <= 256 ? 256
                                               : 512;
  extern __shared__ __align__(1024) uint8_t smem_buf[];
  const uint32_t smem_base = device::ptx::to_shared(smem_buf);
  constexpr uint32_t kSmemAOff = 0, kSmemBOff = kD3NS * kD3ABytes;

  __shared__ __align__(8) uint64_t tma_mbars[12];
  __shared__ __align__(8) uint64_t mma_mbars[12];
  __shared__ __align__(8) uint64_t mainloop_mbars[2];
  __shared__ __align__(8) uint64_t epi_mbars[2];
  __shared__ __align__(4) uint32_t s_taddr[1];
  __shared__ uint32_t cta_epoch;

  const int tid = threadIdx.x, warp_id = tid >> 5, lane_id = tid & 31;
  const int bid = int(blockIdx.x), num_bids = int(gridDim.x);
  if (tid == 0) cta_epoch = atomicAdd(prm.epoch_base + prm.fam, 1u) / uint32_t(gridDim.x);
  __syncthreads();
  const uint32_t epoch = cta_epoch;
  const int parity = int(epoch & 1);
  const int ring = int(epoch % kRing);
  const size_t foff = flags_off(R) + size_t(prm.fam) * 512;
  const size_t doff2 = done_off(R) + size_t(prm.fam) * kRing * kMaxCTA * 4;
  uint32_t* const flag_local = reinterpret_cast<uint32_t*>(prm.uc_base[prm.my_rank] + foff) + ring;
  uint32_t* const done_local = reinterpret_cast<uint32_t*>(prm.uc_base[prm.my_rank] + doff2) + size_t(blockIdx.x);
  uint32_t* const gather_fam = prm.gather + size_t(prm.fam) * 2 * kRing;
  const uint32_t wrap = epoch / kRing + 1;
  const uint32_t flag_target = wrap * R;
  const uint32_t gath_target = wrap * gridDim.x;
  const size_t sb = slot_off(parity, prm.my_rank, R);

  if (warp_id == 0 && device::ptx::elect_one()) {
    for (int i = 0; i < kD3NS; ++i) {
      device::ptx::mbar_init(&tma_mbars[i], 1);
      device::ptx::mbar_init(&mma_mbars[i], 1);
    }
    for (int i = 0; i < 2; ++i) {
      device::ptx::mbar_init(&mainloop_mbars[i], 1);
      device::ptx::mbar_init(&epi_mbars[i], 4 * 32);
    }
  } else if (warp_id == 1) {
    device::ptx::tcgen05_alloc(device::ptx::to_shared(s_taddr), kTmemCols);
  }
  __syncthreads();
  const uint32_t taddr = s_taddr[0];

  constexpr uint32_t i_desc = device::ptx::mma_inst_desc_f16(
      kD3BM,
      kD3BN,
      device::ptx::F16Type::BF16,
      device::ptx::F16Type::BF16,
      device::ptx::DType::F32,
      device::ptx::Major::K,
      device::ptx::Major::K);
  auto tile_mn = [&](int linear) -> int2 {
    return dense_gemm_mainloop::group_n_swizzle<1, kD3GroupN>(linear, 0, kGridM, kGridN);
  };




  if (warp_id == 0 && device::ptx::elect_one()) {
    constexpr int kPrefetch = kIters < kD3NS ? kIters : kD3NS;
    const int2 mn = tile_mn(bid);
#pragma unroll
    for (int k = 0; k < kPrefetch; ++k) {
      device::ptx::mbar_arrive_expect_tx(&tma_mbars[k], kD3ABytes + kD3BBytes);
      device::ptx::cp_async_bulk_tensor_2d_load(
          smem_base + (SWAP ? kSmemAOff + k * kD3ABytes : kSmemBOff + k * kD3BBytes),
          &w_tmap,
          k * kD3BK,
          (SWAP ? mn.x * kD3BM : mn.y * kD3BN),
          &tma_mbars[k]);
    }
  }


  device::PDLWaitPrimary<true>();


  {
    if (tid == 0 && epoch >= 2) {
      const uint32_t e2 = epoch - 2;
      const uint32_t tgt = (e2 / kRing + 1) * R;
      while (device::ptx::load_acquire_sys(done_local + size_t(e2 % kRing) * kMaxCTA) < tgt) {
      }
    }
    __syncthreads();
  }

  if (warp_id == 0 && device::ptx::elect_one()) {

    int stage = 0, mma_phase = 1;
    for (int t = bid; t < kTiles; t += num_bids) {
      const int2 mn = tile_mn(t);
      for (int k = 0; k < kIters; ++k) {
        device::ptx::mbar_wait_parity(&mma_mbars[stage], mma_phase);
        constexpr bool kDropA = false;
        const bool prefetched = (t == bid && k < kD3NS);
        if (!prefetched) device::ptx::mbar_arrive_expect_tx(&tma_mbars[stage], (kDropA ? 0 : kD3ABytes) + kD3BBytes);
        if constexpr (!kDropA) {

          if (!prefetched || !SWAP)
            device::ptx::cp_async_bulk_tensor_2d_load(
                smem_base + kSmemAOff + stage * kD3ABytes,
                SWAP ? &w_tmap : &x_tmap,
                k * kD3BK,
                mn.x * kD3BM,
                &tma_mbars[stage]);
        }

        if (!prefetched || SWAP)
          device::ptx::cp_async_bulk_tensor_2d_load(
              smem_base + kSmemBOff + stage * kD3BBytes,
              SWAP ? &x_tmap : &w_tmap,
              k * kD3BK,
              mn.y * kD3BN,
              &tma_mbars[stage]);
        if (++stage == kD3NS) {
          stage = 0;
          mma_phase ^= 1;
        }
      }
    }
  } else if (warp_id == 1 && device::ptx::elect_one()) {

    int stage = 0, tma_phase = 0, ml_stage = 0, epi_phase = 1;
    for (int t = bid; t < kTiles; t += num_bids) {
      device::ptx::mbar_wait_parity(&epi_mbars[ml_stage], epi_phase);
      const uint32_t tmem_d = taddr + uint32_t(ml_stage) * kD3BN;
      for (int k = 0; k < kIters; ++k) {
        device::ptx::mbar_wait_parity(&tma_mbars[stage], tma_phase);
        device::ptx::tcgen05_fence_after_thread_sync();
        const uint32_t a_smem = smem_base + kSmemAOff + stage * kD3ABytes;
        const uint32_t b_smem = smem_base + kSmemBOff + stage * kD3BBytes;
#pragma unroll
        for (int k2 = 0; k2 < kD3KPer; ++k2) {
          const uint64_t da = device::ptx::mma_smem_desc_k_major<uint16_t, kD3BK, 128>(a_smem + uint32_t(k2) * 32);
          const uint64_t db = device::ptx::mma_smem_desc_k_major<uint16_t, kD3BK, 128>(b_smem + uint32_t(k2) * 32);
          device::ptx::tcgen05_mma_f16(tmem_d, da, db, i_desc, (k == 0 && k2 == 0) ? 0u : 1u);
        }
        device::ptx::tcgen05_commit_arrive(&mma_mbars[stage]);
        if (++stage == kD3NS) {
          stage = 0;
          tma_phase ^= 1;
        }
      }
      device::ptx::tcgen05_commit_arrive(&mainloop_mbars[ml_stage]);
      ml_stage ^= 1;
      if (ml_stage == 0) epi_phase ^= 1;
    }
  } else if (warp_id >= 4) {

    const int epi_warp = warp_id & 3;
    const uint32_t taddr_lane = uint32_t(epi_warp * 32) << 16;
    int ml_stage = 0, ml_phase = 0;
    for (int t = bid; t < kTiles; t += num_bids) {
      const int2 mn = tile_mn(t);
      device::ptx::mbar_wait_parity(&mainloop_mbars[ml_stage], ml_phase);
      device::ptx::tcgen05_fence_after_thread_sync();
      const uint32_t tmem_d_base = taddr + uint32_t(ml_stage) * kD3BN;
      const int row = mn.x * kD3BM + epi_warp * 32 + lane_id;
#pragma unroll 4
      for (int nb = 0; nb < kNBlk; ++nb) {
        const uint32_t taddr_n = tmem_d_base + uint32_t(nb) * 8 + taddr_lane;
        uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
        device::ptx::tcgen05_ld_32x32b_x8(taddr_n, r0, r1, r2, r3, r4, r5, r6, r7);
        device::ptx::tcgen05_wait_ld();
        uint4 v;
        v.x = device::ptx::cvt_pack_f32x2_to<device::ptx::bf16>(__int_as_float(r1), __int_as_float(r0));
        v.y = device::ptx::cvt_pack_f32x2_to<device::ptx::bf16>(__int_as_float(r3), __int_as_float(r2));
        v.z = device::ptx::cvt_pack_f32x2_to<device::ptx::bf16>(__int_as_float(r5), __int_as_float(r4));
        v.w = device::ptx::cvt_pack_f32x2_to<device::ptx::bf16>(__int_as_float(r7), __int_as_float(r6));
        if (SWAP || row < M) {
          {
            const size_t off = sb + d3_slot_off(t, nb, epi_warp, lane_id, kNBlk);
            if constexpr (k2S)




              *reinterpret_cast<uint4*>(prm.uc_base[t % R] + off) = v;
            else {
#pragma unroll
              for (int r = 0; r < R; ++r)
                *reinterpret_cast<uint4*>(prm.uc_base[r] + off) = v;
            }
          }
        }
      }
      (void)device::ptx::mbar_arrive(&epi_mbars[ml_stage]);
      ml_stage ^= 1;
      if (ml_stage == 0) ml_phase ^= 1;
    }
  }
  __syncthreads();
  device::PDLTriggerSecondary<true>();
  if (warp_id == 1) {
    device::ptx::tcgen05_dealloc(taddr, kTmemCols);
    device::ptx::tcgen05_relinquish();
  }


  if (tid == 0) {
    const uint32_t old = device::ptx::atomic_add_acq_rel_gpu(gather_fam + ring, 1);
    if (old + 1 == gath_target) {
      device::ptx::fence_release_sys();
      {
#pragma unroll
        for (int r = 0; r < R; ++r)
          device::ptx::red_add_relaxed_sys(reinterpret_cast<uint32_t*>(prm.uc_base[r] + foff) + ring, 1);
      }
    }
    while (device::ptx::load_acquire_sys(flag_local) < flag_target) {
    }
  }
  __syncthreads();

  if constexpr (k2S) {




    for (int u = tid + bid * kD3Threads; u < kTiles * kNBlk * 4 * 32; u += num_bids * kD3Threads) {
      const int l = u & 31, w = (u >> 5) & 3, nb = (u >> 7) % kNBlk;
      const int t = u / (kNBlk * 128);
      if (t % R != prm.my_rank) continue;
      const int2 mn = tile_mn(t);
      if (!SWAP && mn.x * kD3BM + w * 32 + l >= M) continue;
      const size_t soff = d3_slot_off(t, nb, w, l, kNBlk);
      float2 acc2[4] = {{0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}};
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const uint4 vv = *reinterpret_cast<const uint4*>(prm.uc_base[prm.my_rank] + slot_off(parity, r, R) + soff);
        const uint32_t w4[4] = {vv.x, vv.y, vv.z, vv.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w4[j]));
          acc2[j].x += f.x;
          acc2[j].y += f.y;
        }
      }
      const uint4 res = make_uint4(bf2_u32(acc2[0]), bf2_u32(acc2[1]), bf2_u32(acc2[2]), bf2_u32(acc2[3]));
      {
#pragma unroll
        for (int r = 0; r < R; ++r)
          *reinterpret_cast<uint4*>(prm.uc_base[r] + pull_off(parity, R) + soff) = res;
      }
    }

    __syncthreads();
    if (tid == 0) {
      uint32_t* const g2 = gather_fam + kRing;
      const uint32_t old = device::ptx::atomic_add_acq_rel_gpu(g2 + ring, 1);
      if (old + 1 == gath_target) {
        device::ptx::fence_release_sys();
        {
#pragma unroll
          for (int r = 0; r < R; ++r)
            device::ptx::red_add_relaxed_sys(reinterpret_cast<uint32_t*>(prm.uc_base[r] + foff + 256) + ring, 1);
        }
      }
      while (device::ptx::load_acquire_sys(reinterpret_cast<uint32_t*>(prm.uc_base[prm.my_rank] + foff + 256) + ring) <
             flag_target) {
      }
    }
    __syncthreads();


    for (int u = tid + bid * kD3Threads; u < kTiles * kNBlk * 4 * 32; u += num_bids * kD3Threads) {
      const int l = u & 31, w = (u >> 5) & 3, nb = (u >> 7) % kNBlk;
      const int t = u / (kNBlk * 128);
      const int2 mn = tile_mn(t);
      const int row = mn.x * kD3BM + w * 32 + l;
      if (!SWAP && row >= M) continue;
      const uint4 res = *reinterpret_cast<const uint4*>(
          prm.uc_base[prm.my_rank] + pull_off(parity, R) + d3_slot_off(t, nb, w, l, kNBlk));
      if constexpr (SWAP) {
        const uint32_t u4[4] = {res.x, res.y, res.z, res.w};
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int mm = mn.y * kD3BN + nb * 8 + j;
          if (mm < M)
            *reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(prm.out) + (size_t(mm) * kN + row) * 2) =
                uint16_t((u4[j >> 1] >> ((j & 1) * 16)) & 0xFFFFu);
        }
      } else {
        *reinterpret_cast<uint4*>(
            reinterpret_cast<uint8_t*>(prm.out) + (size_t(row) * kN + size_t(mn.y) * kD3BN + nb * 8) * 2) = res;
      }
    }
    __syncthreads();
    if (tid == 0) {
      const size_t off = size_t(ring) * kMaxCTA + blockIdx.x;
      {
#pragma unroll
        for (int r = 0; r < R; ++r)
          device::ptx::red_add_relaxed_sys(reinterpret_cast<uint32_t*>(prm.uc_base[r] + doff2) + off, 1);
      }
    }
    return;
  }



  const int units = kTiles * kNBlk * 4 * 32;
  for (int u = tid + bid * kD3Threads; u < units; u += num_bids * kD3Threads) {
    const int l = u & 31, w = (u >> 5) & 3, nb = (u >> 7) % kNBlk;
    const int t = u / (kNBlk * 128);
    const int2 mn = tile_mn(t);
    const int row = mn.x * kD3BM + w * 32 + l;
    if (!SWAP && row >= M) continue;
    const size_t soff = d3_slot_off(t, nb, w, l, kNBlk);
    uint4 res;
    {
      float2 acc2[4] = {{0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}};
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const uint4 vv = *reinterpret_cast<const uint4*>(prm.uc_base[prm.my_rank] + slot_off(parity, r, R) + soff);
        const uint32_t w4[4] = {vv.x, vv.y, vv.z, vv.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w4[j]));
          acc2[j].x += f.x;
          acc2[j].y += f.y;
        }
      }
      res = make_uint4(bf2_u32(acc2[0]), bf2_u32(acc2[1]), bf2_u32(acc2[2]), bf2_u32(acc2[3]));
    }
    if constexpr (SWAP) {
      const uint32_t u4[4] = {res.x, res.y, res.z, res.w};
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int mm = mn.y * kD3BN + nb * 8 + j;
        if (mm < M)
          *reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(prm.out) + (size_t(mm) * kN + row) * 2) =
              uint16_t((u4[j >> 1] >> ((j & 1) * 16)) & 0xFFFFu);
      }
    } else {
      *reinterpret_cast<uint4*>(
          reinterpret_cast<uint8_t*>(prm.out) + (size_t(row) * kN + size_t(mn.y) * kD3BN + nb * 8) * 2) = res;
    }
  }


  __syncthreads();
  if (tid == 0) {
    const size_t off = size_t(ring) * kMaxCTA + blockIdx.x;
    {
#pragma unroll
      for (int r = 0; r < R; ++r)
        device::ptx::red_add_relaxed_sys(reinterpret_cast<uint32_t*>(prm.uc_base[r] + doff2) + off, 1);
    }
  }
}

template <int M, int K, int R, Comm COMM>
struct Launcher3 {
  static constexpr bool kSwap = (M <= 64);
  static constexpr int Mp = kSwap ? (M < 16 ? 16 : ((M + 15) & ~15)) : (M + kD3BM - 1) / kD3BM * kD3BM;
  static constexpr int kBN = kSwap ? (Mp <= 32 ? 16 : 32) : d3_bn(Mp);
  static constexpr int kTiles = kSwap ? (kN / kD3BM) * (Mp / kBN) : (Mp / kD3BM) * (kN / kBN);
  static constexpr int kGrid = kTiles < 152 ? kTiles : 152;
  static constexpr size_t kSmem = size_t(d3_ns(kBN)) * (kD3ABytes + kBN * kD3BK * 2);
  static void set_smem_attr() {
    CHECK_CUDA(cudaFuncSetAttribute(
        oproj_dense_ar_kernel<M, K, R, COMM>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(kSmem)));
    CHECK_CUDA(cudaFuncSetAttribute(
        oproj_dense_ar_kernel<M, K, R, COMM>, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
  }
  static void
  launch(const CUtensorMap& x_tmap, const CUtensorMap& w_tmap, const Params<R>& prm, cudaStream_t stream, bool pdl) {
    cudaLaunchConfig_t cfg{};
    cudaLaunchAttribute attr[1];
    int na = 0;
    if (pdl) {
      attr[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attr[na].val.programmaticStreamSerializationAllowed = 1;
      ++na;
    }
    cfg.gridDim = dim3(unsigned(kGrid));
    cfg.blockDim = dim3(kD3Threads);
    cfg.dynamicSmemBytes = kSmem;
    cfg.stream = stream;
    cfg.attrs = attr;
    cfg.numAttrs = unsigned(na);
    CHECK_CUDA(cudaLaunchKernelEx(&cfg, oproj_dense_ar_kernel<M, K, R, COMM>, x_tmap, w_tmap, prm));
  }
};

template <
    int M,
    int K,
    int R,
    Comm COMM,
    bool GEMM_ON,
    int S = ((M + 15) & ~15) <= 16 ? 6 : (((M + 15) & ~15) == 32 ? 4 : 3),
    int CH = 2,
    int C = 1>


struct Launcher {
  static constexpr int Mp = (M + 15) & ~15;
  static constexpr size_t kSmem = GEMM_ON
                                      ? size_t(S) * CH * (kBNRows * kBK * 2 + Mp * kBK * 2) + 2 * S * sizeof(uint64_t)
                                      : 4096;

  static void set_smem_attr() {
    CHECK_CUDA(cudaFuncSetAttribute(
        oproj_ar_kernel<M, K, R, COMM, GEMM_ON, S, CH, C>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(kSmem)));



    CHECK_CUDA(cudaFuncSetAttribute(
        oproj_ar_kernel<M, K, R, COMM, GEMM_ON, S, CH, C>, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
  }





  static void launch(
      const CUtensorMap& w_map,
      const CUtensorMap& x_map,
      const Params<R>& prm,
      int ncta,
      cudaStream_t stream,
      bool pdl) {
    cudaLaunchConfig_t cfg{};
    cudaLaunchAttribute attr[2];
    int na = 0;
    if (pdl) {
      attr[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attr[na].val.programmaticStreamSerializationAllowed = 1;
      ++na;
    }
    if constexpr (C > 1) {
      attr[na].id = cudaLaunchAttributeClusterDimension;
      attr[na].val.clusterDim.x = C;
      attr[na].val.clusterDim.y = 1;
      attr[na].val.clusterDim.z = 1;
      ++na;
    }
    cfg.gridDim = dim3(unsigned(ncta));
    cfg.blockDim = dim3(kThreads);
    cfg.dynamicSmemBytes = kSmem;
    cfg.stream = stream;
    cfg.attrs = attr;
    cfg.numAttrs = unsigned(na);
    CHECK_CUDA(cudaLaunchKernelEx(&cfg, oproj_ar_kernel<M, K, R, COMM, GEMM_ON, S, CH, C>, w_map, x_map, prm));
  }
};

}

}



#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <array>
#include <mutex>
#include <unordered_map>

namespace sglang {

namespace oproj_ar_ffi {

using namespace oproj_ar;
using tvm::ffi::TensorView;

constexpr int kCellList[7] = {8, 16, 32, 64, 128, 256, 512};

inline int cell_of(int m) {
  for (int c : kCellList)
    if (m <= c) return c;
  return -1;
}

template <int K, int R, bool kUsePDL>
struct GemmArKernel {
  static_assert(R >= 2 && R <= 8, "R outside the validated 2..8 range");
  static_assert(K % 128 == 0 && K >= 128, "K must be a multiple of 128");

  static constexpr int kTwoShotMinM = R >= 8 ? 128 : 256;


  static int64_t fam_of(int64_t m) {
    const int cell = cell_of(int(m));
    CHECK_HOST(cell > 0) << "gemm_ar: M=" << m << " outside [1, " << kMMax << "]";
    for (int i = 0; i < 7; ++i)
      if (kCellList[i] == cell) return i;
    return -1;
  }

  static int64_t cell_of_ffi(int64_t m) {
    return cell_of(int(m));
  }
  static int64_t region_nbytes() {
    return int64_t(region_bytes(R));
  }
  static int64_t gather_words() {
    return int64_t(kFams) * 2 * kRing;
  }
  static int64_t num_fams() {
    return kFams;
  }
  static int64_t max_tokens() {
    return kMMax;
  }


  struct WMaps {
    CUtensorMap w48, w64, w128, w256;
  };

  static const WMaps& w_maps(void* w) {
    static std::unordered_map<void*, WMaps> cache;
    static std::mutex mu;
    std::lock_guard<std::mutex> lk(mu);
    auto it = cache.find(w);
    if (it == cache.end()) {
      auto enc = [&](uint32_t box_rows) {
        return tmap::encode_tiled_2d(
            w, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, kN, K, size_t(K) * 2, box_rows, kBK, CU_TENSOR_MAP_SWIZZLE_128B);
      };
      it = cache.emplace(w, WMaps{enc(kBNRows), enc(64), enc(128), enc(256)}).first;
    }
    return it->second;
  }



  static CUtensorMap x_map(void* x, int m, uint32_t box_rows) {
    return tmap::encode_tiled_2d(
        x, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, uint64_t(m), K, size_t(K) * 2, box_rows, kBK, CU_TENSOR_MAP_SWIZZLE_128B);
  }

  static void set_smem_attrs_once() {
    static bool done = [] {
      Launcher<8, K, R, Comm::kPeer, true>::set_smem_attr();
      Launcher<16, K, R, Comm::kPeer, true>::set_smem_attr();
      Launcher3<32, K, R, Comm::kPeer>::set_smem_attr();
      Launcher3<64, K, R, Comm::kPeer>::set_smem_attr();
      if constexpr (kTwoShotMinM > 128) Launcher3<128, K, R, Comm::kPeer>::set_smem_attr();
      if constexpr (kTwoShotMinM <= 128) Launcher3<128, K, R, Comm::kTwoShotPeer>::set_smem_attr();
      Launcher3<256, K, R, Comm::kTwoShotPeer>::set_smem_attr();
      Launcher3<512, K, R, Comm::kTwoShotPeer>::set_smem_attr();
      return true;
    }();
    (void)done;
  }

  template <int CELL>
  static void enqueue_cell(const WMaps& wm, void* x, int m, const Params<R>& prm, cudaStream_t stream, bool pdl) {
    if constexpr (CELL <= 16) {
      const CUtensorMap xm = x_map(x, m, 16);
      Launcher<CELL, K, R, Comm::kPeer, true>::launch(wm.w48, xm, prm, kM1Grid_(), stream, pdl);
    } else if constexpr (CELL < kTwoShotMinM) {
      const CUtensorMap xm = x_map(x, m, CELL <= 32 ? 16 : (CELL <= 64 ? 32 : 128));
      const CUtensorMap& wmap = CELL <= 64 ? wm.w128 : wm.w64;
      Launcher3<CELL, K, R, Comm::kPeer>::launch(xm, wmap, prm, stream, pdl);
    } else {
      const CUtensorMap xm = x_map(x, m, 128);
      const CUtensorMap& wmap = CELL == 128 ? wm.w64 : (CELL == 256 ? wm.w128 : wm.w256);
      Launcher3<CELL, K, R, Comm::kTwoShotPeer>::launch(xm, wmap, prm, stream, pdl);
    }
  }

  static constexpr int kM1Grid_() {
    return 152;
  }




  static std::array<uint8_t*, R>& bases_store() {
    static std::array<uint8_t*, R> a{};
    return a;
  }

  static void set_bases(TensorView uc_bases) {
    using namespace host;
    TensorMatcher({R}).with_dtype<int64_t>().verify(uc_bases);
    const int64_t* b = static_cast<const int64_t*>(uc_bases.data_ptr());
    for (int r = 0; r < R; ++r)
      bases_store()[r] = reinterpret_cast<uint8_t*>(b[r]);
  }

  static void
  run(TensorView out,
      TensorView x,
      TensorView w,
      TensorView gather,
      TensorView epochs,
      int64_t my_rank) {
    using namespace host;

    auto M = SymbolicSize{"num_tokens"};
    auto CellRows = SymbolicSize{"cell_rows"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, K}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({kN, K}).with_dtype<bf16_t>().with_device(device).verify(w);
    TensorMatcher({CellRows, kN}).with_dtype<bf16_t>().with_device(device).verify(out);

    const int m = int(M.unwrap());
    const int cell = cell_of(m);
    CHECK_HOST(cell > 0) << "gemm_ar: M=" << m << " outside [1, " << kMMax << "]";
    CHECK_HOST(int64_t(CellRows.unwrap()) == cell)
        << "out must have cell(M)=" << cell << " rows, got " << CellRows.unwrap();
    CHECK_HOST(my_rank >= 0 && my_rank < R);
    CHECK_HOST(bases_store()[0] != nullptr) << "gemm_ar: set_bases not called";
    TensorMatcher({int64_t(kFams) * 2 * kRing}).with_dtype<int32_t>().verify(gather);
    TensorMatcher({kFams}).with_dtype<int32_t>().verify(epochs);

    set_smem_attrs_once();

    const DLDevice dev = device.unwrap();
    const auto stream = static_cast<cudaStream_t>(::TVMFFIEnvGetStream(dev.device_type, dev.device_id));

    Params<R> prm{};
    prm.mc_base = nullptr;
    for (int r = 0; r < R; ++r)
      prm.uc_base[r] = bases_store()[r];
    prm.gather = static_cast<uint32_t*>(gather.data_ptr());
    prm.out = static_cast<__nv_bfloat16*>(out.data_ptr());
    prm.partial_in = nullptr;
    prm.epoch_base = static_cast<uint32_t*>(epochs.data_ptr());
    prm.my_rank = int(my_rank);
    prm.fam = int(fam_of(m));

    const bool pdl = kUsePDL;
    const WMaps& wm = w_maps(w.data_ptr());
    switch (cell) {
      case 8:
        enqueue_cell<8>(wm, x.data_ptr(), m, prm, stream, pdl);
        break;
      case 16:
        enqueue_cell<16>(wm, x.data_ptr(), m, prm, stream, pdl);
        break;
      case 32:
        enqueue_cell<32>(wm, x.data_ptr(), m, prm, stream, pdl);
        break;
      case 64:
        enqueue_cell<64>(wm, x.data_ptr(), m, prm, stream, pdl);
        break;
      case 128:
        enqueue_cell<128>(wm, x.data_ptr(), m, prm, stream, pdl);
        break;
      case 256:
        enqueue_cell<256>(wm, x.data_ptr(), m, prm, stream, pdl);
        break;
      default:
        enqueue_cell<512>(wm, x.data_ptr(), m, prm, stream, pdl);
        break;
    }
    CHECK_CUDA(cudaGetLastError()) << "gemm_ar launch (cell=" << cell << ")";
  }
};

}

using oproj_ar_ffi::GemmArKernel;

}
