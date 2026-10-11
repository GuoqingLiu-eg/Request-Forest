




































#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <map>
#include <type_traits>
#include <vector>



namespace ptx {

template <typename T>
static __device__ __forceinline__ uint32_t to_shared(T* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}


static __device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(to_shared(bar)), "r"(count));
}
static __device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" ::"r"(to_shared(bar)), "r"(bytes));
}
static __device__ __forceinline__ void mbar_wait_parity(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "WAIT_%=: mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n\t"
      "@!p bra WAIT_%=;\n\t}\n" ::"r"(to_shared(bar)),
      "r"(parity));
}


static __device__ __forceinline__ void cp_async_16(void* smem, const void* gmem) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(to_shared(smem)), "l"(gmem));
}

static __device__ __forceinline__ void cp_async_16_zfill(void* smem, const void* gmem, int ignore_src) {
  asm volatile(
      "{\n\t.reg .pred pz;\n\t"
      "setp.ne.b32 pz, %2, 0;\n\t"
      "cp.async.cg.shared.global [%0], [%1], 16, pz;\n\t}" ::"r"(to_shared(smem)),
      "l"(gmem),
      "r"(ignore_src));
}
static __device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;");
}
static __device__ __forceinline__ void cp_async_wait_pending(int pending) {
  switch (pending) {
    case 0:
      asm volatile("cp.async.wait_group 0;");
      break;
    default:
      asm volatile("cp.async.wait_group 1;");
      break;
  }
}


static __device__ __forceinline__ void prefetch_tensormap(const void* tmap) {
  asm volatile("prefetch.tensormap [%0];" ::"l"(tmap) : "memory");
}
static __device__ __forceinline__ void
cp_async_bulk_tensor_2d_load(uint32_t dst_smem, const CUtensorMap* tmap, int32_t x, int32_t y, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::"
      "complete_tx::bytes"
      " [%0], [%1, {%2, %3}], [%4];" ::"r"(dst_smem),
      "l"(tmap),
      "r"(x),
      "r"(y),
      "r"(to_shared(bar))
      : "memory");
}


static __device__ __forceinline__ void
ldmatrix_x4_b16(uint32_t row_addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(row_addr));
}
static __device__ __forceinline__ void ldmatrix_x2_b16(uint32_t row_addr, uint32_t& r0, uint32_t& r1) {
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


static __device__ __forceinline__ void red_add_rel_b32(uint32_t* ptr, uint32_t value) {
  asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(ptr), "r"(value));
}
static __device__ __forceinline__ uint32_t ld_acq_b32(const uint32_t* ptr) {
  uint32_t ret;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(ret) : "l"(ptr));
  return ret;
}


static __device__ __forceinline__ void fence_async_global() {
  asm volatile("fence.proxy.async.global;");
}


static __device__ __forceinline__ void tcgen05_alloc(uint32_t smem_addr_for_taddr, uint32_t n_cols) {
  asm volatile(
      "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_addr_for_taddr), "r"(n_cols));
}
static __device__ __forceinline__ void tcgen05_dealloc(uint32_t taddr, uint32_t n_cols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(n_cols));
}
static __device__ __forceinline__ void tcgen05_relinquish() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
}
static __device__ __forceinline__ void
tcgen05_st_32x32b_x4(uint32_t taddr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1, %2, %3, %4};" ::"r"(taddr), "r"(r0), "r"(r1), "r"(r2), "r"(r3));
}
static __device__ __forceinline__ void tcgen05_st_32x32b_x8(
    uint32_t taddr,
    uint32_t r0,
    uint32_t r1,
    uint32_t r2,
    uint32_t r3,
    uint32_t r4,
    uint32_t r5,
    uint32_t r6,
    uint32_t r7) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], "
      " {%1, %2, %3, %4, %5, %6, %7, %8};" ::"r"(taddr),
      "r"(r0),
      "r"(r1),
      "r"(r2),
      "r"(r3),
      "r"(r4),
      "r"(r5),
      "r"(r6),
      "r"(r7));
}
static __device__ __forceinline__ void tcgen05_st_32x32b_x16(uint32_t taddr, const uint32_t* r) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x16.b32 [%16], "
      " {%0, %1, %2, %3, %4, %5, %6, %7,"
      "  %8, %9, %10, %11, %12, %13, %14, %15};" ::"r"(r[0]),
      "r"(r[1]),
      "r"(r[2]),
      "r"(r[3]),
      "r"(r[4]),
      "r"(r[5]),
      "r"(r[6]),
      "r"(r[7]),
      "r"(r[8]),
      "r"(r[9]),
      "r"(r[10]),
      "r"(r[11]),
      "r"(r[12]),
      "r"(r[13]),
      "r"(r[14]),
      "r"(r[15]),
      "r"(taddr));
}
static __device__ __forceinline__ void tcgen05_ld_32x32b_x8(
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
static __device__ __forceinline__ void tcgen05_ld_32x32b_x16(
    uint32_t taddr,
    uint32_t& r0,
    uint32_t& r1,
    uint32_t& r2,
    uint32_t& r3,
    uint32_t& r4,
    uint32_t& r5,
    uint32_t& r6,
    uint32_t& r7,
    uint32_t& r8,
    uint32_t& r9,
    uint32_t& r10,
    uint32_t& r11,
    uint32_t& r12,
    uint32_t& r13,
    uint32_t& r14,
    uint32_t& r15) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
      " {%0, %1, %2, %3, %4, %5, %6, %7,"
      "  %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
      : "=r"(r0),
        "=r"(r1),
        "=r"(r2),
        "=r"(r3),
        "=r"(r4),
        "=r"(r5),
        "=r"(r6),
        "=r"(r7),
        "=r"(r8),
        "=r"(r9),
        "=r"(r10),
        "=r"(r11),
        "=r"(r12),
        "=r"(r13),
        "=r"(r14),
        "=r"(r15)
      : "r"(taddr));
}
static __device__ __forceinline__ void tcgen05_ld_32x32b_x32(uint32_t taddr, uint32_t* r) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
      " {%0, %1, %2, %3, %4, %5, %6, %7,"
      "  %8, %9, %10, %11, %12, %13, %14, %15,"
      "  %16, %17, %18, %19, %20, %21, %22, %23,"
      "  %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
      : "=r"(r[0]),
        "=r"(r[1]),
        "=r"(r[2]),
        "=r"(r[3]),
        "=r"(r[4]),
        "=r"(r[5]),
        "=r"(r[6]),
        "=r"(r[7]),
        "=r"(r[8]),
        "=r"(r[9]),
        "=r"(r[10]),
        "=r"(r[11]),
        "=r"(r[12]),
        "=r"(r[13]),
        "=r"(r[14]),
        "=r"(r[15]),
        "=r"(r[16]),
        "=r"(r[17]),
        "=r"(r[18]),
        "=r"(r[19]),
        "=r"(r[20]),
        "=r"(r[21]),
        "=r"(r[22]),
        "=r"(r[23]),
        "=r"(r[24]),
        "=r"(r[25]),
        "=r"(r[26]),
        "=r"(r[27]),
        "=r"(r[28]),
        "=r"(r[29]),
        "=r"(r[30]),
        "=r"(r[31])
      : "r"(taddr));
}
static __device__ __forceinline__ void tcgen05_wait_ld() {
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}
static __device__ __forceinline__ void tcgen05_wait_st() {
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}
static __device__ __forceinline__ void tcgen05_commit_arrive(uint64_t* bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" ::"r"(to_shared(bar)));
}
static __device__ __forceinline__ void tcgen05_fence_before_thread_sync() {
  asm volatile("tcgen05.fence::before_thread_sync;");
}
static __device__ __forceinline__ void tcgen05_fence_after_thread_sync() {
  asm volatile("tcgen05.fence::after_thread_sync;");
}
static __device__ __forceinline__ void
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

static __device__ __forceinline__ void
tcgen05_mma_f16_atmem(uint32_t d, uint32_t a_tmem, uint64_t desc_b, uint32_t inst_desc_high, uint32_t scale_c) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n\t}\n" ::"r"(d),
      "r"(a_tmem),
      "l"(desc_b),
      "r"(inst_desc_high),
      "r"(scale_c));
}


enum class Major : uint8_t { K = 0, MN = 1 };
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



template <typename T, int BLOCK_K, int BLOCK_MN, int SWIZZLE_BYTES>
__host__ __device__ static __forceinline__ constexpr uint64_t
mma_smem_desc_mn_major(uint32_t addr, uint32_t base_offset = 0) {
  constexpr int BLOCK_MN_BYTES = BLOCK_MN * int(sizeof(T));
  static_assert(
      SWIZZLE_BYTES == 32 || SWIZZLE_BYTES == 64 || SWIZZLE_BYTES == 128, "MN-major requires SWZ in {32, 64, 128}");
  static_assert(BLOCK_MN_BYTES % SWIZZLE_BYTES == 0, "MN-major: BLOCK_MN * sizeof(T) must tile the swizzle atom");
  return mma_smem_desc(
      addr,
              uint32_t(BLOCK_K * SWIZZLE_BYTES),
              uint32_t(8 * SWIZZLE_BYTES),
      base_offset,
      SWIZZLE_BYTES);
}
__host__ __device__ static __forceinline__ constexpr uint32_t mma_inst_desc_f16(
    uint32_t M,
    uint32_t N,
    F16Type a_type = F16Type::BF16,
    F16Type b_type = F16Type::BF16,
    DType d_type = DType::F32,
    Major a_major = Major::K,
    Major b_major = Major::K) {
  uint32_t d = 0;
  d |= (static_cast<uint32_t>(d_type) & 0x3u) << 4;
  d |= (static_cast<uint32_t>(a_type) & 0x7u) << 7;
  d |= (static_cast<uint32_t>(b_type) & 0x7u) << 10;
  d |= (static_cast<uint32_t>(a_major) & 0x1u) << 15;
  d |= (static_cast<uint32_t>(b_major) & 0x1u) << 16;
  d |= ((N >> 3) & 0x3Fu) << 17;
  d |= ((M >> 4) & 0x1Fu) << 24;
  return d;
}
__host__ __device__ static __forceinline__ constexpr uint32_t mma_inst_desc_tf32(
    uint32_t M, uint32_t N, DType d_type = DType::F32, Major a_major = Major::K, Major b_major = Major::K) {
  constexpr uint32_t TF32 = 2u;
  uint32_t d = 0;
  d |= (static_cast<uint32_t>(d_type) & 0x3u) << 4;
  d |= TF32 << 7;
  d |= TF32 << 10;
  d |= (static_cast<uint32_t>(a_major) & 0x1u) << 15;
  d |= (static_cast<uint32_t>(b_major) & 0x1u) << 16;
  d |= ((N >> 3) & 0x3Fu) << 17;
  d |= ((M >> 4) & 0x1Fu) << 24;
  return d;
}

enum class MmaDenseKind : uint8_t { F16 = 0 };
template <MmaDenseKind KIND>
__host__ __device__ static __forceinline__ constexpr uint32_t mma_inst_desc_dense(
    uint32_t M,
    uint32_t N,
    F16Type a_type,
    F16Type b_type,
    DType d_type = DType::F32,
    Major a_major = Major::K,
    Major b_major = Major::K) {
  static_assert(KIND == MmaDenseKind::F16);
  return mma_inst_desc_f16(M, N, a_type, b_type, d_type, a_major, b_major);
}

}

namespace kda {

constexpr int BT = 64;
constexpr int K = 128;
using bf16 = __nv_bfloat16;



__device__ static inline int piece_c0(int p, int nc, int NP) {
  const int base = nc / NP, rem = nc % NP;
  return base * p + (p < rem ? p : rem);
}




static __device__ __forceinline__ void
mma_tf32_atmem(uint32_t d, uint32_t a_tmem, uint64_t desc_b, uint32_t inst_desc_high, uint32_t scale_c) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::tf32 [%0], [%1], %2, %3, p;\n\t}\n" ::"r"(d),
      "r"(a_tmem),
      "l"(desc_b),
      "r"(inst_desc_high),
      "r"(scale_c));
}








static __device__ __forceinline__ void widen_map_tf32(float* slot, const bf16* pad, int vc, int ch) {
  uint32_t w[16];
#pragma unroll
  for (int g = 0; g < 4; ++g) {
    const int in0 = ch * 32 + g * 8;
    reinterpret_cast<uint4*>(w)[g] =
        *reinterpret_cast<const uint4*>(pad + (in0 >> 6) * (K * 64) + vc * 64 + ((((in0 >> 3) & 7) ^ (vc & 7)) << 3));
  }
  __syncthreads();
#pragma unroll
  for (int g = 0; g < 8; ++g) {
    const int in0 = ch * 32 + g * 4;
    const float2 f0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w[2 * g]));
    const float2 f1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w[2 * g + 1]));
    const int half = ((in0 >> 2) & 1) ^ ((vc >> 2) & 1);
    *reinterpret_cast<float4*>(slot + (in0 >> 3) * 1024 + vc * 8 + half * 4) = float4{f0.x, f0.y, f1.x, f1.y};
  }
}


struct K1Smem {
  bf16 sgb[2][BT][K];

  float sgc[BT][K];
  float sA[BT][BT];
  float sU[BT][2 * K + 4];
  float sb[BT];
  union SPool {
    struct {
      bf16 kw[BT][K + 8];
      bf16 qw[BT][K + 8];
      bf16 kz[10][16][K + 8];
    } a;
    struct {
      __align__(1024) bf16 kdT[K][64];
      __align__(1024) bf16 pTt[K][64];
      __align__(1024) bf16 u0Th[K][64];
      __align__(1024) bf16 u0Tl[K][64];
    } b;
    struct {
      __align__(1024) float sLt[16][128][8];
    } g;

    struct {
      bf16 aC_h[BT][56];
      bf16 aC_l[BT][56];
      bf16 upT_h[2 * K][56];
      bf16 upT_l[2 * K][56];
    } f;
  } sp;
  uint64_t mb_f;
  uint32_t s_taddr;
};
struct ChainSmem {
  __align__(1024) bf16 sPneg[2][2][64][64];
  __align__(1024) bf16 sKdT[2][128][64];
  __align__(1024) bf16 sQd[2][2][64][64];
  __align__(1024) bf16 sAh[2][64][64];
  __align__(1024) bf16 sAl[2][64][64];


  __align__(1024) float sU0[2][BT][K];

  uint64_t mb_tma, mb_p1, mb_p2, mb_o, mb_pre, mb_preC, mb_cmL, mb_cmC;
  uint32_t s_taddr;
};










constexpr float L2_EPS = 1e-6f;








template <bool VL>
static __device__ __forceinline__ void row_rnorm(
    float2& dst, const bf16* __restrict__ q, const bf16* __restrict__ kk, size_t gbase, int H, int C_act, int tid) {
  const int r = tid >> 3;
  float sk = 0.f, sq = 0.f;
  if (!VL || r < C_act) {
    const size_t o = gbase + (size_t)r * H * K + (tid & 7) * 16;
    const uint4 xb[4] = {
        *reinterpret_cast<const uint4*>(kk + o),
        *reinterpret_cast<const uint4*>(kk + o + 8),
        *reinterpret_cast<const uint4*>(q + o),
        *reinterpret_cast<const uint4*>(q + o + 8)};
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(xb);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const float2 kf = __bfloat1622float2(x2[e]);
      const float2 qf = __bfloat1622float2(x2[e + 8]);
      sk += kf.x * kf.x + kf.y * kf.y;
      sq += qf.x * qf.x + qf.y * qf.y;
    }
  }
#pragma unroll
  for (int m = 1; m < 8; m <<= 1) {
    sk += __shfl_xor_sync(0xffffffffu, sk, m);
    sq += __shfl_xor_sync(0xffffffffu, sq, m);
  }
  if ((tid & 7) == 0) dst = float2{__frcp_rn(__fsqrt_rn(sk + L2_EPS)), __frcp_rn(__fsqrt_rn(sq + L2_EPS))};
}

static __device__ __forceinline__ __nv_bfloat162 l2_bf16(float2 x, float rst) {
  return __floats2bfloat162_rn(x.x * rst, x.y * rst);
}
static __device__ __forceinline__ float2 l2_round(float2 x, float rst) {
  return __bfloat1622float2(l2_bf16(x, rst));
}



template <bool BSIG>
static __device__ __forceinline__ float beta_in(float b) {
  return BSIG ? __bfloat162float(__float2bfloat16(1.f / (1.f + __expf(-b)))) : b;
}
















template <bool VL = false, int GM = 0, bool RAW = false, bool BSIG = false>
__device__ static void k1_body(
    K1Smem& S,
    int job0,
    int njobs,
    const bf16* __restrict__ q,
    const bf16* __restrict__ kk,
    const bf16* __restrict__ v,
    const bf16* __restrict__ glog,
    const float* __restrict__ beta,
    int T,
    int H,
    float scale,
    bf16* __restrict__ P,
    float* __restrict__ u0,
    bf16* __restrict__ kdec,
    bf16* __restrict__ qdec,
    bf16* __restrict__ aqk_h,
    bf16* __restrict__ aqk_l,
    float* __restrict__ gC,
    bf16* __restrict__ pieceL,
    float* __restrict__ piecec,
    int pidx,


    int nc_tot = 0,
    int tokoff = 0,
    int tend = 0,

    const float* __restrict__ a_log = nullptr,
    const float* __restrict__ dtb = nullptr,
    float lb = 0.f,



    const CUtensorMap* ptm = nullptr) {
  constexpr int SB = 16, NSB = BT / SB;
  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  auto& sA = S.sA;
  auto& sU = S.sU;
  auto& sb = S.sb;
  auto& sp = S.sp;
  auto& kw = sp.a.kw;
  auto& qw = sp.a.qw;
  auto& kz = sp.a.kz;
  auto& mb_f = S.mb_f;
  auto& s_taddr = S.s_taddr;



  auto rn2 = [&](int i) -> float2& { return *reinterpret_cast<float2*>(&sU[i][2 * K]); };
  if (pieceL) {

    if (tid == 0) ptx::mbar_init(&mb_f, 1);
    if (warp == 0) ptx::tcgen05_alloc(ptx::to_shared(&s_taddr), 512);
  }
  int mbph = 0;
  bool pend = false;
  float bnext = 0.f;


  auto pend_wait = [&] {
    if (pend) {
      ptx::mbar_wait_parity(&mb_f, mbph & 1);
      ++mbph;
      pend = false;
    }
  };
  const int nc1 = VL ? nc_tot : T / BT;
  const int tse = VL ? tend : T;

  const float ga = GM != 0 ? expf(a_log[job0 / nc1]) : 0.f;






  auto gate_xform = [&](bf16(&gt)[BT][K], int rows) {
    const float* dtr = dtb + (size_t)(job0 / nc1) * K;
    for (int p = tid * 8; p < BT * K; p += blockDim.x * 8) {
      if (VL && p / K >= rows) continue;
      uint4* gp = reinterpret_cast<uint4*>(&gt[p / K][p % K]);
      uint4 g4 = *gp;
      __nv_bfloat162* g2 = reinterpret_cast<__nv_bfloat162*>(&g4);
      const float4 d0 = *reinterpret_cast<const float4*>(&dtr[p % K]);
      const float4 d1 = *reinterpret_cast<const float4*>(&dtr[p % K + 4]);
      const float dv[8] = {d0.x, d0.y, d0.z, d0.w, d1.x, d1.y, d1.z, d1.w};
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 gw = __bfloat1622float2(g2[e]);
        float y[2];
#pragma unroll
        for (int x = 0; x < 2; ++x) {
          const float g = (x ? gw.y : gw.x) + dv[2 * e + x];
          y[x] =
              GM == 1 ? -ga * (fmaxf(g, 0.f) + __logf(1.f + __expf(-fabsf(g)))) : lb * (1.f / (1.f + __expf(-ga * g)));
        }
        g2[e] = __floats2bfloat162_rn(y[0], y[1]);
      }
      *gp = g4;
    }
  };


  for (int sub = 0; sub < njobs; ++sub) {
    const int job = job0 + sub;
    const int c = job % nc1, h = job / nc1, t0 = c * BT + tokoff;
    const int C_act = VL ? min(BT, tse - t0) : BT;
    bf16(&sgb)[BT][K] = S.sgb[sub & 1];
    float (&sg)[BT][K] = S.sgc;
    if (sub == 0) {
      for (int p = tid; p < BT * K; p += blockDim.x)
        sgb[p / K][p % K] =
            !VL || p / K < C_act ? glog[(size_t)(t0 + p / K) * H * K + h * K + p % K] : __float2bfloat16(0.f);
    } else {
      ptx::cp_async_wait_pending(0);
    }
    if (tid < BT)

      sb[tid] = sub == 0 ? (!VL || tid < C_act ? beta_in<BSIG>(beta[(size_t)(t0 + tid) * H + h]) : 0.f) : bnext;
    if (sub + 1 < njobs) {
      const int jn = job + 1;
      const int t1i = (jn % nc1) * BT + tokoff;
      const size_t t1 = (size_t)t1i;
      const int hn = jn / nc1;
      const int Cn = VL ? min(BT, tse - t1i) : BT;
      bf16* dst = &S.sgb[(sub + 1) & 1][0][0];
      for (int p = tid; p < BT * K / 8; p += blockDim.x) {
        const bf16* src = glog + (t1 + p * 8 / K) * H * K + (size_t)hn * K + (p * 8) % K;
        if constexpr (VL)
          ptx::cp_async_16_zfill(dst + p * 8, src, p * 8 / K >= Cn);
        else
          ptx::cp_async_16(dst + p * 8, src);
      }
      ptx::cp_async_commit();
      if (tid < BT) bnext = !VL || tid < Cn ? beta_in<BSIG>(beta[(t1 + tid) * H + hn]) : 0.f;
    }
    if constexpr (RAW)

      row_rnorm<VL>(rn2(tid >> 3), q, kk, (size_t)t0 * H * K + (size_t)h * K, H, C_act, tid);
    __syncthreads();
    if constexpr (GM != 0)


      if (sub == 0 || !pieceL) {
        gate_xform(sgb, C_act);
        __syncthreads();
      }
    {



      const int col = tid & (K - 1), r0 = (tid >> 7) * (BT / 4);
      float acc = 0.f;
      for (int r = r0; r < r0 + BT / 4; ++r) {
        acc += __bfloat162float(sgb[r][col]);
        sg[r][col] = acc;
      }
    }
    __syncthreads();
    {

      const int colc = tid & (K - 1), rc = tid >> 7;
      for (int seg = 1; seg < 4; ++seg) {
        const float carry = sg[seg * (BT / 4) - 1][colc];
#pragma unroll
        for (int it = 0; it < 4; ++it)
          sg[seg * (BT / 4) + rc + it * 4][colc] += carry;
        __syncthreads();
      }
    }





    const int colp = (tid & (K / 2 - 1)) * 2;
    float a0v[NSB], a1v[NSB], ea0[NSB], ea1[NSB];
    a0v[0] = a1v[0] = 0.f;
    ea0[0] = ea1[0] = 1.f;
#pragma unroll
    for (int s = 1; s < NSB; ++s) {
      a0v[s] = sg[s * SB - 1][colp];
      a1v[s] = sg[s * SB - 1][colp + 1];
      ea0[s] = __expf(a0v[s]);
      ea1[s] = __expf(a1v[s]);
    }

    {


      bf16(*stg)[K + 4] = reinterpret_cast<bf16(*)[K + 4]>(&kz[0][0][0]);


      const float gl0 = sg[BT - 1][colp], gl1 = sg[BT - 1][colp + 1];



      const size_t gp0 = (size_t)(t0 + tid * 2 / K) * H * K + h * K + tid * 2 % K;
      const size_t gst = (size_t)8 * H * K;
      const bf16* pk = &kk[gp0];
      const bf16* pq = &q[gp0];
      const bf16* pv = &v[gp0];
      bf16* qd = &qdec[((size_t)c * H + h) * BT * K + (tid >> 6) * K + colp];



      pend_wait();
#pragma unroll
      for (int hb = 0; hb < 2; ++hb) {
        float2 kv[4], qv[4], vv[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          if (VL && tid * 2 / K + (hb * 4 + j) * 8 >= C_act) {
            kv[j] = qv[j] = vv[j] = float2{0.f, 0.f};
          } else {
            kv[j] = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(pk));
            qv[j] = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(pq));
            vv[j] = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(pv));
            if constexpr (RAW) {
              const float2 r = rn2((tid >> 6) + (hb * 4 + j) * 8);
              kv[j] = l2_round(kv[j], r.x);
              qv[j] = l2_round(qv[j], r.y);
            }
          }
          pk += gst;
          pq += gst;
          pv += gst;
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int i = (tid >> 6) + (hb * 4 + j) * 8;
          const int si = hb * 2 + (j >> 1);
          const float ei0 = __expf(sg[i][colp] - a0v[si]);
          const float ei1 = __expf(sg[i][colp + 1] - a1v[si]);
          const float kw0 = kv[j].x * ei0, kw1 = kv[j].y * ei1;
          *reinterpret_cast<__nv_bfloat162*>(&kw[i][colp]) = __floats2bfloat162_rn(kw0, kw1);
          const float qw0 = qv[j].x * ei0 * scale;
          const float qw1 = qv[j].y * ei1 * scale;
          *reinterpret_cast<__nv_bfloat162*>(&qw[i][colp]) = __floats2bfloat162_rn(qw0, qw1);
          const float bi = sb[i];
          *reinterpret_cast<float2*>(&sU[i][colp]) =
              float2{kw0 * (bi * ea0[si]), kw1 * (bi * ea1[si])};
          *reinterpret_cast<float2*>(&sU[i][K + colp]) = float2{vv[j].x * bi, vv[j].y * bi};
          *reinterpret_cast<__nv_bfloat162*>(qd) = __floats2bfloat162_rn(qw0 * ea0[si], qw1 * ea1[si]);
          qd += 8 * K;
          *reinterpret_cast<__nv_bfloat162*>(&stg[i][colp]) =
              __floats2bfloat162_rn(kv[j].x * __expf(gl0 - sg[i][colp]), kv[j].y * __expf(gl1 - sg[i][colp + 1]));
        }
      }
      __syncthreads();



      const int rr = tid * 4 % BT, cc = tid * 4 / BT;
      bf16* kd = &kdec[((size_t)c * H + h) * BT * K + tid * 4];
#pragma unroll
      for (int n = 0; n < BT * K / 2048; ++n) {
        const int cn = cc + n * 32;
        alignas(8) const __nv_bfloat162 p2[2] = {{stg[rr][cn], stg[rr + 1][cn]}, {stg[rr + 2][cn], stg[rr + 3][cn]}};
        *reinterpret_cast<float2*>(kd + n * 2048) = *reinterpret_cast<const float2*>(p2);
      }
    }
    __syncthreads();





    {
      const int u = tid >> 6;
      float f0[NSB - 1], f1[NSB - 1];
#pragma unroll
      for (int s = 1; s < NSB; ++s) {
        f0[s - 1] = __expf(a0v[s] - a0v[s - 1]);
        f1[s - 1] = __expf(a1v[s] - a1v[s - 1]);
      }
      __nv_bfloat162 kvz[NSB][2];



      const bf16* pz = &kk[(size_t)(t0 + u) * H * K + h * K + colp];
#pragma unroll
      for (int sj = 0; sj < NSB; ++sj)
#pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
          const int j = sj * SB + u + rr * 8;
          kvz[sj][rr] =
              !VL || j < C_act ? *reinterpret_cast<const __nv_bfloat162*>(pz) : __floats2bfloat162_rn(0.f, 0.f);
          if constexpr (RAW)

            kvz[sj][rr] = l2_bf16(__bfloat1622float2(kvz[sj][rr]), rn2(j).x);
          pz += (size_t)8 * H * K;
        }
#pragma unroll
      for (int sj = 0; sj < NSB; ++sj)
#pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
          const int j = sj * SB + u + rr * 8, jj = u + rr * 8;
          const float2 kf = __bfloat1622float2(kvz[sj][rr]);



          const bool real = !VL || j < C_act;
          float v0 = real ? kf.x * __expf(a0v[sj] - sg[j][colp]) : 0.f;
          float v1 = real ? kf.y * __expf(a1v[sj] - sg[j][colp + 1]) : 0.f;
#pragma unroll
          for (int si = sj; si < NSB; ++si) {
            if (si > sj) {
              v0 *= f0[si - 1];
              v1 *= f1[si - 1];
            }
            *reinterpret_cast<__nv_bfloat162*>(&kz[si * (si + 1) / 2 + sj][jj][colp]) = __floats2bfloat162_rn(v0, v1);
          }
        }
    }
    if (tid < K / 4) {

      const float4 g4 = *reinterpret_cast<const float4*>(&sg[BT - 1][tid * 4]);
      *reinterpret_cast<float4*>(&gC[((size_t)c * H + h) * K + tid * 4]) =
          float4{__expf(g4.x), __expf(g4.y), __expf(g4.z), __expf(g4.w)};
    }
    __syncthreads();



    if (warp >= 10) {
      const int jn = job + 1;
      if (jn < nc1 * H && sub + 1 < njobs) {
        const int t1p = (jn % nc1) * BT + tokoff;
        const int Cp = VL ? min(BT, tse - t1p) : BT;
        const size_t tb = (size_t)t1p * H * K + (size_t)(jn / nc1) * K;
        const char* pq = reinterpret_cast<const char*>(q + tb);
        const char* pk = reinterpret_cast<const char*>(kk + tb);
        const char* pv = reinterpret_cast<const char*>(v + tb);
        const char* pg = reinterpret_cast<const char*>(glog + tb);
        for (int i = tid - 320; i < Cp; i += 192) {
          const size_t ro = (size_t)i * H * K * 2;
          for (int l = 0; l < 256; l += 128) {
            asm volatile("prefetch.global.L2 [%0];" ::"l"(pq + ro + l));
            asm volatile("prefetch.global.L2 [%0];" ::"l"(pk + ro + l));
            asm volatile("prefetch.global.L2 [%0];" ::"l"(pv + ro + l));
            asm volatile("prefetch.global.L2 [%0];" ::"l"(pg + ro + l));
          }
        }
      }
    }


    for (int pi = warp; pi < NSB * (NSB + 1) / 2; pi += 16) {
      int si = 0;
      while ((si + 1) * (si + 2) / 2 <= pi)
        ++si;
      const int sj = pi - si * (si + 1) / 2;
      float4 acck[2], accq[2];
      acck[0] = acck[1] = accq[0] = accq[1] = float4{0, 0, 0, 0};
      const int arow = lane & 15, aka = lane >> 4;
#pragma unroll
      for (int k16 = 0; k16 < K / 16; ++k16) {
        uint32_t a0, a1, a2, a3, q0, q1, q2, q3;
        ptx::ldmatrix_x4_b16(ptx::to_shared(&kw[si * SB + arow][k16 * 16 + aka * 8]), a0, a1, a2, a3);
        ptx::ldmatrix_x4_b16(ptx::to_shared(&qw[si * SB + arow][k16 * 16 + aka * 8]), q0, q1, q2, q3);
#pragma unroll
        for (int n8 = 0; n8 < 2; ++n8) {
          uint32_t b0, b1;
          ptx::ldmatrix_x2_b16(ptx::to_shared(&kz[pi][(lane & 7) + n8 * 8][k16 * 16 + ((lane >> 3) & 1) * 8]), b0, b1);
          ptx::mma_m16n8k16_bf16f32(acck[n8], a0, a1, a2, a3, b0, b1);
          ptx::mma_m16n8k16_bf16f32(accq[n8], q0, q1, q2, q3, b0, b1);
        }
      }




      const int r = lane >> 2, c2 = (lane & 3) * 2;
      const size_t abase = ((size_t)c * H + h) * BT * BT;
#pragma unroll
      for (int n8 = 0; n8 < 2; ++n8) {
        const float vals[4] = {acck[n8].x, acck[n8].y, acck[n8].z, acck[n8].w};
        const float valq[4] = {accq[n8].x, accq[n8].y, accq[n8].z, accq[n8].w};
#pragma unroll
        for (int e2 = 0; e2 < 2; ++e2) {
          const int ii = si * SB + r + e2 * 8;
          const int jj = sj * SB + n8 * 8 + c2;
          sA[ii][jj] = (jj < ii) ? sb[ii] * vals[e2 * 2] : 0.f;
          sA[ii][jj + 1] = (jj + 1 < ii) ? sb[ii] * vals[e2 * 2 + 1] : 0.f;
          if (jj <= ii) {
            const float q0 = valq[e2 * 2];
            const float q1 = jj + 1 <= ii ? valq[e2 * 2 + 1] : 0.f;
            const __nv_bfloat162 ah{__float2bfloat16(q0), __float2bfloat16(q1)};
            *reinterpret_cast<__nv_bfloat162*>(&aqk_h[abase + ii * BT + jj]) = ah;
            *reinterpret_cast<__nv_bfloat162*>(&aqk_l[abase + ii * BT + jj]) =
                __floats2bfloat162_rn(q0 - __bfloat162float(ah.x), q1 - __bfloat162float(ah.y));
          }
        }
      }
    }
    __syncthreads();

    pend_wait();


    for (int p2 = tid; p2 < BT * 24; p2 += blockDim.x) {
      const int p = p2 * 2;
      const int i = p / 48, j = p % 48;



      const float2 av = j / SB > i / SB ? float2{0.f, 0.f} : *reinterpret_cast<const float2*>(&sA[i][j]);
      const bf16 ah0 = __float2bfloat16(av.x);
      const bf16 ah1 = __float2bfloat16(av.y);
      *reinterpret_cast<__nv_bfloat162*>(&sp.f.aC_h[i][j]) = __nv_bfloat162{ah0, ah1};
      *reinterpret_cast<__nv_bfloat162*>(&sp.f.aC_l[i][j]) =
          __floats2bfloat162_rn(av.x - __bfloat162float(ah0), av.y - __bfloat162float(ah1));
    }
    __syncthreads();



#pragma unroll
    for (int b = 0; b < NSB; ++b) {
      if (b) {
        const int arow = lane & 15, aka = lane >> 4;
        float4 acc[2];
        acc[0] = acc[1] = float4{0, 0, 0, 0};
        for (int pr = 0; pr < 3; ++pr) {
          const bf16(*ta)[56] = pr == 1 ? sp.f.aC_l : sp.f.aC_h;
          const bf16(*tb)[56] = pr == 2 ? sp.f.upT_l : sp.f.upT_h;
          for (int k16 = 0; k16 < b; ++k16) {
            uint32_t a0, a1, a2, a3;
            ptx::ldmatrix_x4_b16(ptx::to_shared(&ta[b * SB + arow][k16 * 16 + aka * 8]), a0, a1, a2, a3);
#pragma unroll
            for (int n8 = 0; n8 < 2; ++n8) {
              uint32_t b0, b1;
              ptx::ldmatrix_x2_b16(
                  ptx::to_shared(&tb[warp * 16 + n8 * 8 + (lane & 7)][k16 * 16 + ((lane >> 3) & 1) * 8]), b0, b1);
              ptx::mma_m16n8k16_bf16f32(acc[n8], a0, a1, a2, a3, b0, b1);
            }
          }
        }
        const int fr = lane >> 2, fc = (lane & 3) * 2;
#pragma unroll
        for (int n8 = 0; n8 < 2; ++n8) {
          const float vals[4] = {acc[n8].x, acc[n8].y, acc[n8].z, acc[n8].w};
#pragma unroll
          for (int e = 0; e < 4; ++e)
            sU[b * SB + fr + (e >> 1) * 8][warp * 16 + n8 * 8 + fc + (e & 1)] -= vals[e];
        }
      }
      __syncthreads();
      if (tid < 2 * K) {
        const int col = tid;
        float r[SB];
#pragma unroll
        for (int i = 0; i < SB; ++i)
          r[i] = sU[b * SB + i][col];
#pragma unroll
        for (int i = 1; i < SB; ++i)
#pragma unroll
          for (int j = 0; j < i; ++j)
            r[i] -= sA[b * SB + i][b * SB + j] * r[j];
#pragma unroll
        for (int i = 0; i < SB; ++i)
          sU[b * SB + i][col] = r[i];
        if (b + 1 < NSB)


#pragma unroll
          for (int i = 0; i < SB; i += 8) {
            alignas(16) __nv_bfloat162 hv[4], lv[4];
#pragma unroll
            for (int w = 0; w < 4; ++w) {
              const float e0 = r[i + 2 * w], e1 = r[i + 2 * w + 1];
              hv[w] = __nv_bfloat162{__float2bfloat16(e0), __float2bfloat16(e1)};
              lv[w] = __floats2bfloat162_rn(e0 - __bfloat162float(hv[w].x), e1 - __bfloat162float(hv[w].y));
            }
            *reinterpret_cast<uint4*>(&sp.f.upT_h[col][b * SB + i]) = *reinterpret_cast<const uint4*>(hv);
            *reinterpret_cast<uint4*>(&sp.f.upT_l[col][b * SB + i]) = *reinterpret_cast<const uint4*>(lv);
          }
      }
      __syncthreads();
    }
    const size_t base = ((size_t)c * H + h) * BT * K;
    for (int p2 = tid; p2 < BT * K / 2; p2 += blockDim.x) {
      const int p = p2 * 2;
      *reinterpret_cast<__nv_bfloat162*>(&P[base + p]) =
          __floats2bfloat162_rn(-sU[p / K][p % K], -sU[p / K][p % K + 1]);
      *reinterpret_cast<float2*>(&u0[base + p]) = float2{sU[p / K][K + p % K], sU[p / K][K + p % K + 1]};
    }
    if (!pieceL) continue;












    __syncthreads();






    if (tid == 0) {
      ptx::fence_async_global();
      ptx::mbar_arrive_expect_tx(&mb_f, uint32_t(sizeof(sp.b.pTt)));
      ptx::cp_async_bulk_tensor_2d_load(ptx::to_shared(&sp.b.pTt[0][0]), ptm, 0, (c * H + h) * BT, &mb_f);
      ptx::cp_async_bulk_tensor_2d_load(ptx::to_shared(&sp.b.pTt[64][0]), ptm, 64, (c * H + h) * BT, &mb_f);
    }
    const size_t cbase = ((size_t)c * H + h) * BT * K;
#pragma unroll
    for (int hb = 0; hb < 2; ++hb) {
      bf16 kdv[8];
#pragma unroll
      for (int j = 0; j < 8; ++j)
        kdv[j] = kdec[cbase + tid + (hb * 8 + j) * 512];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int p = tid + (hb * 8 + j) * 512;
        const int kc = p / BT, tok = p % BT;

        const int tokS = (((tok >> 3) ^ (kc & 7)) << 3) | (tok & 7);
        sp.b.kdT[kc][tokS] = kdv[j];
        const float uv = sU[tok][K + kc];
        const bf16 uh = __float2bfloat16(uv);
        sp.b.u0Th[kc][tokS] = uh;
        sp.b.u0Tl[kc][tokS] = __float2bfloat16(uv - __bfloat162float(uh));
      }
    }
    ptx::tcgen05_fence_before_thread_sync();
    __syncthreads();
    const uint32_t taddr = s_taddr;
    const uint32_t idw = ptx::mma_inst_desc_dense<ptx::MmaDenseKind::F16>(
        128, 128, ptx::F16Type::BF16, ptx::F16Type::BF16, ptx::DType::F32, ptx::Major::K, ptx::Major::K);

    const uint32_t idwm = ptx::mma_inst_desc_dense<ptx::MmaDenseKind::F16>(
        128, 128, ptx::F16Type::BF16, ptx::F16Type::BF16, ptx::DType::F32, ptx::Major::K, ptx::Major::MN);
    const uint32_t tp = (sub & 1) ? 256u : 0u;


    ptx::mbar_wait_parity(&mb_f, mbph & 1);
    ++mbph;
    if (tid == 0) {
      ptx::tcgen05_fence_after_thread_sync();

#pragma unroll
      for (int k16 = 0; k16 < BT / 16; ++k16) {
        const uint64_t da = ptx::mma_smem_desc_k_major<uint16_t, 64, 128>(ptx::to_shared(&sp.b.kdT[0][0]) + k16 * 32);
        const uint64_t db =
            ptx::mma_smem_desc_mn_major<uint16_t, 64, 128, 128>(ptx::to_shared(&sp.b.pTt[0][0]) + k16 * 2048);
        ptx::tcgen05_mma_f16(taddr + tp, da, db, idwm, k16 ? 1u : 0u);
      }


      for (int half = 0; half < 2; ++half)
#pragma unroll
        for (int k16 = 0; k16 < BT / 16; ++k16) {
          const uint64_t da = ptx::mma_smem_desc_k_major<uint16_t, 64, 128>(
              ptx::to_shared(half ? &sp.b.u0Tl[0][0] : &sp.b.u0Th[0][0]) + k16 * 32);
          const uint64_t db = ptx::mma_smem_desc_k_major<uint16_t, 64, 128>(ptx::to_shared(&sp.b.kdT[0][0]) + k16 * 32);
          ptx::tcgen05_mma_f16(taddr + tp + 128u, da, db, idw, (half | k16) ? 1u : 0u);
        }
      ptx::tcgen05_commit_arrive(&mb_f);
    }
    if constexpr (GM != 0)




      if (sub + 1 < njobs) {
        ptx::cp_async_wait_pending(0);
        const int t1x = ((job + 1) % nc1) * BT + tokoff;
        gate_xform(S.sgb[(sub + 1) & 1], VL ? min(BT, tse - t1x) : BT);
      }
    ptx::mbar_wait_parity(&mb_f, mbph & 1);
    ++mbph;
    ptx::tcgen05_fence_after_thread_sync();
    const int band = (warp & 3) * 32, ch = warp >> 2;
    const uint32_t lane_hi = uint32_t(band) << 16;




    for (int b = 0; b < 4; ++b) {
      uint32_t r[8];
      ptx::tcgen05_ld_32x32b_x8(
          taddr + lane_hi + (tp + uint32_t(ch * 32 + b * 8)), r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);
      ptx::tcgen05_wait_ld();
      float lt[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int kin = ch * 32 + b * 8 + j;


        lt[j] = (band + lane == kin ? gC[((size_t)c * H + h) * K + kin] : 0.f) + __int_as_float(r[j]);
      }
      const bool sw = ((band + lane) >> 2) & 1;
      const float4 lo = float4{lt[0], lt[1], lt[2], lt[3]};
      const float4 hi = float4{lt[4], lt[5], lt[6], lt[7]};
      float4* dst = reinterpret_cast<float4*>(&sp.g.sLt[ch * 4 + b][band + lane][0]);
      dst[0] = sw ? hi : lo;
      dst[1] = sw ? lo : hi;
    }
    ptx::tcgen05_fence_before_thread_sync();
    __syncthreads();
    {
      const int vc = band + lane;
      auto drain_reg = [&](uint32_t dc, float* r2) {
#pragma unroll
        for (int b2 = 0; b2 < 4; ++b2) {
          uint32_t r[8];
          ptx::tcgen05_ld_32x32b_x8(
              taddr + lane_hi + dc + uint32_t(ch * 32 + b2 * 8), r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);
          ptx::tcgen05_wait_ld();
#pragma unroll
          for (int j = 0; j < 8; ++j)
            r2[b2 * 8 + j] = __int_as_float(r[j]);
        }
      };
      auto store_maps = [&](const float* lr2, const float* cr2) {
        const size_t sb2 = (size_t)pidx * K * K;





#pragma unroll
        for (int j = 0; j < 32; ++j) {
          pieceL[sb2 + (size_t)(ch * 32 + j) * K + vc] = __float2bfloat16(lr2[j]);
          piecec[sb2 + (size_t)(ch * 32 + j) * K + vc] = cr2[j];
        }
      };
      const bool piece_end = sub + 1 == njobs;
      if (sub == 0) {


        alignas(16) float lr[32];
#pragma unroll
        for (int j = 0; j < 32; ++j) {
          const int out = ch * 32 + j;
          lr[j] = sp.g.sLt[vc >> 3][out][(vc & 7) ^ (((out >> 2) & 1) << 2)];
        }
        if (piece_end) {
          alignas(16) float cr[32];
          drain_reg(tp + 128u, cr);
          store_maps(lr, cr);
        } else {


#pragma unroll
          for (int c4 = 0; c4 < 32; c4 += 4)
            ptx::tcgen05_st_32x32b_x4(
                taddr + lane_hi + (tp + uint32_t(ch * 32 + c4)),
                __float_as_int(lr[c4]),
                __float_as_int(lr[c4 + 1]),
                __float_as_int(lr[c4 + 2]),
                __float_as_int(lr[c4 + 3]));
          ptx::tcgen05_wait_st();
        }
      } else {




        if (tid == 0) {
          const uint32_t idt = ptx::mma_inst_desc_tf32(128, K);
          const uint32_t sp2 = tp ^ 256u;
          for (int st2 = 0; st2 < 2; ++st2)
#pragma unroll
            for (int s = 0; s < 16; ++s)
              mma_tf32_atmem(
                  taddr + tp + uint32_t(st2 * 128),
                  taddr + (sp2 + uint32_t(st2 * 128 + 8 * s)),
                  ptx::mma_smem_desc_k_major<uint32_t, 8, 32>(ptx::to_shared(&sp.g.sLt[s][0][0])),
                  idt,
                  (st2 | s) ? 1u : 0u);
          ptx::tcgen05_commit_arrive(&mb_f);
        }
        if (piece_end) {
          ptx::mbar_wait_parity(&mb_f, mbph & 1);
          ++mbph;
          ptx::tcgen05_fence_after_thread_sync();
          alignas(16) float lr[32], cr[32];
          drain_reg(tp, lr);
          drain_reg(tp + 128u, cr);
          store_maps(lr, cr);
        } else {
          pend = true;
        }
      }
    }
    if (sub + 1 < njobs) __syncthreads();
  }
  if (pieceL) {
    if (warp == 0) ptx::tcgen05_dealloc(s_taddr, 512);
    ptx::tcgen05_relinquish();
  }
}

template <int GM = 0, bool RAW = false, bool BSIG = false>
__global__ void __launch_bounds__(512) k1_factors_mma(
    const bf16* __restrict__ q,
    const bf16* __restrict__ kk,
    const bf16* __restrict__ v,
    const bf16* __restrict__ glog,
    const float* __restrict__ a_log,
    const float* __restrict__ dtb,
    float lb,
    const float* __restrict__ beta,
    int T,
    int H,
    float scale,
    bf16* __restrict__ P,
    float* __restrict__ u0,
    bf16* __restrict__ kdec,
    bf16* __restrict__ qdec,
    bf16* __restrict__ aqk_h,
    bf16* __restrict__ aqk_l,
    float* __restrict__ gC,
    bf16* __restrict__ pieceL,
    float* __restrict__ piecec,
    int NP) {
  __shared__ K1Smem S;


  const int nc1 = T / BT, bid = blockIdx.x;
  const int h = bid / NP, pc = bid % NP;
  const int c00 = piece_c0(pc, nc1, NP);
  const int njobs = piece_c0(pc + 1, nc1, NP) - c00;
  k1_body<false, GM, RAW, BSIG>(
      S,
      h * nc1 + c00,
      njobs,
      q,
      kk,
      v,
      glog,
      beta,
      T,
      H,
      scale,
      P,
      u0,
      kdec,
      qdec,
      aqk_h,
      aqk_l,
      gC,
      pieceL,
      piecec,
      pc * H + h,
      0,
      0,
      0,
      a_log,
      dtb,
      lb);
}














template <bool VL = false>
__device__ static void chain_body(
    ChainSmem& S,
    int h,
    int sg,
    int nseg,
    const CUtensorMap& pneg_map,
    const CUtensorMap& kdt_map,
    const CUtensorMap& qd_map,
    const CUtensorMap& aqh_map,
    const CUtensorMap& aql_map,
    const CUtensorMap& u0f_map,
    const float* __restrict__ gC,
    int n_chunks,
    int H,
    bf16* __restrict__ o,
    float* __restrict__ Sf,


    void* __restrict__ hpc,
    bool hpc_bf16,
    bool hpc_v_first,
    const CUtensorMap* sl_map,
    const CUtensorMap* sc_map,
    const float* __restrict__ h0s,
    const uint32_t* __restrict__ pflags = nullptr,



    int cbase = 0,
    int pbase = 0,
    int tokoff = 0,
    int tend = 0) {
  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int band = (warp & 3) * 32, ch = warp >> 2;
  const int vc = band + lane;
  auto& sPneg = S.sPneg;
  auto& sKdT = S.sKdT;
  auto& sQd = S.sQd;
  auto& sAh = S.sAh;
  auto& sAl = S.sAl;
  auto& sU0 = S.sU0;
  auto& mb_tma = S.mb_tma;
  auto& mb_p1 = S.mb_p1;
  auto& mb_p2 = S.mb_p2;
  auto& mb_o = S.mb_o;
  auto& s_taddr = S.s_taddr;
  if (tid == 0) {
    ptx::mbar_init(&mb_tma, 1);
    ptx::mbar_init(&mb_p1, 1);
    ptx::mbar_init(&mb_p2, 1);
    ptx::mbar_init(&mb_o, 2);
    ptx::mbar_init(&S.mb_pre, 1);
    ptx::mbar_init(&S.mb_preC, 1);
    ptx::mbar_init(&S.mb_cmL, 1);
    ptx::mbar_init(&S.mb_cmC, 1);
    ptx::prefetch_tensormap(&pneg_map);
    ptx::prefetch_tensormap(&kdt_map);
  }
  if (warp == 0) ptx::tcgen05_alloc(ptx::to_shared(&s_taddr), 512);
  __syncthreads();
  const uint32_t taddr = s_taddr;
  const uint32_t lane_hi = uint32_t(band) << 16;

  const int c0 = cbase + piece_c0(sg, n_chunks, nseg);
  const int per_seg = cbase + piece_c0(sg + 1, n_chunks, nseg) - c0;
  const int tse = VL ? tend : n_chunks * BT;
  alignas(16) float Sreg[32];
















#pragma unroll
  for (int j = 0; j < 32; ++j)
    Sreg[j] = h0s[((size_t)h * K + (ch * 32 + j)) * K + vc];
  if (sg > 0) {
    float* bL = reinterpret_cast<float*>(&sPneg[0][0][0][0]);
    float* bC = bL + K * K;
    bf16* pL = reinterpret_cast<bf16*>(bL + K * K / 2);
    static_assert(sizeof(S.sPneg) + sizeof(S.sKdT) + sizeof(S.sQd) + sizeof(S.sAh) + sizeof(S.sAl) == 2 * K * K * 4);
#pragma unroll
    for (int c4 = 0; c4 < 32; c4 += 4) {
      uint32_t iv[4], sv[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        iv[e] = __float_as_int(vc == ch * 32 + c4 + e ? 1.f : 0.f);
        sv[e] = __float_as_int(Sreg[c4 + e]);
      }
      ptx::tcgen05_st_32x32b_x4(taddr + lane_hi + uint32_t(256 + ch * 32 + c4), iv[0], iv[1], iv[2], iv[3]);
      ptx::tcgen05_st_32x32b_x4(taddr + lane_hi + uint32_t(ch * 32 + c4), sv[0], sv[1], sv[2], sv[3]);
    }
    ptx::tcgen05_wait_st();
    ptx::tcgen05_fence_before_thread_sync();
    __syncthreads();
    const uint32_t idt = ptx::mma_inst_desc_tf32(128, K);
    for (int q2 = 0; q2 < sg; ++q2) {
      if (tid == 0) {
        if (pflags)
          while (ptx::ld_acq_b32(&pflags[(pbase + q2) * H + h]) < 1u)
            __nanosleep(128);
        if (q2) ptx::mbar_wait_parity(&S.mb_cmL, (q2 - 1) & 1);
        ptx::mbar_arrive_expect_tx(&S.mb_pre, K * K * 2);
        ptx::cp_async_bulk_tensor_2d_load(ptx::to_shared(pL), sl_map, 0, ((pbase + q2) * H + h) * K, &S.mb_pre);
        ptx::cp_async_bulk_tensor_2d_load(
            ptx::to_shared(pL) + uint32_t(K * 64 * 2), sl_map, 64, ((pbase + q2) * H + h) * K, &S.mb_pre);
        if (q2) ptx::mbar_wait_parity(&S.mb_cmC, (q2 - 1) & 1);
        ptx::mbar_arrive_expect_tx(&S.mb_preC, K * K * 4);
        for (int s = 0; s < 16; ++s)
          ptx::cp_async_bulk_tensor_2d_load(
              ptx::to_shared(bC) + uint32_t(s * 4096), sc_map, 8 * s, ((pbase + q2) * H + h) * K, &S.mb_preC);
      }



      ptx::mbar_wait_parity(&S.mb_pre, q2 & 1);
      widen_map_tf32(bL, pL, vc, ch);
      ptx::tcgen05_fence_before_thread_sync();
      __syncthreads();
      if (tid == 0) {
        ptx::tcgen05_fence_after_thread_sync();
        const uint32_t src = (q2 & 1) ? 128u : 0u, tgt = 128u - src;
        for (int s = 0; s < 16; ++s)
          mma_tf32_atmem(
              taddr + tgt,
              taddr + src + uint32_t(8 * s),
              ptx::mma_smem_desc_k_major<uint32_t, 8, 32>(ptx::to_shared(bL) + uint32_t(s * 4096)),
              idt,
              s ? 1u : 0u);
        ptx::tcgen05_commit_arrive(&S.mb_cmL);
        ptx::mbar_wait_parity(&S.mb_preC, q2 & 1);
        for (int s = 0; s < 16; ++s)
          mma_tf32_atmem(
              taddr + tgt,
              taddr + uint32_t(256 + 8 * s),
              ptx::mma_smem_desc_k_major<uint32_t, 8, 32>(ptx::to_shared(bC) + uint32_t(s * 4096)),
              idt,
              1u);
        ptx::tcgen05_commit_arrive(&S.mb_cmC);
      }
    }
    if (tid == 0)
      ptx::mbar_wait_parity(&S.mb_cmC, (sg - 1) & 1);
    __syncthreads();
    ptx::tcgen05_fence_after_thread_sync();
    const uint32_t fin = ((sg - 1) & 1) ? 0u : 128u;
#pragma unroll
    for (int b2 = 0; b2 < 4; ++b2) {
      uint32_t r[8];
      ptx::tcgen05_ld_32x32b_x8(
          taddr + lane_hi + fin + uint32_t(ch * 32 + b2 * 8), r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);
      ptx::tcgen05_wait_ld();
#pragma unroll
      for (int j = 0; j < 8; ++j)
        Sreg[b2 * 8 + j] = __int_as_float(r[j]);
    }
    __syncthreads();
  }

  auto pack2 = [](float e0, float e1, uint32_t& hi, uint32_t& lo) {
    const __nv_bfloat162 h2 = __floats2bfloat162_rn(e0, e1);
    const __nv_bfloat162 l2 = __floats2bfloat162_rn(e0 - __bfloat162float(h2.x), e1 - __bfloat162float(h2.y));
    hi = *reinterpret_cast<const uint32_t*>(&h2);
    lo = *reinterpret_cast<const uint32_t*>(&l2);
  };
  if (pflags) {
    if (tid == 0)
      while (ptx::ld_acq_b32(&pflags[(pbase + sg) * H + h]) < 1u)
        __nanosleep(128);
    __syncthreads();
  }
  auto tma_issue = [&](int cc) {
    const int bu2 = (cc - c0) & 1;
    ptx::mbar_arrive_expect_tx(&mb_tma, 6 * 64 * 64 * 2 + 128 * 64 * 2 + BT * K * 4);
    ptx::cp_async_bulk_tensor_2d_load(ptx::to_shared(&sPneg[bu2][0][0][0]), &pneg_map, 0, (cc * H + h) * BT, &mb_tma);
    ptx::cp_async_bulk_tensor_2d_load(ptx::to_shared(&sPneg[bu2][1][0][0]), &pneg_map, 64, (cc * H + h) * BT, &mb_tma);
    ptx::cp_async_bulk_tensor_2d_load(ptx::to_shared(&sKdT[bu2][0][0]), &kdt_map, 0, (cc * H + h) * K, &mb_tma);
    ptx::cp_async_bulk_tensor_2d_load(ptx::to_shared(&sQd[bu2][0][0][0]), &qd_map, 0, (cc * H + h) * BT, &mb_tma);
    ptx::cp_async_bulk_tensor_2d_load(ptx::to_shared(&sQd[bu2][1][0][0]), &qd_map, 64, (cc * H + h) * BT, &mb_tma);
    ptx::cp_async_bulk_tensor_2d_load(ptx::to_shared(&sAh[bu2][0][0]), &aqh_map, 0, (cc * H + h) * BT, &mb_tma);
    ptx::cp_async_bulk_tensor_2d_load(ptx::to_shared(&sAl[bu2][0][0]), &aql_map, 0, (cc * H + h) * BT, &mb_tma);
    ptx::cp_async_bulk_tensor_2d_load(ptx::to_shared(&sU0[bu2][0][0]), &u0f_map, 0, (cc * H + h) * BT, &mb_tma);
  };
  if (tid == 0 && (!VL || per_seg > 0)) tma_issue(c0);



  const float* gCp = gC + ((size_t)c0 * H + h) * K + ch * 32;
  const uint32_t a1st = taddr + lane_hi + (256u + ch * 16);
  for (int c = c0; c < c0 + per_seg; ++c) {
    const int buf_idx = (c - c0) & 1;


    alignas(16) float gv[32];
#pragma unroll
    for (int j = 0; j < 8; ++j)
      reinterpret_cast<float4*>(gv)[j] = reinterpret_cast<const float4*>(gCp)[j];
    gCp += (size_t)H * K;



    {
      uint32_t hw[16];
#pragma unroll
      for (int w = 0; w < 16; ++w) {
        const __nv_bfloat162 h2 = __floats2bfloat162_rn(Sreg[2 * w], Sreg[2 * w + 1]);
        hw[w] = *reinterpret_cast<const uint32_t*>(&h2);
      }
      ptx::tcgen05_st_32x32b_x16(a1st, hw);
    }



    if (hpc) {
      const size_t hb = ((size_t)c * H + h) * K * K;
      if (hpc_v_first) {
        const size_t off = hb + (size_t)vc * K + ch * 32;
        if (hpc_bf16) {
          auto* p = reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<bf16*>(hpc) + off);
#pragma unroll
          for (int j = 0; j < 16; ++j)
            p[j] = __floats2bfloat162_rn(Sreg[2 * j], Sreg[2 * j + 1]);
        } else {
          auto* p = reinterpret_cast<float4*>(reinterpret_cast<float*>(hpc) + off);
#pragma unroll
          for (int j = 0; j < 8; ++j)
            p[j] = reinterpret_cast<const float4*>(Sreg)[j];
        }
      } else {
        const size_t off = hb + (size_t)(ch * 32) * K + vc;
        if (hpc_bf16) {
          bf16* p = reinterpret_cast<bf16*>(hpc) + off;
#pragma unroll
          for (int j = 0; j < 32; ++j)
            p[j * K] = __float2bfloat16(Sreg[j]);
        } else {
          float* p = reinterpret_cast<float*>(hpc) + off;
#pragma unroll
          for (int j = 0; j < 32; ++j)
            p[j * K] = Sreg[j];
        }
      }
    }
    ptx::tcgen05_wait_st();
    ptx::tcgen05_fence_before_thread_sync();
    __syncthreads();




    ptx::mbar_wait_parity(&mb_tma, (c - c0) & 1);
    if (tid == 0 || tid == 32) {

      if (tid == 0 && c + 1 < c0 + per_seg) tma_issue(c + 1);
      ptx::tcgen05_fence_after_thread_sync();
      const uint32_t idesc1 = ptx::mma_inst_desc_dense<ptx::MmaDenseKind::F16>(
          128, 64, ptx::F16Type::BF16, ptx::F16Type::BF16, ptx::DType::F32, ptx::Major::K, ptx::Major::K);
      if (tid == 0) {
#pragma unroll
        for (int k16 = 0; k16 < 8; ++k16) {
          const uint64_t db = ptx::mma_smem_desc_k_major<uint16_t, 64, 128>(
              ptx::to_shared(&sPneg[buf_idx][k16 >> 2][0][0]) + (k16 & 3) * 32);
          ptx::tcgen05_mma_f16_atmem(taddr, taddr + (256u + k16 * 8), db, idesc1, k16 ? 1u : 0u);
        }
        ptx::tcgen05_commit_arrive(&mb_p1);
      } else {
#pragma unroll
        for (int k16 = 0; k16 < 8; ++k16) {
          const uint64_t db = ptx::mma_smem_desc_k_major<uint16_t, 64, 128>(
              ptx::to_shared(&sQd[buf_idx][k16 >> 2][0][0]) + (k16 & 3) * 32);
          ptx::tcgen05_mma_f16_atmem(taddr + 384u, taddr + (256u + k16 * 8), db, idesc1, k16 ? 1u : 0u);
        }
        ptx::tcgen05_commit_arrive(&mb_o);
      }
    }
    ptx::mbar_wait_parity(&mb_p1, (c - c0) & 1);
    ptx::tcgen05_fence_after_thread_sync();


    {
      alignas(16) float Ur[16];
      {
        uint32_t r[16];
        ptx::tcgen05_ld_32x32b_x16(
            taddr + lane_hi + uint32_t(ch * 16),
            r[0],
            r[1],
            r[2],
            r[3],
            r[4],
            r[5],
            r[6],
            r[7],
            r[8],
            r[9],
            r[10],
            r[11],
            r[12],
            r[13],
            r[14],
            r[15]);
        ptx::tcgen05_wait_ld();
#pragma unroll
        for (int j = 0; j < 16; ++j)
          Ur[j] = __int_as_float(r[j]) + sU0[buf_idx][ch * 16 + j][vc];
      }
      uint32_t hw[8], lw[8];
#pragma unroll
      for (int w = 0; w < 8; ++w)
        pack2(Ur[2 * w], Ur[2 * w + 1], hw[w], lw[w]);
      ptx::tcgen05_st_32x32b_x8(
          taddr + lane_hi + (192u + ch * 8), hw[0], hw[1], hw[2], hw[3], hw[4], hw[5], hw[6], hw[7]);
      ptx::tcgen05_st_32x32b_x8(
          taddr + lane_hi + (224u + ch * 8), lw[0], lw[1], lw[2], lw[3], lw[4], lw[5], lw[6], lw[7]);
    }
    ptx::tcgen05_wait_st();
    ptx::tcgen05_fence_before_thread_sync();
    __syncthreads();
    if (tid == 0 || tid == 32) {
      ptx::tcgen05_fence_after_thread_sync();
      if (tid == 0) {
        const uint32_t idesc2 = ptx::mma_inst_desc_dense<ptx::MmaDenseKind::F16>(
            128, 128, ptx::F16Type::BF16, ptx::F16Type::BF16, ptx::DType::F32, ptx::Major::K, ptx::Major::K);
        for (int half = 0; half < 2; ++half)
#pragma unroll
          for (int k16 = 0; k16 < 4; ++k16) {
            const uint64_t db =
                ptx::mma_smem_desc_k_major<uint16_t, 64, 128>(ptx::to_shared(&sKdT[buf_idx][0][0]) + k16 * 32);
            ptx::tcgen05_mma_f16_atmem(
                taddr + 64u, taddr + (192u + half * 32 + k16 * 8), db, idesc2, (half | k16) ? 1u : 0u);
          }
        ptx::tcgen05_commit_arrive(&mb_p2);
      } else {
        const uint32_t idesc4 = ptx::mma_inst_desc_dense<ptx::MmaDenseKind::F16>(
            128, 64, ptx::F16Type::BF16, ptx::F16Type::BF16, ptx::DType::F32, ptx::Major::K, ptx::Major::K);

        for (int pr = 0; pr < 3; ++pr) {
          const uint32_t acol = pr == 2 ? 224u : 192u;
          const bf16* ab = pr == 1 ? &sAl[buf_idx][0][0] : &sAh[buf_idx][0][0];
#pragma unroll
          for (int k16 = 0; k16 < 4; ++k16) {
            const uint64_t db = ptx::mma_smem_desc_k_major<uint16_t, 64, 128>(ptx::to_shared(ab) + k16 * 32);
            ptx::tcgen05_mma_f16_atmem(taddr + 384u, taddr + (acol + k16 * 8), db, idesc4, 1u);
          }
        }
        ptx::tcgen05_commit_arrive(&mb_o);
      }
    }
    ptx::mbar_wait_parity(&mb_p2, (c - c0) & 1);
    ptx::tcgen05_fence_after_thread_sync();


    {
      uint32_t r[32];
      ptx::tcgen05_ld_32x32b_x32(taddr + lane_hi + uint32_t(64 + ch * 32), r);
      ptx::tcgen05_wait_ld();
#pragma unroll
      for (int jj = 0; jj < 32; ++jj)
        Sreg[jj] = gv[jj] * Sreg[jj] + __int_as_float(r[jj]);
    }

    ptx::mbar_wait_parity(&mb_o, (c - c0) & 1);
    ptx::tcgen05_fence_after_thread_sync();
    {
      uint32_t r[16];
      ptx::tcgen05_ld_32x32b_x16(
          taddr + lane_hi + uint32_t(384 + ch * 16),
          r[0],
          r[1],
          r[2],
          r[3],
          r[4],
          r[5],
          r[6],
          r[7],
          r[8],
          r[9],
          r[10],
          r[11],
          r[12],
          r[13],
          r[14],
          r[15]);
      ptx::tcgen05_wait_ld();


      const int row0 = c * BT + tokoff + ch * 16;
      bf16* op = o + ((size_t)row0 * H + h) * K + vc;
      const size_t orow = (size_t)H * K;
#pragma unroll
      for (int j = 0; j < 16; ++j, op += orow)
        if (!VL || row0 + j < tse)
          *op = __float2bfloat16(__int_as_float(r[j]));
    }
    __syncthreads();
  }
  if (sg == nseg - 1)
#pragma unroll
    for (int j = 0; j < 32; ++j)
      Sf[((size_t)h * K + ch * 32 + j) * K + vc] = Sreg[j];
  __syncthreads();
  if (warp == 0) ptx::tcgen05_dealloc(taddr, 512);
  ptx::tcgen05_relinquish();
}


__global__ void __launch_bounds__(512) k2_chain_tc(
    const __grid_constant__ CUtensorMap pneg_map,
    const __grid_constant__ CUtensorMap kdt_map,
    const __grid_constant__ CUtensorMap qd_map,
    const __grid_constant__ CUtensorMap aqh_map,
    const __grid_constant__ CUtensorMap aql_map,
    const __grid_constant__ CUtensorMap sl_map,
    const __grid_constant__ CUtensorMap sc_map,
    const __grid_constant__ CUtensorMap u0f_map,
    const float* __restrict__ gC,
    const float* __restrict__ h0,
    int n_chunks,
    int H,
    bf16* __restrict__ o,
    float* __restrict__ Sf,
    void* __restrict__ hpc,
    bool hpc_bf16,
    bool hpc_v_first) {
  __shared__ ChainSmem S;
  chain_body(
      S,
      blockIdx.x,
      blockIdx.y,
      gridDim.y,
      pneg_map,
      kdt_map,
      qd_map,
      aqh_map,
      aql_map,
      u0f_map,
      gC,
      n_chunks,
      H,
      o,
      Sf,
      hpc,
      hpc_bf16,
      hpc_v_first,
      &sl_map,
      &sc_map,
      h0);
}







template <int GM = 0, bool RAW = false, bool BSIG = false>
__global__ void __launch_bounds__(512) kda_fused(
    const bf16* __restrict__ q,
    const bf16* __restrict__ kk,
    const bf16* __restrict__ v,
    const bf16* __restrict__ glog,
    const float* __restrict__ a_log,
    const float* __restrict__ dtb,
    float lb,
    const float* __restrict__ beta,
    int T,
    int H,
    float scale,
    bf16* __restrict__ P,
    float* __restrict__ u0,
    bf16* __restrict__ kdec,
    bf16* __restrict__ qdec,
    bf16* __restrict__ aqk_h,
    bf16* __restrict__ aqk_l,
    float* __restrict__ gC,
    bf16* __restrict__ pieceL,
    float* __restrict__ piecec,
    int NP,
    const __grid_constant__ CUtensorMap pneg_map,
    const __grid_constant__ CUtensorMap kdt_map,
    const __grid_constant__ CUtensorMap qd_map,
    const __grid_constant__ CUtensorMap aqh_map,
    const __grid_constant__ CUtensorMap aql_map,
    const __grid_constant__ CUtensorMap u0f_map,
    const __grid_constant__ CUtensorMap sl_map,
    const __grid_constant__ CUtensorMap sc_map,
    const float* __restrict__ h0,
    bf16* __restrict__ o,
    float* __restrict__ Sf,
    void* __restrict__ hpc,
    bool hpc_bf16,
    bool hpc_v_first,
    uint32_t* __restrict__ pflags) {
  union FusedSmem {
    K1Smem k1;
    ChainSmem chain;
  };
  __shared__ FusedSmem S;
  const int tid = threadIdx.x, bid = blockIdx.x;
  const int nc1 = T / BT;
  if (bid < NP * H) {
    const int h = bid / NP, pc = bid % NP;
    const int c00 = piece_c0(pc, nc1, NP);
    const int njobs = piece_c0(pc + 1, nc1, NP) - c00;
    k1_body<false, GM, RAW, BSIG>(
        S.k1,
        h * nc1 + c00,
        njobs,
        q,
        kk,
        v,
        glog,
        beta,
        T,
        H,
        scale,
        P,
        u0,
        kdec,
        qdec,
        aqk_h,
        aqk_l,
        gC,
        pieceL,
        piecec,
        pc * H + h,
        0,
        0,
        0,
        a_log,
        dtb,
        lb,
        &pneg_map);
    __syncthreads();
    if (tid == 0) {
      ptx::fence_async_global();
      ptx::red_add_rel_b32(&pflags[pc * H + h], 1u);
    }
  } else {
    const int r = bid - NP * H, h = r % H, sg = r / H;
    chain_body(
        S.chain,
        h,
        sg,
        NP,
        pneg_map,
        kdt_map,
        qd_map,
        aqh_map,
        aql_map,
        u0f_map,
        gC,
        nc1,
        H,
        o,
        Sf,
        hpc,
        hpc_bf16,
        hpc_v_first,
        &sl_map,
        &sc_map,
        h0,
        pflags);
  }
}















struct VlPiece {
  int seq;
  int cbase;
  int nc;
  int np;
  int sg;
  int tokoff;
  int tend;
  int tail;
};
template <int GM = 0, bool RAW = false, bool BSIG = false>
__global__ void __launch_bounds__(512) kda_fused_vl(
    const bf16* __restrict__ q,
    const bf16* __restrict__ kk,
    const bf16* __restrict__ v,
    const bf16* __restrict__ glog,
    const float* __restrict__ a_log,
    const float* __restrict__ dtb,
    float lb,
    const float* __restrict__ beta,
    int T,
    int H,
    float scale,
    bf16* __restrict__ P,
    float* __restrict__ u0,
    bf16* __restrict__ kdec,
    bf16* __restrict__ qdec,
    bf16* __restrict__ aqk_h,
    bf16* __restrict__ aqk_l,
    float* __restrict__ gC,
    bf16* __restrict__ pieceL,
    float* __restrict__ piecec,
    const VlPiece* __restrict__ pieces,
    int npt,
    int nc_tot,
    const __grid_constant__ CUtensorMap pneg_map,
    const __grid_constant__ CUtensorMap kdt_map,
    const __grid_constant__ CUtensorMap qd_map,
    const __grid_constant__ CUtensorMap aqh_map,
    const __grid_constant__ CUtensorMap aql_map,
    const __grid_constant__ CUtensorMap u0f_map,
    const __grid_constant__ CUtensorMap sl_map,
    const __grid_constant__ CUtensorMap sc_map,
    const float* __restrict__ h0,
    bf16* __restrict__ o,
    float* __restrict__ Sf,
    void* __restrict__ hpc,
    bool hpc_bf16,
    bool hpc_v_first,
    uint32_t* __restrict__ pflags) {
  union FusedSmem {
    K1Smem k1;
    ChainSmem chain;
  };
  __shared__ FusedSmem S;
  const int tid = threadIdx.x, bid = blockIdx.x;
  if (bid < npt * H) {
    const int h = bid / npt, p = bid % npt;
    const VlPiece* pc = pieces + p;
    const int c00 = pc->cbase + piece_c0(pc->sg, pc->nc, pc->np);
    const int njobs = pc->cbase + piece_c0(pc->sg + 1, pc->nc, pc->np) - c00;
    if (pc->tail)
      k1_body<true, GM, RAW, BSIG>(
          S.k1,
          h * nc_tot + c00,
          njobs,
          q,
          kk,
          v,
          glog,
          beta,
          T,
          H,
          scale,
          P,
          u0,
          kdec,
          qdec,
          aqk_h,
          aqk_l,
          gC,
          pieceL,
          piecec,
          p * H + h,
          nc_tot,
          pc->tokoff,
          pc->tend,
          a_log,
          dtb,
          lb,
          &pneg_map);
    else
      k1_body<false, GM, RAW, BSIG>(
          S.k1,
          h * nc_tot + c00,
          njobs,
          q,
          kk,
          v,
          glog,
          beta,
          nc_tot * BT,
          H,
          scale,
          P,
          u0,
          kdec,
          qdec,
          aqk_h,
          aqk_l,
          gC,
          pieceL,
          piecec,
          p * H + h,
          nc_tot,
          pc->tokoff,
          pc->tend,
          a_log,
          dtb,
          lb,
          &pneg_map);
    __syncthreads();
    if (tid == 0) {
      ptx::fence_async_global();
      ptx::red_add_rel_b32(&pflags[p * H + h], 1u);
    }
  } else {
    const int r = bid - npt * H, h = r % H, p = r / H;
    const VlPiece* pc = pieces + p;
    const size_t so = (size_t)pc->seq * H * K * K;
    if (pc->tail)
      chain_body<true>(
          S.chain,
          h,
          pc->sg,
          pc->np,
          pneg_map,
          kdt_map,
          qd_map,
          aqh_map,
          aql_map,
          u0f_map,
          gC,
          pc->nc,
          H,
          o,
          Sf + so,
          hpc,
          hpc_bf16,
          hpc_v_first,
          &sl_map,
          &sc_map,
          h0 + so,
          pflags,
          pc->cbase,
          p - pc->sg,
          pc->tokoff,
          pc->tend);
    else
      chain_body<false>(
          S.chain,
          h,
          pc->sg,
          pc->np,
          pneg_map,
          kdt_map,
          qd_map,
          aqh_map,
          aql_map,
          u0f_map,
          gC,
          pc->nc,
          H,
          o,
          Sf + so,
          hpc,
          hpc_bf16,
          hpc_v_first,
          &sl_map,
          &sc_map,
          h0 + so,
          pflags,
          pc->cbase,
          p - pc->sg,
          pc->tokoff,
          pc->tend);
  }
}






















































static __device__ __forceinline__ int su_c(int r, int c) {
  return ((((c >> 2) + 3 * (r & 1)) & 31) << 2) | (c & 3);
}










static __device__ __forceinline__ constexpr int kdec_r(int r) {
  static_assert(BT == 64, "the map rotates a 6-bit staging row index");
  return (r >> 2) + ((r & 3) << 4);
}
struct K1SmemTF {

  union __align__(128) Pool {
    bf16 sgb[BT][K];
    struct {
      bf16 kw[BT][K + 8];
      bf16 qw[BT][K + 8];
      bf16 kz[6][16][K + 8];
    } a;
    struct {
      bf16 aC_h[BT][56];
      float sU[BT][K + 4];
    } f;
  } sp;
  union __align__(16) GPool {
    float sgc[BT][K];
    bf16 upT_h[K][56];
  } g;
  float tri[4][120];




  alignas(16) bf16 Mh[4][16][24];
  alignas(16) bf16 Ml[4][16][24];
  float sb[BT];
  uint64_t mb_in;
};




struct K1SmemTFRaw : K1SmemTF {
  float2 rn[BT];
};






template <int GM = 0, bool VL = false, bool RAW = false, bool BSIG = false>
__device__ static void k1_tf_body(
    K1SmemTF& S,
    int job0,
    int njobs,
    const bf16* __restrict__ q,
    const bf16* __restrict__ kk,
    const bf16* __restrict__ v,
    const bf16* __restrict__ glog,
    const CUtensorMap& glog_map,
    const float* __restrict__ beta,
    int T,
    int H,
    float scale,
    bf16* __restrict__ P,
    float* __restrict__ u0,
    bf16* __restrict__ kdec,
    bf16* __restrict__ qdec,
    bf16* __restrict__ aqk_h,
    bf16* __restrict__ aqk_l,
    float* __restrict__ gC,

    const float* __restrict__ a_log = nullptr,
    const float* __restrict__ dtb = nullptr,
    float lb = 0.f,


    int tokoff = 0,
    int tend = 0,
    float2* __restrict__ rn = nullptr)
{
  constexpr int SB = 16, NSB = BT / SB;
  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  auto& kw = S.sp.a.kw;
  auto& qw = S.sp.a.qw;
  auto& kz = S.sp.a.kz;
  auto& aC_h = S.sp.f.aC_h;
  float (&sgc)[BT][K] = S.g.sgc;
  auto& upT_h = S.g.upT_h;
  auto& sU = S.sp.f.sU;
  auto& tri = S.tri;
  auto& Mh = S.Mh;
  auto& Ml = S.Ml;
  auto& sb = S.sb;
  const int nc1 = T / BT;
  const int tse = VL ? tend : T;

  const float ga = GM != 0 ? expf(a_log[job0 / nc1]) : 0.f;


  const float dtv_xf = GM != 0 ? dtb[(size_t)(job0 / nc1) * K + (tid & (K - 1))] : 0.f;
  if (tid == 0) {
    ptx::mbar_init(&S.mb_in, 1);
    ptx::prefetch_tensormap(&glog_map);
  }
  __syncthreads();
  for (int sub = 0; sub < njobs; ++sub) {
    const int job = job0 + sub;
    const int c = job % nc1, h = job / nc1, t0 = c * BT + tokoff;
    const int C_act = VL ? min(BT, tse - t0) : BT;
    bf16(&sgb)[BT][K] = S.sp.sgb;
    if (sub) __syncthreads();
    if (tid == 0) {
      ptx::mbar_arrive_expect_tx(&S.mb_in, BT * K * 2);
      ptx::cp_async_bulk_tensor_2d_load(ptx::to_shared(&sgb[0][0]), &glog_map, h * K, t0, &S.mb_in);
    }
    if (tid < BT) sb[tid] = !VL || tid < C_act ? beta_in<BSIG>(beta[(size_t)(t0 + tid) * H + h]) : 0.f;
    if constexpr (RAW)

      row_rnorm<VL>(rn[tid >> 3], q, kk, (size_t)t0 * H * K + (size_t)h * K, H, C_act, tid);
    ptx::mbar_wait_parity(&S.mb_in, sub & 1);
    if constexpr (VL) {



      for (int p = tid; p < (BT - C_act) * K / 8; p += blockDim.x)
        *reinterpret_cast<uint4*>(&sgb[C_act + p * 8 / K][p * 8 % K]) = uint4{0, 0, 0, 0};
    }
    __syncthreads();
    {
      const int col = tid & (K - 1), r0 = (tid >> 7) * (BT / 4);
      float acc = 0.f;
      for (int r = r0; r < r0 + BT / 4; ++r) {


        float gv;
        if constexpr (GM != 0) {
          const float g = __bfloat162float(sgb[r][col]) + dtv_xf;
          const float y =
              GM == 1 ? -ga * (fmaxf(g, 0.f) + __logf(1.f + __expf(-fabsf(g)))) : lb * (1.f / (1.f + __expf(-ga * g)));
          gv = __bfloat162float(__float2bfloat16(y));
        } else {
          gv = __bfloat162float(sgb[r][col]);
        }
        if (VL && r >= C_act) gv = 0.f;
        acc += gv;
        sgc[r][col] = acc;
      }
    }
    __syncthreads();
    {
      const int colc = (tid & (K / 2 - 1)) * 2, rc = tid >> 6;
      for (int seg = 1; seg < 4; ++seg) {
        const float2 carry = *reinterpret_cast<const float2*>(&sgc[seg * (BT / 4) - 1][colc]);
#pragma unroll
        for (int it = 0; it < 2; ++it) {
          float2& v2 = *reinterpret_cast<float2*>(&sgc[seg * (BT / 4) + rc + it * 8][colc]);
          v2 = float2{v2.x + carry.x, v2.y + carry.y};
        }
        __syncthreads();
      }
    }
    const int colp = (tid & (K / 2 - 1)) * 2;
    float a0v[NSB], a1v[NSB];
    float ea0[NSB], ea1[NSB];
    ea0[0] = ea1[0] = 1.f;
    a0v[0] = a1v[0] = 0.f;
#pragma unroll
    for (int s = 1; s < NSB; ++s) {
      a0v[s] = sgc[s * SB - 1][colp];
      a1v[s] = sgc[s * SB - 1][colp + 1];
      ea0[s] = __expf(a0v[s]);
      ea1[s] = __expf(a1v[s]);
    }


    __nv_bfloat162 kvz[NSB][2];
    {
      bf16(*stg)[K + 4] = reinterpret_cast<bf16(*)[K + 4]>(&kz[0][0][0]);
      const float gl0 = sgc[BT - 1][colp], gl1 = sgc[BT - 1][colp + 1];
      const size_t gp0 = (size_t)(t0 + tid * 2 / K) * H * K + h * K + tid * 2 % K;
      const size_t gst = (size_t)8 * H * K;
      const bf16* pk = &kk[gp0];
      const bf16* pq = &q[gp0];
      bf16* qd = &qdec[((size_t)c * H + h) * BT * K + (tid >> 6) * K + colp];



      const int sr0 = kdec_r(tid >> 6), srt = kdec_r(8);
#pragma unroll
      for (int hb = 0; hb < 2; ++hb) {
        float2 kv[4], qv[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int m = hb * 4 + j;
          if (VL && tid * 2 / K + m * 8 >= C_act) {
            kv[j] = qv[j] = float2{0.f, 0.f};
            kvz[m >> 1][m & 1] = __floats2bfloat162_rn(0.f, 0.f);
          } else {
            kvz[m >> 1][m & 1] = *reinterpret_cast<const __nv_bfloat162*>(pk);
            qv[j] = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(pq));
            if constexpr (RAW) {
              const float2 r = rn[(tid >> 6) + m * 8];
              kvz[m >> 1][m & 1] = l2_bf16(__bfloat1622float2(kvz[m >> 1][m & 1]), r.x);
              qv[j] = l2_round(qv[j], r.y);
            }
            kv[j] = __bfloat1622float2(kvz[m >> 1][m & 1]);
          }
          pk += gst;
          pq += gst;
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int i = (tid >> 6) + (hb * 4 + j) * 8;
          const int si = hb * 2 + (j >> 1);
          const float ei0 = __expf(sgc[i][colp] - a0v[si]);
          const float ei1 = __expf(sgc[i][colp + 1] - a1v[si]);
          const float kw0 = kv[j].x * ei0, kw1 = kv[j].y * ei1;
          *reinterpret_cast<__nv_bfloat162*>(&kw[i][colp]) = __floats2bfloat162_rn(kw0, kw1);
          const float qw0 = qv[j].x * ei0 * scale;
          const float qw1 = qv[j].y * ei1 * scale;
          *reinterpret_cast<__nv_bfloat162*>(&qw[i][colp]) = __floats2bfloat162_rn(qw0, qw1);
          *reinterpret_cast<__nv_bfloat162*>(qd) = __floats2bfloat162_rn(qw0 * ea0[si], qw1 * ea1[si]);
          qd += 8 * K;
          *reinterpret_cast<__nv_bfloat162*>(&stg[sr0 + srt * (hb * 4 + j)][colp]) =
              __floats2bfloat162_rn(kv[j].x * __expf(gl0 - sgc[i][colp]), kv[j].y * __expf(gl1 - sgc[i][colp + 1]));
        }
      }
      __syncthreads();

      const int rr = tid * 4 % BT, cc = tid * 4 / BT;
      bf16* kd = &kdec[((size_t)c * H + h) * BT * K + tid * 4];
#pragma unroll
      for (int n = 0; n < BT * K / 2048; ++n) {
        const int cn = cc + n * 32;
        alignas(8) const __nv_bfloat162 p2[2] = {
            {stg[kdec_r(rr)][cn], stg[kdec_r(rr + 1)][cn]}, {stg[kdec_r(rr + 2)][cn], stg[kdec_r(rr + 3)][cn]}};
        *reinterpret_cast<float2*>(kd + n * 2048) = *reinterpret_cast<const float2*>(p2);
      }
    }
    __syncthreads();

    const int u = tid >> 6;
    float f0[NSB - 1], f1[NSB - 1];
#pragma unroll
    for (int s = 1; s < NSB; ++s) {
      f0[s - 1] = __expf(a0v[s] - a0v[s - 1]);
      f1[s - 1] = __expf(a1v[s] - a1v[s - 1]);
    }
    {
#pragma unroll
      for (int sj = 0; sj < NSB; ++sj)
#pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
          const int j = sj * SB + u + rr * 8, jj = u + rr * 8;
          const float2 kf = __bfloat1622float2(kvz[sj][rr]);


          const bool real = !VL || j < C_act;
          const float v0 = real ? kf.x * __expf(a0v[sj] - sgc[j][colp]) : 0.f;
          const float v1 = real ? kf.y * __expf(a1v[sj] - sgc[j][colp + 1]) : 0.f;
          *reinterpret_cast<__nv_bfloat162*>(&kz[sj][jj][colp]) = __floats2bfloat162_rn(v0, v1);
        }
    }
    if (tid < K / 4) {
      const float4 g4 = *reinterpret_cast<const float4*>(&sgc[BT - 1][tid * 4]);
      *reinterpret_cast<float4*>(&gC[((size_t)c * H + h) * K + tid * 4]) =
          float4{__expf(g4.x), __expf(g4.y), __expf(g4.z), __expf(g4.w)};
    }
    __syncthreads();
    if (warp >= 10) {
      const int jn = job + 1;
      if (jn < nc1 * H && sub + 1 < njobs) {
        const int t1p = (jn % nc1) * BT + tokoff;
        const int Cp = VL ? min(BT, tse - t1p) : BT;
        const size_t tb = (size_t)t1p * H * K + (size_t)(jn / nc1) * K;
        const char* pq = reinterpret_cast<const char*>(q + tb);
        const char* pk = reinterpret_cast<const char*>(kk + tb);
        const char* pv = reinterpret_cast<const char*>(v + tb);
        const char* pg = reinterpret_cast<const char*>(glog + tb);
        for (int i = tid - 320; i < Cp; i += 192) {
          const size_t ro = (size_t)i * H * K * 2;
          for (int l = 0; l < 256; l += 128) {
            asm volatile("prefetch.global.L2 [%0];" ::"l"(pq + ro + l));
            asm volatile("prefetch.global.L2 [%0];" ::"l"(pk + ro + l));
            asm volatile("prefetch.global.L2 [%0];" ::"l"(pv + ro + l));
            asm volatile("prefetch.global.L2 [%0];" ::"l"(pg + ro + l));
          }
        }
      }
    }



    auto pair_mma = [&](int si2, int kzslot, float4* acck, float4* accq) {
      acck[0] = acck[1] = accq[0] = accq[1] = float4{0, 0, 0, 0};
      const int arow = lane & 15, aka = lane >> 4;
#pragma unroll
      for (int k16 = 0; k16 < K / 16; ++k16) {
        uint32_t a0, a1, a2, a3, q0, q1, q2, q3;
        ptx::ldmatrix_x4_b16(ptx::to_shared(&kw[si2 * SB + arow][k16 * 16 + aka * 8]), a0, a1, a2, a3);
        ptx::ldmatrix_x4_b16(ptx::to_shared(&qw[si2 * SB + arow][k16 * 16 + aka * 8]), q0, q1, q2, q3);
#pragma unroll
        for (int n8 = 0; n8 < 2; ++n8) {
          uint32_t b0, b1;
          ptx::ldmatrix_x2_b16(
              ptx::to_shared(&kz[kzslot][(lane & 7) + n8 * 8][k16 * 16 + ((lane >> 3) & 1) * 8]), b0, b1);
          ptx::mma_m16n8k16_bf16f32(acck[n8], a0, a1, a2, a3, b0, b1);
          ptx::mma_m16n8k16_bf16f32(accq[n8], q0, q1, q2, q3, b0, b1);
        }
      }
    };










    auto aqk_scatter = [&](int si2, int sj2, const float4* accq) {
      const int r = lane >> 2, c2 = (lane & 3) * 2;
      const size_t abase = ((size_t)c * H + h) * BT * BT;
      const int odd = lane & 1, jj = sj2 * SB + c2 + 6 * odd;
#pragma unroll
      for (int e2 = 0; e2 < 2; ++e2) {
        const int ii = si2 * SB + r + e2 * 8, lim = ii - jj;
        const float m0 = e2 ? accq[0].z : accq[0].x;
        const float m1 = e2 ? accq[0].w : accq[0].y;
        const float p0 = e2 ? accq[1].z : accq[1].x;
        const float p1 = e2 ? accq[1].w : accq[1].y;

        const float s0 = __shfl_xor_sync(0xffffffffu, odd ? m0 : p0, 1);
        const float s1 = __shfl_xor_sync(0xffffffffu, odd ? m1 : p1, 1);
        float v[4] = {odd ? s0 : m0, odd ? s1 : m1, odd ? p0 : s0, odd ? p1 : s1};
        if (lim >= 0) {
#pragma unroll
          for (int e = 1; e < 4; ++e)
            if (lim < e) v[e] = 0.f;
          const __nv_bfloat162 ah[2] = {
              {__float2bfloat16(v[0]), __float2bfloat16(v[1])}, {__float2bfloat16(v[2]), __float2bfloat16(v[3])}};
          *reinterpret_cast<float2*>(&aqk_h[abase + ii * BT + jj]) = *reinterpret_cast<const float2*>(ah);
          const __nv_bfloat162 al[2] = {
              __floats2bfloat162_rn(v[0] - __bfloat162float(ah[0].x), v[1] - __bfloat162float(ah[0].y)),
              __floats2bfloat162_rn(v[2] - __bfloat162float(ah[1].x), v[3] - __bfloat162float(ah[1].y))};
          *reinterpret_cast<float2*>(&aqk_l[abase + ii * BT + jj]) = *reinterpret_cast<const float2*>(al);
        }
      }
    };
    auto tri_scatter = [&](int si2, const float4* acck) {
      const int r = lane >> 2, c2 = (lane & 3) * 2;
#pragma unroll
      for (int n8 = 0; n8 < 2; ++n8) {
        const float vals[4] = {acck[n8].x, acck[n8].y, acck[n8].z, acck[n8].w};
#pragma unroll
        for (int e2 = 0; e2 < 2; ++e2) {
          const int il = r + e2 * 8, jl = n8 * 8 + c2;
          const int ii = si2 * SB + il, tb = il * (il - 1) / 2;
          if (jl < il) tri[si2][tb + jl] = sb[ii] * vals[e2 * 2];
          if (jl + 1 < il) tri[si2][tb + jl + 1] = sb[ii] * vals[e2 * 2 + 1];
        }
      }
    };
    float4 offk[2];
    int offsi = -1, offsj = -1;
    if (warp < NSB) {
      float4 acck[2], accq[2];
      pair_mma(warp, warp, acck, accq);
      aqk_scatter(warp, warp, accq);
      tri_scatter(warp, acck);
    }
    __syncthreads();
    {
#pragma unroll
      for (int sj = 0; sj < NSB - 1; ++sj)
#pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
          const int j = sj * SB + u + rr * 8, jj = u + rr * 8;
          const float2 kf = __bfloat1622float2(kvz[sj][rr]);
          const bool real = !VL || j < C_act;
          float v0 = real ? kf.x * __expf(a0v[sj] - sgc[j][colp]) : 0.f;
          float v1 = real ? kf.y * __expf(a1v[sj] - sgc[j][colp + 1]) : 0.f;
#pragma unroll
          for (int si = sj + 1; si < NSB; ++si) {
            v0 *= f0[si - 1];
            v1 *= f1[si - 1];
            *reinterpret_cast<__nv_bfloat162*>(&kz[si * (si - 1) / 2 + sj][jj][colp]) = __floats2bfloat162_rn(v0, v1);
          }
        }
    }
    __syncthreads();





    constexpr int MBW = NSB * (NSB - 1) / 2;
    if (tid >= MBW * 32 && tid < MBW * 32 + NSB * SB) {
      const int td = tid - MBW * 32;
      const int b = td >> 4, j = td & 15;
      float m[SB];
#pragma unroll
      for (int i = 0; i < SB; ++i) {
        float s = i == j ? 1.f : 0.f;

#pragma unroll
        for (int k2 = 0; k2 < i; ++k2)
          s -= tri[b][i * (i - 1) / 2 + k2] * m[k2];
        m[i] = s;
      }
#pragma unroll
      for (int i = 0; i < SB; ++i) {
        const bf16 hi = __float2bfloat16(m[i]);
        Mh[b][i][j] = hi;
        Ml[b][i][j] = __float2bfloat16(m[i] - __bfloat162float(hi));
      }
    }
    if (warp < NSB * (NSB - 1) / 2) {
      int si2 = 1;
      while (si2 * (si2 + 1) / 2 <= warp)
        ++si2;
      const int sj2 = warp - si2 * (si2 - 1) / 2;
      float4 acck[2], accq[2];
      pair_mma(si2, warp, acck, accq);
      aqk_scatter(si2, sj2, accq);
      offk[0] = acck[0];
      offk[1] = acck[1];
      offsi = si2;
      offsj = sj2;
    }
    __syncthreads();

    if (offsi >= 0) {
      const int r = lane >> 2, c2 = (lane & 3) * 2;
#pragma unroll
      for (int n8 = 0; n8 < 2; ++n8) {
        const float vals[4] = {offk[n8].x, offk[n8].y, offk[n8].z, offk[n8].w};
#pragma unroll
        for (int e2 = 0; e2 < 2; ++e2) {
          const int ii = offsi * SB + r + e2 * 8;
          const int jj = offsj * SB + n8 * 8 + c2;
          *reinterpret_cast<__nv_bfloat162*>(&aC_h[ii][jj]) =
              __floats2bfloat162_rn(sb[ii] * vals[e2 * 2], sb[ii] * vals[e2 * 2 + 1]);
        }
      }
    }
    for (int p2 = tid; p2 < BT * 24; p2 += blockDim.x) {
      const int p = p2 * 2;
      const int i = p / 48, j = p % 48;
      if (j / SB < i / SB) continue;
      float2 av{0.f, 0.f};
      if (j / SB == i / SB) {
        const int il = i % SB, jl = j % SB, tb = il * (il - 1) / 2;
        if (jl < il) av.x = tri[i / SB][tb + jl];
        if (jl + 1 < il) av.y = tri[i / SB][tb + jl + 1];
      }
      *reinterpret_cast<__nv_bfloat162*>(&aC_h[i][j]) = __floats2bfloat162_rn(av.x, av.y);
    }
    __syncthreads();
    const size_t base = ((size_t)c * H + h) * BT * K;



    auto rhs_fill = [&](int pane) {
      const size_t gp0 = (size_t)(t0 + tid * 2 / K) * H * K + h * K + tid * 2 % K;
      const size_t gst = (size_t)8 * H * K;
      const bf16* px = &v[gp0];
#pragma unroll
      for (int hb = 0; hb < 2; ++hb) {
        __nv_bfloat162 xr[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int m = hb * 4 + j;
          xr[j] = pane == 0 ? kvz[m >> 1][m & 1]
                            : (!VL || tid * 2 / K + m * 8 < C_act ? *reinterpret_cast<const __nv_bfloat162*>(px)
                                                                  : __floats2bfloat162_rn(0.f, 0.f));
          px += gst;
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int i = (tid >> 6) + (hb * 4 + j) * 8;
          const int si = hb * 2 + (j >> 1);
          const float2 xv = __bfloat1622float2(xr[j]);
          const float bi = sb[i];
          if (pane == 0) {
            const float ei0 = __expf(sgc[i][colp] - a0v[si]);
            const float ei1 = __expf(sgc[i][colp + 1] - a1v[si]);
            const float kw0 = xv.x * ei0, kw1 = xv.y * ei1;
            *reinterpret_cast<float2*>(&sU[i][su_c(i, colp)]) = float2{kw0 * (bi * ea0[si]), kw1 * (bi * ea1[si])};
          } else {
            *reinterpret_cast<float2*>(&sU[i][su_c(i, colp)]) = float2{xv.x * bi, xv.y * bi};
          }
        }
      }
    };
#pragma unroll 1
    for (int pane = 0; pane < 2; ++pane) {
      rhs_fill(pane);
      __syncthreads();
      auto pu_store4p = [&](int p) {
        const float* r4 = &sU[p / K][su_c(p / K, p % K)];
        if (pane == 0) {
          const __nv_bfloat162 h01 = __floats2bfloat162_rn(-r4[0], -r4[1]);
          const __nv_bfloat162 h23 = __floats2bfloat162_rn(-r4[2], -r4[3]);
          *reinterpret_cast<uint2*>(&P[base + p]) =
              uint2{*reinterpret_cast<const uint32_t*>(&h01), *reinterpret_cast<const uint32_t*>(&h23)};
        } else {
          *reinterpret_cast<float4*>(&u0[base + p]) = *reinterpret_cast<const float4*>(r4);
        }
      };





#pragma unroll
      for (int b = 0; b < NSB; ++b) {
        float crr[4] = {};
        if (b) {
          const int arow = lane & 15, aka = lane >> 4;
          float4 acc = float4{0, 0, 0, 0};
          for (int k16 = 0; k16 < b; ++k16) {
            uint32_t a0, a1, a2, a3, b0, b1;
            ptx::ldmatrix_x4_b16(ptx::to_shared(&aC_h[b * SB + arow][k16 * 16 + aka * 8]), a0, a1, a2, a3);
            ptx::ldmatrix_x2_b16(
                ptx::to_shared(&upT_h[warp * 8 + (lane & 7)][k16 * 16 + ((lane >> 3) & 1) * 8]), b0, b1);
            ptx::mma_m16n8k16_bf16f32(acc, a0, a1, a2, a3, b0, b1);
          }






          const int fr = lane >> 2;
          const int sla = (lane & 3) * 8 + (fr >> 1), slb = sla + 4;
          const unsigned fm = 0xffffffffu;
          const float ax = __shfl_sync(fm, acc.x, sla);
          const float ay = __shfl_sync(fm, acc.y, sla);
          const float az = __shfl_sync(fm, acc.z, sla);
          const float aw = __shfl_sync(fm, acc.w, sla);
          const float bx = __shfl_sync(fm, acc.x, slb);
          const float by = __shfl_sync(fm, acc.y, slb);
          const float bz = __shfl_sync(fm, acc.z, slb);
          const float bw = __shfl_sync(fm, acc.w, slb);
          crr[0] = (fr & 1) ? ay : ax;
          crr[1] = (fr & 1) ? by : bx;
          crr[2] = (fr & 1) ? aw : az;
          crr[3] = (fr & 1) ? bw : bz;
        }
        {


          const int arow = lane & 15, aka = lane >> 4;
          const int fr = lane >> 2, fc = (lane & 3) * 2;
          const int bcol = warp * 8 + fr;
          uint32_t ah[4], al[4], bh[2], bl[2];
          ptx::ldmatrix_x4_b16(ptx::to_shared(&Mh[b][arow][aka * 8]), ah[0], ah[1], ah[2], ah[3]);
          ptx::ldmatrix_x4_b16(ptx::to_shared(&Ml[b][arow][aka * 8]), al[0], al[1], al[2], al[3]);
#pragma unroll
          for (int kp = 0; kp < 2; ++kp) {
            const int rr0 = b * SB + fc + kp * 8;
            float r0 = sU[rr0][su_c(rr0, bcol)];
            float r1 = sU[rr0 + 1][su_c(rr0 + 1, bcol)];
            if (b) {
              r0 -= crr[kp * 2];
              r1 -= crr[kp * 2 + 1];
            }
            const __nv_bfloat162 hv = __floats2bfloat162_rn(r0, r1);
            const __nv_bfloat162 lv = __floats2bfloat162_rn(r0 - __bfloat162float(hv.x), r1 - __bfloat162float(hv.y));
            bh[kp] = *reinterpret_cast<const uint32_t*>(&hv);
            bl[kp] = *reinterpret_cast<const uint32_t*>(&lv);
          }
          float4 acc = float4{0, 0, 0, 0};
          ptx::mma_m16n8k16_bf16f32(acc, ah[0], ah[1], ah[2], ah[3], bh[0], bh[1]);
          ptx::mma_m16n8k16_bf16f32(acc, al[0], al[1], al[2], al[3], bh[0], bh[1]);
          ptx::mma_m16n8k16_bf16f32(acc, ah[0], ah[1], ah[2], ah[3], bl[0], bl[1]);
          __syncwarp();
          const float vals[4] = {acc.x, acc.y, acc.z, acc.w};
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int xr = b * SB + fr + (e >> 1) * 8;
            const int xc = warp * 8 + fc + (e & 1);
            if (!(e & 1))
              *reinterpret_cast<float2*>(&sU[xr][su_c(xr, xc)]) = float2{vals[e], vals[e + 1]};
            if (b + 1 < NSB)
              upT_h[xc][xr] = __float2bfloat16(vals[e]);
          }
          __syncwarp();
        }
      }


      __syncthreads();
      for (int p = tid * 4; p < BT * K; p += blockDim.x * 4)
        pu_store4p(p);
      __syncthreads();
    }
  }
}




template <int GM = 0, bool RAW = false, bool BSIG = false>
__global__ void __launch_bounds__(512, 2) k1_tf_builder(
    const bf16* __restrict__ q,
    const bf16* __restrict__ kk,
    const bf16* __restrict__ v,
    const bf16* __restrict__ glog,
    const __grid_constant__ CUtensorMap glog_map,
    const float* __restrict__ a_log,
    const float* __restrict__ dtb,
    float lb,
    const float* __restrict__ beta,
    int T,
    int H,
    float scale,
    bf16* __restrict__ P,
    float* __restrict__ u0,
    bf16* __restrict__ kdec,
    bf16* __restrict__ qdec,
    bf16* __restrict__ aqk_h,
    bf16* __restrict__ aqk_l,
    float* __restrict__ gC,
    int NP) {
  __shared__ std::conditional_t<RAW, K1SmemTFRaw, K1SmemTF> S;
  float2* rn = nullptr;
  if constexpr (RAW) rn = S.rn;
  const int nc1 = T / BT, bid = blockIdx.x;
  const int h = bid % H, pc = bid / H;
  const int c00 = piece_c0(pc, nc1, NP);
  const int njobs = piece_c0(pc + 1, nc1, NP) - c00;
  k1_tf_body<GM, false, RAW, BSIG>(
      S,
      h * nc1 + c00,
      njobs,
      q,
      kk,
      v,
      glog,
      glog_map,
      beta,
      T,
      H,
      scale,
      P,
      u0,
      kdec,
      qdec,
      aqk_h,
      aqk_l,
      gC,
      a_log,
      dtb,
      lb,
      0,
      0,
      rn);
}
template <int GM = 0, bool RAW = false, bool BSIG = false>
__global__ void __launch_bounds__(512, 2) k1_tf_builder_vl(
    const bf16* __restrict__ q,
    const bf16* __restrict__ kk,
    const bf16* __restrict__ v,
    const bf16* __restrict__ glog,
    const __grid_constant__ CUtensorMap glog_map,
    const float* __restrict__ a_log,
    const float* __restrict__ dtb,
    float lb,
    const float* __restrict__ beta,
    int H,
    float scale,
    bf16* __restrict__ P,
    float* __restrict__ u0,
    bf16* __restrict__ kdec,
    bf16* __restrict__ qdec,
    bf16* __restrict__ aqk_h,
    bf16* __restrict__ aqk_l,
    float* __restrict__ gC,
    const VlPiece* __restrict__ pieces,
    int nc_tot) {
  __shared__ std::conditional_t<RAW, K1SmemTFRaw, K1SmemTF> S;
  float2* rn = nullptr;
  if constexpr (RAW) rn = S.rn;
  const int h = blockIdx.x % H;
  const VlPiece* pc = pieces + blockIdx.x / H;
  const int c00 = pc->cbase + piece_c0(pc->sg, pc->nc, pc->np);
  const int njobs = pc->cbase + piece_c0(pc->sg + 1, pc->nc, pc->np) - c00;
  if (pc->tail)
    k1_tf_body<GM, true, RAW, BSIG>(
        S,
        h * nc_tot + c00,
        njobs,
        q,
        kk,
        v,
        glog,
        glog_map,
        beta,
        nc_tot * BT,
        H,
        scale,
        P,
        u0,
        kdec,
        qdec,
        aqk_h,
        aqk_l,
        gC,
        a_log,
        dtb,
        lb,
        pc->tokoff,
        pc->tend,
        rn);
  else
    k1_tf_body<GM, false, RAW, BSIG>(
        S,
        h * nc_tot + c00,
        njobs,
        q,
        kk,
        v,
        glog,
        glog_map,
        beta,
        nc_tot * BT,
        H,
        scale,
        P,
        u0,
        kdec,
        qdec,
        aqk_h,
        aqk_l,
        gC,
        a_log,
        dtb,
        lb,
        pc->tokoff,
        pc->tend,
        rn);
}



__global__ void __launch_bounds__(512) k2_chain_tc_vl(
    const __grid_constant__ CUtensorMap pneg_map,
    const __grid_constant__ CUtensorMap kdt_map,
    const __grid_constant__ CUtensorMap qd_map,
    const __grid_constant__ CUtensorMap aqh_map,
    const __grid_constant__ CUtensorMap aql_map,
    const __grid_constant__ CUtensorMap u0f_map,
    const float* __restrict__ gC,
    const float* __restrict__ h0,
    int H,
    bf16* __restrict__ o,
    float* __restrict__ Sf,
    void* __restrict__ hpc,
    bool hpc_bf16,
    bool hpc_v_first,
    const VlPiece* __restrict__ seqs) {
  __shared__ ChainSmem S;
  const int s = blockIdx.x / H, h = blockIdx.x % H;
  const VlPiece* sq = seqs + s;
  const size_t so = (size_t)s * H * K * K;
  if (sq->tail)
    chain_body<true>(
        S,
        h,
        0,
        1,
        pneg_map,
        kdt_map,
        qd_map,
        aqh_map,
        aql_map,
        u0f_map,
        gC,
        sq->nc,
        H,
        o,
        Sf + so,
        hpc,
        hpc_bf16,
        hpc_v_first,
        nullptr,
        nullptr,
        h0 + so,
        nullptr,
        sq->cbase,
        0,
        sq->tokoff,
        sq->tend);
  else
    chain_body<false>(
        S,
        h,
        0,
        1,
        pneg_map,
        kdt_map,
        qd_map,
        aqh_map,
        aql_map,
        u0f_map,
        gC,
        sq->nc,
        H,
        o,
        Sf + so,
        hpc,
        hpc_bf16,
        hpc_v_first,
        nullptr,
        nullptr,
        h0 + so,
        nullptr,
        sq->cbase,
        0,
        sq->tokoff,
        sq->tend);
}



#define KDA_CU_CHECK(expr)                                                                             \
  do {                                                                                                 \
    CUresult _e = (expr);                                                                              \
    if (_e != CUDA_SUCCESS) {                                                                          \
      const char* _s = nullptr;                                                                        \
      cuGetErrorString(_e, &_s);                                                                       \
      TORCH_CHECK(false, "CUDA driver call failed at ", __FILE__, ":", __LINE__, ": ", _s ? _s : "?"); \
    }                                                                                                  \
  } while (0)


static CUtensorMap enc2d(void* ptr, uint64_t rows, uint64_t cols, uint32_t brows, uint32_t bcols) {
  cuuint64_t gdim[2] = {cols, rows};
  cuuint64_t gstr[1] = {cols * 2};
  cuuint32_t bdim[2] = {bcols, brows};
  cuuint32_t estr[2] = {1, 1};
  CUtensorMap m{};
  KDA_CU_CHECK(cuTensorMapEncodeTiled(
      &m,
      CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
      2,
      ptr,
      gdim,
      gstr,
      bdim,
      estr,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return m;
}



static CUtensorMap enc2df(void* ptr, uint64_t rows, uint64_t cols) {
  cuuint64_t gdim[2] = {cols, rows};
  cuuint64_t gstr[1] = {cols * 4};
  cuuint32_t bdim[2] = {8, 128};
  cuuint32_t estr[2] = {1, 1};
  CUtensorMap m{};
  KDA_CU_CHECK(cuTensorMapEncodeTiled(
      &m,
      CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
      2,
      ptr,
      gdim,
      gstr,
      bdim,
      estr,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_32B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return m;
}



static CUtensorMap enc2dfn(void* ptr, uint64_t rows, uint64_t cols) {
  cuuint64_t gdim[2] = {cols, rows};
  cuuint64_t gstr[1] = {cols * 4};
  cuuint32_t bdim[2] = {(cuuint32_t)cols, BT};
  cuuint32_t estr[2] = {1, 1};
  CUtensorMap m{};
  KDA_CU_CHECK(cuTensorMapEncodeTiled(
      &m,
      CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
      2,
      ptr,
      gdim,
      gstr,
      bdim,
      estr,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return m;
}




static CUtensorMap enc2dgb(void* ptr, uint64_t rows, uint64_t cols) {
  cuuint64_t gdim[2] = {cols, rows};
  cuuint64_t gstr[1] = {cols * 2};
  cuuint32_t bdim[2] = {K, BT};
  cuuint32_t estr[2] = {1, 1};
  CUtensorMap m{};
  KDA_CU_CHECK(cuTensorMapEncodeTiled(
      &m,
      CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
      2,
      ptr,
      gdim,
      gstr,
      bdim,
      estr,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return m;
}












struct Workspace {
  torch::Tensor P, u0, kdec, qdec, aqk_h, aqk_l, gC, pieceL, piecec, pflags;
  torch::Tensor h0z;
  torch::Tensor pieces_dev;
  std::vector<int64_t> cu;
  CUtensorMap pneg_map, kdt_map, qd_map, aqh_map, aql_map, sl_map, sc_map, u0f_map, gin_map;
  const void* gin_ptr = nullptr;
};




static int64_t builder_slots(bool tf, bool varlen) {
  auto query = [](auto fn) {
    int occ = 0;
    AT_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, 512, 0));
    return int64_t(at::cuda::getCurrentDeviceProperties()->multiProcessorCount) * std::max(1, occ);
  };
  static const int64_t slots[4] = {
      query(kda_fused<0>), query(kda_fused_vl<0>), query(k1_tf_builder<0>), query(k1_tf_builder_vl<0>)};
  return slots[2 * int(tf) + int(varlen)];
}


static std::pair<int64_t, int64_t> piece_plan(const std::vector<int64_t>& cu, int64_t T, int npb, bool varlen) {
  if (!varlen) {
    const int64_t nc = T / BT;
    const int64_t np = std::min<int64_t>(npb, std::max<int64_t>(1, nc / 2));
    return {np, (nc + np - 1) / np};
  }
  int nc_tot = 0;
  for (size_t s = 0; s + 1 < cu.size(); ++s)
    nc_tot += int((cu[s + 1] - cu[s] + BT - 1) / BT);
  int64_t pieces = 0, mx = 0;
  for (size_t s = 0; s + 1 < cu.size(); ++s) {
    const int ncs = int((cu[s + 1] - cu[s] + BT - 1) / BT);
    const int np = std::min(std::min(npb, std::max(1, ncs / 2)), std::max(1, (2 * npb * ncs + nc_tot) / (2 * nc_tot)));
    pieces += np;
    mx = std::max<int64_t>(mx, (ncs + np - 1) / np);
  }
  return {pieces, mx};
}











static int piece_budget(const std::vector<int64_t>& cu, int64_t T, int64_t H, bool varlen, int64_t slots) {
  int64_t best_cost = -1, best_pieces = 0;
  int best_npb = 1;
  for (int npb = 1; npb <= 64; ++npb) {
    const auto [pieces, mx] = piece_plan(cu, T, npb, varlen);
    const int64_t builders = pieces * H;
    const int64_t waves = (builders + slots - 1) / slots;
    const int64_t cost = waves * mx * slots + builders;
    if (best_cost < 0 || cost < best_cost || (cost == best_cost && pieces < best_pieces)) {
      best_cost = cost;
      best_pieces = pieces;
      best_npb = npb;
    }
  }
  return best_npb;
}








constexpr double C_JOB_TF = 10.2, C_JOB_FUSED = 16.2, C_STEP = 2.1;















struct Route {
  bool seq0;
  int npb;
};
static Route pick_route(const std::vector<int64_t>& cu, int64_t T, int64_t H, bool varlen) {
  const double sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  int64_t nc_tot = 0, depth = 0;
  if (varlen)
    for (size_t s = 0; s + 1 < cu.size(); ++s) {
      const int64_t ncs = (cu[s + 1] - cu[s] + BT - 1) / BT;
      nc_tot += ncs;
      depth = std::max(depth, ncs);
    }
  else
    nc_tot = depth = T / BT;
  const double jobs = double(nc_tot * H) / sms;
  struct Arm {
    int npb;
    int64_t pieces;
    double wall;
  };
  auto arm = [&](bool tf) {
    const int64_t slots = builder_slots(tf, varlen);
    const int npb = piece_budget(cu, T, H, varlen, slots);
    const auto [pieces, mx] = piece_plan(cu, T, npb, varlen);
    const double c_job = tf ? C_JOB_TF : C_JOB_FUSED;
    const double share =
        std::min(double(slots) / sms, std::max(1.0, double(pieces * H) / sms));
    const double build = std::max(jobs * (tf ? c_job : c_job + C_STEP), double(mx) * share * c_job);
    const double chain = (tf ? std::max(double(depth), jobs) : double(mx)) * C_STEP;
    return Arm{npb, pieces, build + chain};
  };
  const Arm tf = arm(true), fused = arm(false);
  if (tf.pieces > 1 && tf.wall <= 0.975 * fused.wall) return {true, tf.npb};
  return {false, fused.npb};
}

static Workspace&
get_workspace(const torch::Device& dev, int64_t T, int64_t nc, int64_t H, int64_t npieces, int64_t N) {

  static std::map<std::array<int64_t, 6>, Workspace> cache;
  const std::array<int64_t, 6> key{dev.index(), T, nc, H, npieces, N};
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  Workspace ws;
  const auto ob = torch::TensorOptions().dtype(torch::kBFloat16).device(dev);
  const auto of = torch::TensorOptions().dtype(torch::kFloat).device(dev);
  const auto oi = torch::TensorOptions().dtype(torch::kInt).device(dev);
  const int64_t chk = nc * H * BT * K;
  ws.P = torch::empty({chk}, ob);
  ws.u0 = torch::empty({chk}, of);
  ws.kdec = torch::empty({chk}, ob);
  ws.qdec = torch::empty({chk}, ob);
  ws.aqk_h = torch::zeros({nc * H * BT * BT}, ob);
  ws.aqk_l = torch::zeros({nc * H * BT * BT}, ob);
  ws.gC = torch::empty({nc * H * K}, of);


  ws.pieceL = torch::empty({npieces * H * K * K}, ob);
  ws.piecec = torch::empty({npieces * H * K * K}, of);
  ws.pflags = torch::empty({npieces * H}, oi);
  ws.pneg_map = enc2d(ws.P.data_ptr(), (uint64_t)(nc * H * BT), K, BT, 64);
  ws.kdt_map = enc2d(ws.kdec.data_ptr(), (uint64_t)(nc * H * K), BT, K, 64);
  ws.qd_map = enc2d(ws.qdec.data_ptr(), (uint64_t)(nc * H * BT), K, BT, 64);
  ws.aqh_map = enc2d(ws.aqk_h.data_ptr(), (uint64_t)(nc * H * BT), BT, BT, 64);
  ws.aql_map = enc2d(ws.aqk_l.data_ptr(), (uint64_t)(nc * H * BT), BT, BT, 64);
  ws.sl_map = enc2d(ws.pieceL.data_ptr(), (uint64_t)(npieces * H * K), K, K, 64);
  ws.sc_map = enc2df(ws.piecec.data_ptr(), (uint64_t)(npieces * H * K), K);
  ws.u0f_map = enc2dfn(ws.u0.data_ptr(), (uint64_t)(nc * H * BT), K);
  return cache.emplace(key, std::move(ws)).first->second;
}

static_assert(sizeof(VlPiece) == 8 * sizeof(int));



























static std::tuple<torch::Tensor, torch::Tensor> kda_prefill_fwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& g,
    const torch::Tensor& beta,
    double scale,
    const std::optional<torch::Tensor>& initial_state,
    const std::optional<torch::Tensor>& cu_seqlens,
    bool use_gate_in_kernel,
    const std::optional<torch::Tensor>& A_log,
    const std::optional<torch::Tensor>& dt_bias,
    bool safe_gate,
    double lower_bound,
    bool use_qk_l2norm_in_kernel,
    bool use_beta_sigmoid_in_kernel,
    const std::optional<torch::Tensor>& h_per_chunk,
    bool h_v_first) {
  TORCH_CHECK(q.is_cuda() && q.dim() == 3 && q.size(2) == K, "q must be a CUDA [T, H, 128] tensor, got ", q.sizes());
  const int64_t T = q.size(0), H = q.size(1);
  TORCH_CHECK(T >= 1 && H >= 1, "empty input: T=", T, " H=", H);
  for (auto* t : {&q, &k, &v, &g}) {
    TORCH_CHECK(
        t->is_cuda() && t->is_contiguous() && t->scalar_type() == torch::kBFloat16 && t->sizes() == q.sizes(),
        "q/k/v/g must be contiguous CUDA bf16 [T, H, 128] with "
        "matching shapes; got ",
        t->sizes(),
        " ",
        t->dtype());
  }
  TORCH_CHECK(
      beta.is_cuda() && beta.is_contiguous() && beta.dim() == 2 && beta.size(0) == T && beta.size(1) == H,
      "beta must be contiguous CUDA [T, H], got ",
      beta.sizes());
  TORCH_CHECK(
      beta.scalar_type() == torch::kBFloat16 || beta.scalar_type() == torch::kFloat,
      "beta must be bf16 or fp32, got ",
      beta.dtype());
  const int gm = use_gate_in_kernel ? (safe_gate ? 2 : 1) : 0;
  const float lb = float(lower_bound);
  const float* alog_p = nullptr;
  const float* dtb_p = nullptr;
  if (use_gate_in_kernel) {
    TORCH_CHECK(A_log && dt_bias, "use_gate_in_kernel=True requires A_log and dt_bias");
    TORCH_CHECK(
        A_log->is_cuda() && A_log->is_contiguous() && A_log->scalar_type() == torch::kFloat && A_log->numel() == H,
        "A_log must be contiguous CUDA fp32 [H], got ",
        A_log->sizes(),
        " ",
        A_log->dtype());
    TORCH_CHECK(
        dt_bias->is_cuda() && dt_bias->is_contiguous() && dt_bias->scalar_type() == torch::kFloat &&
            dt_bias->numel() == H * K,
        "dt_bias must be contiguous CUDA fp32 with H*128 "
        "elements, got ",
        dt_bias->sizes(),
        " ",
        dt_bias->dtype());
    alog_p = A_log->data_ptr<float>();
    dtb_p = dt_bias->data_ptr<float>();
  }


  std::vector<int64_t> cu;
  if (cu_seqlens) {
    TORCH_CHECK(
        cu_seqlens->dim() == 1 && cu_seqlens->numel() >= 2, "cu_seqlens must be 1-D [N+1], got ", cu_seqlens->sizes());
    const auto cu_cpu =
        cu_seqlens->to(torch::kCPU).to(torch::kLong).contiguous();
    const int64_t* p = cu_cpu.data_ptr<int64_t>();
    cu.assign(p, p + cu_cpu.numel());
    TORCH_CHECK(
        cu.front() == 0 && cu.back() == T,
        "cu_seqlens must span [0, T=",
        T,
        "], got [",
        cu.front(),
        ", ",
        cu.back(),
        "]");
    for (size_t s = 0; s + 1 < cu.size(); ++s)
      TORCH_CHECK(
          cu[s] < cu[s + 1],
          "cu_seqlens must be strictly increasing (empty "
          "sequences unsupported) at index ",
          s);
  } else if (T % BT != 0) {
    cu = {0, T};
  }
  const bool varlen = !cu.empty();
  const int64_t N = varlen ? int64_t(cu.size()) - 1 : 1;
  TORCH_CHECK(!(varlen && N > 1) || cu_seqlens, "internal: synthetic cu is always single-sequence");





  const Route route = pick_route(cu, T, H, varlen);
  const int npb = route.npb;
  std::vector<VlPiece> pieces;
  int64_t nc = 0, npieces = 0;
  if (varlen) {
    int nc_tot = 0;
    for (size_t s = 0; s + 1 < cu.size(); ++s)
      nc_tot += int((cu[s + 1] - cu[s] + BT - 1) / BT);
    std::vector<VlPiece> seqs;
    int nc_run = 0;
    for (size_t s = 0; s + 1 < cu.size(); ++s) {
      const int len = int(cu[s + 1] - cu[s]);
      const int ncs = (len + BT - 1) / BT;
      const int np =
          std::min(std::min(npb, std::max(1, ncs / 2)), std::max(1, (2 * npb * ncs + nc_tot) / (2 * nc_tot)));
      const int tokoff = int(cu[s]) - nc_run * BT;
      for (int sg = 0; sg < np; ++sg)
        pieces.push_back(VlPiece{int(s), nc_run, ncs, np, sg, tokoff, int(cu[s + 1]), sg == np - 1 && len % BT != 0});
      seqs.push_back(VlPiece{int(s), nc_run, ncs, 1, 0, tokoff, int(cu[s + 1]), len % BT != 0});
      nc_run += ncs;
    }
    nc = nc_run;
    npieces = int64_t(pieces.size());
    pieces.insert(pieces.end(), seqs.begin(), seqs.end());
  } else {
    nc = T / BT;
    npieces = std::min<int64_t>(npb, std::max<int64_t>(1, nc / 2));
  }

  const c10::cuda::CUDAGuard guard(q.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  Workspace& ws = get_workspace(q.device(), T, nc, H, npieces, N);


  const float* h0_p;
  if (initial_state) {
    TORCH_CHECK(
        initial_state->is_cuda() && initial_state->is_contiguous() && initial_state->scalar_type() == torch::kFloat &&
            initial_state->sizes() == torch::IntArrayRef({N, H, int64_t(K), int64_t(K)}),
        "initial_state must be contiguous CUDA fp32 [N=",
        N,
        ", H=",
        H,
        ", 128, 128], got ",
        initial_state->sizes(),
        " ",
        initial_state->dtype());
    h0_p = initial_state->data_ptr<float>();
  } else {
    if (!ws.h0z.defined())
      ws.h0z =
          torch::zeros({N, H, int64_t(K), int64_t(K)}, torch::TensorOptions().dtype(torch::kFloat).device(q.device()));
    h0_p = ws.h0z.data_ptr<float>();
  }


  auto o = torch::empty({T, H, int64_t(K)}, torch::TensorOptions().dtype(torch::kBFloat16).device(q.device()));
  auto Sf =
      torch::empty({N, H, int64_t(K), int64_t(K)}, torch::TensorOptions().dtype(torch::kFloat).device(q.device()));





  void* hpc_p = nullptr;
  bool hpc_bf16 = false;
  if (h_per_chunk) {
    const auto& hp = *h_per_chunk;
    TORCH_CHECK(
        hp.is_cuda() && hp.device() == q.device() && hp.is_contiguous() &&
            hp.sizes() == torch::IntArrayRef({nc, H, int64_t(K), int64_t(K)}),
        "h_per_chunk must be a contiguous [nc=",
        nc,
        ", H=",
        H,
        ", 128, 128] tensor on ",
        q.device(),
        ", got ",
        hp.sizes(),
        " on ",
        hp.device());
    TORCH_CHECK(
        hp.scalar_type() == torch::kFloat || hp.scalar_type() == torch::kBFloat16,
        "h_per_chunk must be fp32 or bf16, got ",
        hp.dtype());
    hpc_bf16 = hp.scalar_type() == torch::kBFloat16;
    hpc_p = hp.data_ptr();
  }

  const torch::Tensor beta_f =
      beta.scalar_type() == torch::kFloat ? beta : beta.to(torch::kFloat);

  const bf16* q_p = reinterpret_cast<const bf16*>(q.const_data_ptr());
  const bf16* k_p = reinterpret_cast<const bf16*>(k.const_data_ptr());
  const bf16* v_p = reinterpret_cast<const bf16*>(v.const_data_ptr());
  const bf16* g_p = reinterpret_cast<const bf16*>(g.const_data_ptr());
  const float* beta_p = beta_f.data_ptr<float>();
  bf16* o_p = reinterpret_cast<bf16*>(o.data_ptr());
  float* Sf_p = Sf.data_ptr<float>();
  bf16* P_p = reinterpret_cast<bf16*>(ws.P.data_ptr());
  bf16* kdec_p = reinterpret_cast<bf16*>(ws.kdec.data_ptr());
  bf16* qdec_p = reinterpret_cast<bf16*>(ws.qdec.data_ptr());
  bf16* aqh_p = reinterpret_cast<bf16*>(ws.aqk_h.data_ptr());
  bf16* aql_p = reinterpret_cast<bf16*>(ws.aqk_l.data_ptr());
  float* u0_p = ws.u0.data_ptr<float>();
  float* gC_p = ws.gC.data_ptr<float>();
  bf16* pL_p = reinterpret_cast<bf16*>(ws.pieceL.data_ptr());
  float* pc_p = ws.piecec.data_ptr<float>();
  uint32_t* pf_p = reinterpret_cast<uint32_t*>(ws.pflags.data_ptr());
  const float scl = float(scale);



  auto gm_dispatch = [&](auto&& f) {
    auto bsig = [&](auto gmv, auto rawv) {
      if (use_beta_sigmoid_in_kernel)
        f(gmv, rawv, std::true_type{});
      else
        f(gmv, rawv, std::false_type{});
    };
    auto raw = [&](auto gmv) {
      if (use_qk_l2norm_in_kernel)
        bsig(gmv, std::true_type{});
      else
        bsig(gmv, std::false_type{});
    };
    if (gm == 2)
      raw(std::integral_constant<int, 2>{});
    else if (gm == 1)
      raw(std::integral_constant<int, 1>{});
    else
      raw(std::integral_constant<int, 0>{});
  };

  const VlPiece* pt_p = nullptr;
  if (varlen) {


    if (ws.cu != cu) {
      static_assert(std::is_standard_layout_v<VlPiece>);
      const auto hp =
          torch::from_blob(pieces.data(), {(npieces + N) * 8}, torch::TensorOptions().dtype(torch::kInt)).clone();
      if (!ws.pieces_dev.defined())
        ws.pieces_dev = torch::empty({(npieces + N) * 8}, torch::TensorOptions().dtype(torch::kInt).device(q.device()));
      ws.pieces_dev.copy_(hp);
      ws.cu = cu;
    }
    pt_p = reinterpret_cast<const VlPiece*>(ws.pieces_dev.const_data_ptr());
  }
  if (route.seq0 && g_p != ws.gin_ptr) {
    ws.gin_map = enc2dgb(const_cast<bf16*>(g_p), (uint64_t)T, (uint64_t)(H * K));
    ws.gin_ptr = g_p;
  }



  if (route.seq0 && varlen) {
    gm_dispatch([&](auto gmv, auto rawv, auto bsv) {
      k1_tf_builder_vl<gmv(), rawv(), bsv()><<<int(npieces * H), 512, 0, stream>>>(
          q_p,
          k_p,
          v_p,
          g_p,
          ws.gin_map,
          alog_p,
          dtb_p,
          lb,
          beta_p,
          int(H),
          scl,
          P_p,
          u0_p,
          kdec_p,
          qdec_p,
          aqh_p,
          aql_p,
          gC_p,
          pt_p,
          int(nc));
    });
    k2_chain_tc_vl<<<int(N * H), 512, 0, stream>>>(
        ws.pneg_map,
        ws.kdt_map,
        ws.qd_map,
        ws.aqh_map,
        ws.aql_map,
        ws.u0f_map,
        gC_p,
        h0_p,
        int(H),
        o_p,
        Sf_p,
        hpc_p,
        hpc_bf16,
        h_v_first,
        pt_p + npieces);
  } else if (route.seq0) {
    gm_dispatch([&](auto gmv, auto rawv, auto bsv) {
      k1_tf_builder<gmv(), rawv(), bsv()><<<int(npieces * H), 512, 0, stream>>>(
          q_p,
          k_p,
          v_p,
          g_p,
          ws.gin_map,
          alog_p,
          dtb_p,
          lb,
          beta_p,
          int(T),
          int(H),
          scl,
          P_p,
          u0_p,
          kdec_p,
          qdec_p,
          aqh_p,
          aql_p,
          gC_p,
          int(npieces));
    });
    k2_chain_tc<<<int(H), 512, 0, stream>>>(
        ws.pneg_map,
        ws.kdt_map,
        ws.qd_map,
        ws.aqh_map,
        ws.aql_map,
        ws.sl_map,
        ws.sc_map,
        ws.u0f_map,
        gC_p,
        h0_p,
        int(nc),
        int(H),
        o_p,
        Sf_p,
        hpc_p,
        hpc_bf16,
        h_v_first);
  } else if (varlen) {
    AT_CUDA_CHECK(cudaMemsetAsync(pf_p, 0, size_t(npieces * H) * 4, stream));
    gm_dispatch([&](auto gmv, auto rawv, auto bsv) {
      kda_fused_vl<gmv(), rawv(), bsv()><<<2 * int(npieces * H), 512, 0, stream>>>(
          q_p,
          k_p,
          v_p,
          g_p,
          alog_p,
          dtb_p,
          lb,
          beta_p,
          int(T),
          int(H),
          scl,
          P_p,
          u0_p,
          kdec_p,
          qdec_p,
          aqh_p,
          aql_p,
          gC_p,
          pL_p,
          pc_p,
          pt_p,
          int(npieces),
          int(nc),
          ws.pneg_map,
          ws.kdt_map,
          ws.qd_map,
          ws.aqh_map,
          ws.aql_map,
          ws.u0f_map,
          ws.sl_map,
          ws.sc_map,
          h0_p,
          o_p,
          Sf_p,
          hpc_p,
          hpc_bf16,
          h_v_first,
          pf_p);
    });
  } else if (npieces > 1) {
    AT_CUDA_CHECK(cudaMemsetAsync(pf_p, 0, size_t(npieces * H) * 4, stream));
    gm_dispatch([&](auto gmv, auto rawv, auto bsv) {
      kda_fused<gmv(), rawv(), bsv()><<<2 * int(npieces * H), 512, 0, stream>>>(
          q_p,
          k_p,
          v_p,
          g_p,
          alog_p,
          dtb_p,
          lb,
          beta_p,
          int(T),
          int(H),
          scl,
          P_p,
          u0_p,
          kdec_p,
          qdec_p,
          aqh_p,
          aql_p,
          gC_p,
          pL_p,
          pc_p,
          int(npieces),
          ws.pneg_map,
          ws.kdt_map,
          ws.qd_map,
          ws.aqh_map,
          ws.aql_map,
          ws.u0f_map,
          ws.sl_map,
          ws.sc_map,
          h0_p,
          o_p,
          Sf_p,
          hpc_p,
          hpc_bf16,
          h_v_first,
          pf_p);
    });
  } else {
    gm_dispatch([&](auto gmv, auto rawv, auto bsv) {
      k1_factors_mma<gmv(), rawv(), bsv()><<<int(H), 512, 0, stream>>>(
          q_p,
          k_p,
          v_p,
          g_p,
          alog_p,
          dtb_p,
          lb,
          beta_p,
          int(T),
          int(H),
          scl,
          P_p,
          u0_p,
          kdec_p,
          qdec_p,
          aqh_p,
          aql_p,
          gC_p,
          nullptr,
          nullptr,
          1);
    });
    k2_chain_tc<<<int(H), 512, 0, stream>>>(
        ws.pneg_map,
        ws.kdt_map,
        ws.qd_map,
        ws.aqh_map,
        ws.aql_map,
        ws.sl_map,
        ws.sc_map,
        ws.u0f_map,
        gC_p,
        h0_p,
        int(nc),
        int(H),
        o_p,
        Sf_p,
        hpc_p,
        hpc_bf16,
        h_v_first);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {o, Sf};
}

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "kda_prefill_fwd",
      &kda::kda_prefill_fwd,
      "KDA chunked prefill forward (inference): returns (o, final_state)",
      py::arg("q"),
      py::arg("k"),
      py::arg("v"),
      py::arg("g"),
      py::arg("beta"),
      py::arg("scale"),
      py::arg("initial_state") = std::nullopt,
      py::arg("cu_seqlens") = std::nullopt,
      py::arg("use_gate_in_kernel") = false,
      py::arg("A_log") = std::nullopt,
      py::arg("dt_bias") = std::nullopt,
      py::arg("safe_gate") = false,
      py::arg("lower_bound") = -5.0,
      py::arg("use_qk_l2norm_in_kernel") = false,
      py::arg("use_beta_sigmoid_in_kernel") = false,
      py::arg("h_per_chunk") = std::nullopt,
      py::arg("h_v_first") = false);
}
