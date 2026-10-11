#pragma once

#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tuple.h>

#include <cstdint>

namespace sglang {

namespace device::compress {

struct alignas(16) PrefillPlan {
  uint32_t ragged_id;
  uint32_t batch_id;
  uint32_t position;
  uint32_t window_len;

  bool is_valid(const uint32_t ratio, const bool is_overlap) const {
    const uint32_t max_window_len = ratio * (1 + is_overlap);
    return window_len < max_window_len;
  }
};

}

namespace host::compress {

using device::compress::PrefillPlan;
using PrefillPlanTensorDtype = uint8_t;
inline constexpr int64_t kPrefillPlanDim = 16;

static_assert(alignof(PrefillPlan) == sizeof(PrefillPlan));
static_assert(sizeof(PrefillPlan) == kPrefillPlanDim * sizeof(PrefillPlanTensorDtype));

}

}
