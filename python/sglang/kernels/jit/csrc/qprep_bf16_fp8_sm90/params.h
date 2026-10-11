/* Copyright 2026 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/





#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace sglang {

struct QprepBf16Fp8Sm90Params {
  int num_tokens;
  int num_heads;


  const void* q_nope;
  int64_t a_s0, a_s1;



  const void* w_kc;
  int64_t b_s0, b_s2;


  const void* q_rope;
  int64_t r_s0, r_s1;

  bool rope_vec16;


  void* out;
  int64_t o_s0, o_s1;


  bool out_vec16;

  cudaStream_t stream;
};

}
