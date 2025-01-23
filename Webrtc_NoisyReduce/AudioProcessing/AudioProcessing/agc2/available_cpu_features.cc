/*
 *  Copyright (c) 2020 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <sstream>
#include "available_cpu_features.h"

#include "base/arch.h"
#include "base/cpu_features_wrapper.h"

namespace webrtc {

std::string AvailableCpuFeatures::ToString() const {
  std::ostringstream builder;
  bool first = true;
  if (sse2) {
    builder << (first ? "SSE2" : "_SSE2");
    first = false;
  }
  if (avx2) {
    builder << (first ? "AVX2" : "_AVX2");
    first = false;
  }
  if (neon) {
    builder << (first ? "NEON" : "_NEON");
    first = false;
  }
  if (first) {
    return "none";
  }
  return builder.str();
}

// Detects available CPU features.
AvailableCpuFeatures GetAvailableCpuFeatures() {
#if defined(WEBRTC_ARCH_X86_FAMILY)
  return {/*sse2=*/GetCPUInfo(kSSE2) != 0,
          /*avx2=*/GetCPUInfo(kAVX2) != 0,
          /*neon=*/false};
#elif defined(WEBRTC_HAS_NEON)
  return {/*sse2=*/false,
          /*avx2=*/false,
          /*neon=*/true};
#else
  return {/*sse2=*/false,
          /*avx2=*/false,
          /*neon=*/false};
#endif
}

AvailableCpuFeatures NoAvailableCpuFeatures() {
  return {/*sse2=*/false, /*avx2=*/false, /*neon=*/false};
}

}  // namespace webrtc
