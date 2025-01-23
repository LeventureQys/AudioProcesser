#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "ns_common.h"
#include "signal_processing_library.h"


int16_t WebRtcSpl_SatW32ToW16(int32_t value32) {
    int16_t out16 = (int16_t)value32;
    if (value32 > 32767)
    out16 = 32767;
    else if (value32 < -32768)
    out16 = -32768;

    return out16;
}

int32_t WebRtcSpl_AddSatW32(int32_t a, int32_t b) {
    // Do the addition in unsigned numbers, since signed overflow is undefined
    // behavior.
    int32_t sum = (int32_t)((uint32_t)a + (uint32_t)b);

    // a + b can't overflow if a and b have different signs. If they have the
    // same sign, a + b also has the same sign iff it didn't overflow.
    if ((a < 0) == (b < 0) && (a < 0) != (sum < 0)) {
    // The direction of the overflow is obvious from the sign of a + b.
    return sum < 0 ? INT32_MAX : INT32_MIN;
    }
    return sum;
}

int32_t WebRtcSpl_SubSatW32(int32_t a, int32_t b) {
    // Do the subtraction in unsigned numbers, since signed overflow is undefined
    // behavior.
    int32_t diff = (int32_t)((uint32_t)a - (uint32_t)b);

    // a - b can't overflow if a and b have the same sign. If they have different
    // signs, a - b has the same sign as a iff it didn't overflow.
    if ((a < 0) != (b < 0) && (a < 0) != (diff < 0)) {
    // The direction of the overflow is obvious from the sign of a - b.
    return diff < 0 ? INT32_MAX : INT32_MIN;
    }
    return diff;
}

int16_t WebRtcSpl_AddSatW16(int16_t a, int16_t b) {
    return WebRtcSpl_SatW32ToW16((int32_t)a + (int32_t)b);
}

int16_t WebRtcSpl_SubSatW16(int16_t var1, int16_t var2) {
    return WebRtcSpl_SatW32ToW16((int32_t)var1 - (int32_t)var2);
}

void FloatS16ToS16(const float* src, int size, int16_t* dest) {
  for (int i = 0; i < size; ++i)
    dest[i] = FloatS16ToS16(src[i]);
}

void S16ToFloatS16(const int16_t* src, int size, float* dest) {
  for (int i = 0; i < size; ++i)
    dest[i] = src[i];
}
