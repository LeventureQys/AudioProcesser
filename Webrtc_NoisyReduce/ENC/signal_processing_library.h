#ifndef COMMON_AUDIO_SIGNAL_PROCESSING_INCLUDE_SIGNAL_PROCESSING_LIBRARY_H_
#define COMMON_AUDIO_SIGNAL_PROCESSING_INCLUDE_SIGNAL_PROCESSING_LIBRARY_H_

#include <string.h>
#include <math.h>
#include "ns_common.h"

// C + the 32 most significant bits of A * B
#define WEBRTC_SPL_SCALEDIFF32(A, B, C) \ (C + (B >> 16) * A + (((uint32_t)(B & 0x0000FFFF) * A) >> 16))

static int16_t FloatS16ToS16(float v) {
    v = min_local(v, 32767.f);
    v = max_local(v, -32768.f);
    return (int16_t)(v + copysign(0.5f, v));
}


int16_t WebRtcSpl_SatW32ToW16(int32_t value32);
int32_t WebRtcSpl_AddSatW32(int32_t a, int32_t b);
int32_t WebRtcSpl_SubSatW32(int32_t a, int32_t b);
int16_t WebRtcSpl_AddSatW16(int16_t a, int16_t b);
int16_t WebRtcSpl_SubSatW16(int16_t var1, int16_t var2);

void FloatS16ToS16(const float* src, size_t size, int16_t* dest);
void S16ToFloatS16(const int16_t* src, size_t size, float* dest);

void WebRtcSpl_AllPassQMF(int32_t* in_data, int data_length, int32_t* out_data, uint16_t* filter_coefficients, int32_t* filter_state);
void WebRtcSpl_AnalysisQMF(int16_t* in_data, int in_data_length, int16_t* low_band, int16_t* high_band,
                            int32_t* filter_state1, int32_t* filter_state2);
void WebRtcSpl_SynthesisQMF(int16_t* low_band, int16_t* high_band, int band_length,
                            int16_t* out_data, int32_t* filter_state1, int32_t* filter_state2);

#endif  // COMMON_AUDIO_SIGNAL_PROCESSING_INCLUDE_SIGNAL_PROCESSING_LIBRARY_H_
