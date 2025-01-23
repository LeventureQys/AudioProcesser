#ifndef MODULES_AUDIO_PROCESSING_THREE_BAND_FILTER_BANK_H_
#define MODULES_AUDIO_PROCESSING_THREE_BAND_FILTER_BANK_H_

#include <stdio.h>
#include <string.h>
#include "ns_common.h"

#define kSparsity 4
#define kStrideLog2 2
#define kStride (1 << kStrideLog2)
#define kNumZeroFilters 2
#define kFilterSize 4
#define kMemorySize (kFilterSize * kStride - 1)
#define kNumBands 3
#define kFullBandSize 480
#define kSplitBandSize (kFullBandSize / kNumBands)
#define kNumNonZeroFilters (kSparsity * kNumBands - kNumZeroFilters)
// An implementation of a 3-band FIR filter-bank with DCT modulation, similar to
// the proposed in "Multirate Signal Processing for Communication Systems" by
// Fredric J Harris.
// The low-pass filter prototype has these characteristics:
// * Pass-band ripple = 0.3dB
// * Pass-band frequency = 0.147 (7kHz at 48kHz)
// * Stop-band attenuation = 40dB
// * Stop-band frequency = 0.192 (9.2kHz at 48kHz)
// * Delay = 24 samples (500us at 48kHz)
// * Linear phase
// This filter bank does not satisfy perfect reconstruction. The SNR after
// analysis and synthesis (with no processing in between) is approximately 9.5dB
// depending on the input signal after compensating for the delay.
typedef struct
{
    float state_analysis_[kNumNonZeroFilters][kMemorySize];
    float state_synthesis_[kNumNonZeroFilters][kMemorySize];
} ThreeBandFilterBank;


void ThreeBandFilterBank_init(ThreeBandFilterBank* var);
void ThreeBandFilterBank_Analysis(ThreeBandFilterBank* var, float* in, float* out);
void ThreeBandFilterBank_Synthesis(ThreeBandFilterBank* var, float* in, float* out);

#endif  // MODULES_AUDIO_PROCESSING_THREE_BAND_FILTER_BANK_H_
