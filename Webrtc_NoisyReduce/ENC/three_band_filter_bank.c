// An implementation of a 3-band FIR filter-bank with DCT modulation, similar to
// the proposed in "Multirate Signal Processing for Communication Systems" by
// Fredric J Harris.
//
// The idea is to take a heterodyne system and change the order of the
// components to get something which is efficient to implement digitally.
//
// It is possible to separate the filter using the noble identity as follows:
//
// H(z) = H0(z^3) + z^-1 * H1(z^3) + z^-2 * H2(z^3)
//
// This is used in the analysis stage to first downsample serial to parallel
// and then filter each branch with one of these polyphase decompositions of the
// lowpass prototype. Because each filter is only a modulation of the prototype,
// it is enough to multiply each coefficient by the respective cosine value to
// shift it to the desired band. But because the cosine period is 12 samples,
// it requires separating the prototype even further using the noble identity.
// After filtering and modulating for each band, the output of all filters is
// accumulated to get the downsampled bands.
//
// A similar logic can be applied to the synthesis stage.
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "three_band_filter_bank.h"

void FilterCore(float* filter, float* in, int in_shift, float* out, float* state);

#define kSubSampling (kNumBands)
#define kDctSize (kNumBands)
#define kZeroFilterIndex1 3
#define kZeroFilterIndex2 9

const float kFilterCoeffs[kNumNonZeroFilters][kFilterSize] = {
    {-0.00047749f, -0.00496888f, +0.16547118f, +0.00425496f},
    {-0.00173287f, -0.01585778f, +0.14989004f, +0.00994113f},
    {-0.00304815f, -0.02536082f, +0.12154542f, +0.01157993f},
    {-0.00346946f, -0.02587886f, +0.04760441f, +0.00607594f},
    {-0.00154717f, -0.01136076f, +0.01387458f, +0.00186353f},
    {+0.00186353f, +0.01387458f, -0.01136076f, -0.00154717f},
    {+0.00607594f, +0.04760441f, -0.02587886f, -0.00346946f},
    {+0.00983212f, +0.08543175f, -0.02982767f, -0.00383509f},
    {+0.00994113f, +0.14989004f, -0.01585778f, -0.00173287f},
    {+0.00425496f, +0.16547118f, -0.00496888f, -0.00047749f},
};

const float kDctModulation[kNumNonZeroFilters][kDctSize] = {
    {2.f, 2.f, 2.f},
    {1.73205077f, 0.f, -1.73205077f},
    {1.f, -2.f, 1.f},
    {-1.f, 2.f, -1.f},
    {-1.73205077f, 0.f, 1.73205077f},
    {-2.f, -2.f, -2.f},
    {-1.73205077f, 0.f, 1.73205077f},
    {-1.f, 2.f, -1.f},
    {1.f, -2.f, 1.f},
    {1.73205077f, 0.f, -1.73205077f},
};

// Filters the input signal `in` with the filter `filter` using a shift by
// `in_shift`, taking into account the previous state.
void FilterCore(float* filter, float* in, int in_shift, float* out, float* state) {
    int kMaxInShift = (kStride - 1);
    memset(out, 0, sizeof(float) * (kSplitBandSize));
    for (int k = 0; k < in_shift; ++k) {
        for (int i = 0, j = kMemorySize + k - in_shift; i < kFilterSize; ++i, j -= kStride) {
            out[k] += state[j] * filter[i];
        }
    }

    for (int k = in_shift, shift = 0; k < kFilterSize * kStride; ++k, ++shift) {
        int tmp = 1 + (shift >> kStrideLog2);
        int loop_limit = min_local(kFilterSize, tmp);
        for (int i = 0, j = shift; i < loop_limit; ++i, j -= kStride) {
            out[k] += in[j] * filter[i];
        }
        for (int i = loop_limit, j = kMemorySize + shift - loop_limit * kStride; i < kFilterSize; ++i, j -= kStride) {
            out[k] += state[j] * filter[i];
        }
    }

    for (int k = kFilterSize * kStride, shift = kFilterSize * kStride - in_shift; k < kSplitBandSize; ++k, ++shift) {
        for (int i = 0, j = shift; i < kFilterSize; ++i, j -= kStride) {
            out[k] += in[j] * filter[i];
        }
    }

    // Update current state.
    memcpy(state, in + (kSplitBandSize - kMemorySize), kMemorySize * sizeof(float));
}

// Because the low-pass filter prototype has half bandwidth it is possible to
// use a DCT to shift it in both directions at the same time, to the center
// frequencies [1 / 12, 3 / 12, 5 / 12].
void ThreeBandFilterBank_init(ThreeBandFilterBank* var) {
    memset(var->state_analysis_, 0, sizeof(float) * (kNumNonZeroFilters * kMemorySize));
    memset(var->state_synthesis_, 0, sizeof(float) * (kNumNonZeroFilters * kMemorySize));
}

// The analysis can be separated in these steps:
//   1. Serial to parallel downsampling by a factor of `kNumBands`.
//   2. Filtering of `kSparsity` different delayed signals with polyphase
//      decomposition of the low-pass prototype filter and upsampled by a factor
//      of `kSparsity`.
//   3. Modulating with cosines and accumulating to get the desired band.
void ThreeBandFilterBank_Analysis(ThreeBandFilterBank* var, float* in, float* out) {
    // Initialize the output to zero.
    memset(out, 0, sizeof(float) * (kNumBands * kSplitBandSize));
    for (int downsampling_index = 0; downsampling_index < kSubSampling; ++downsampling_index) {
        // Downsample to form the filter input.
        float in_subsampled[kSplitBandSize];
        for (int k = 0; k < kSplitBandSize; ++k) {
            in_subsampled[k] = in[(kSubSampling - 1) - downsampling_index + kSubSampling * k];
        }

        for (int in_shift = 0; in_shift < kStride; ++in_shift) {
            // Choose filter, skip zero filters.
            int index = downsampling_index + in_shift * kSubSampling;
            if (index == kZeroFilterIndex1 || index == kZeroFilterIndex2) {
                continue;
            }
            int filter_index = index < kZeroFilterIndex1 ? index : (index < kZeroFilterIndex2 ? index - 1 : index - 2);

            float filter[kFilterSize], dct_modulation[kDctSize], state[kMemorySize];

            memcpy(filter, kFilterCoeffs[filter_index], kFilterSize * sizeof(float));
            memcpy(dct_modulation, kDctModulation[filter_index], kDctSize * sizeof(float));
            memcpy(state, var->state_analysis_[filter_index], kMemorySize * sizeof(float));
            
            // Filter.
            float out_subsampled[kSplitBandSize];
            FilterCore(filter, in_subsampled, in_shift, out_subsampled, state);

            // Band and modulate the output.
            for (int band = 0; band < kNumBands; ++band) {
                float* out_band = out[band];
                for (int n = 0; n < kSplitBandSize; ++n) {
                    out_band[n] += dct_modulation[band] * out_subsampled[n];
                }
            }
        }
    }

}

// The synthesis can be separated in these steps:
//   1. Modulating with cosines.
//   2. Filtering each one with a polyphase decomposition of the low-pass
//      prototype filter upsampled by a factor of `kSparsity` and accumulating
//      `kSparsity` signals with different delays.
//   3. Parallel to serial upsampling by a factor of `kNumBands`.
void ThreeBandFilterBank_Synthesis(ThreeBandFilterBank* var, float* in, float* out) {
    memset(out, 0, sizeof(float) * (kFullBandSize));
    for (int upsampling_index = 0; upsampling_index < kSubSampling; ++upsampling_index) {
        for (int in_shift = 0; in_shift < kStride; ++in_shift) {
            // Choose filter, skip zero filters.
            int index = upsampling_index + in_shift * kSubSampling;
            if (index == kZeroFilterIndex1 || index == kZeroFilterIndex2) {
                continue;
            }

            int filter_index = index < kZeroFilterIndex1 ? index : (index < kZeroFilterIndex2 ? index - 1 : index - 2);

            float filter[kFilterSize], dct_modulation[kDctSize], state[kMemorySize];

            memcpy(filter, kFilterCoeffs[filter_index], kFilterSize * sizeof(float));
            memcpy(dct_modulation, kDctModulation[filter_index], kDctSize * sizeof(float));
            memcpy(state, var->state_synthesis_[filter_index], kMemorySize * sizeof(float));

            // Prepare filter input by modulating the banded input.
            float in_subsampled[kSplitBandSize];
            memset(in_subsampled, 0, sizeof(float) * (kSplitBandSize));
            for (int band = 0; band < kNumBands; ++band) {
                float* in_band = in[band];
                for (int n = 0; n < kSplitBandSize; ++n) {
                    in_subsampled[n] += dct_modulation[band] * in_band[n];
                }
            }

            // Filter.
            float out_subsampled[kSplitBandSize];
            FilterCore(filter, in_subsampled, in_shift, out_subsampled, state);

            // Upsample.
            float kUpsamplingScaling = kSubSampling;
            for (int k = 0; k < kSplitBandSize; ++k) {
                out[upsampling_index + kSubSampling * k] += kUpsamplingScaling * out_subsampled[k];
            }
        }
    }

}
