#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include "resampler/sinc_resampler.h"


float SincScaleFactor(float io_ratio) {
    // `sinc_scale_factor` is basically the normalized cutoff frequency of the low-pass filter.
    float sinc_scale_factor = io_ratio > 1.0 ? 1.0 / io_ratio : 1.0;
    // The sinc function is an idealized brick-wall filter, but since we're
    // windowing it the transition from pass to stop does not happen right away.
    // So we should adjust the low pass filter cutoff slightly downward to avoid
    // some aliasing at the very high-end.
    // TODO(crogers): this value is empirical and to be more exact should vary
    // depending on kKernelSize.
    sinc_scale_factor *= 0.9;

    return sinc_scale_factor;
}

void UpdateRegions(bool second_load) {
  // Setup various region pointers in the buffer (see diagram above).  If we're
  // on the second load we need to slide r0_ to the right by kKernelSize / 2.
}

void InitializeKernel(SincResampler* var) {
    // Blackman window parameters.
    float kAlpha = 0.16;
    float kA0 = 0.5 * (1.0 - kAlpha);
    float kA1 = 0.5;
    float kA2 = 0.5 * kAlpha;

    // Generates a set of windowed sinc() kernels.
    // We generate a range of sub-sample offsets from 0.0 to 1.0.
    float sinc_scale_factor = SincScaleFactor(var->io_sample_rate_ratio_);
    for (int offset_idx = 0; offset_idx <= kKernelOffsetCount; ++offset_idx) {
        float subsample_offset = (float)(offset_idx) / kKernelOffsetCount;

        for (int i = 0; i < kKernelSize; ++i) {
            int idx = i + offset_idx * kKernelSize;
            float pre_sinc = M_PI * ((float)(i - kKernelSize / 2) - subsample_offset);
            var->kernel_pre_sinc_storage_[idx] = pre_sinc;

            // Compute Blackman window, matching the offset of the sinc().
            float x = (i - subsample_offset) / kKernelSize;
            float window = kA0 - kA1 * cos(2.0 * M_PI * x) + kA2 * cos(4.0 * M_PI * x);

            var->kernel_window_storage_[idx] = window;

            // Compute the sinc with offset, then window the sinc() function and store at the correct offset.
            var->kernel_storage_[idx] = window * ((pre_sinc == 0) ? sinc_scale_factor
                                   : (sin(sinc_scale_factor * pre_sinc) / pre_sinc));
        }
    }

}

void SetRatio(SincResampler* var, float io_sample_rate_ratio) {
    if (fabs(var->io_sample_rate_ratio_ - io_sample_rate_ratio) < epsilon) {
        return;
    }
    var->io_sample_rate_ratio_ = io_sample_rate_ratio;
    // Optimize reinitialization by reusing values which are independent of `sinc_scale_factor`.  Provides a 3x speedup.
    float sinc_scale_factor = SincScaleFactor(var->io_sample_rate_ratio_);
    for (int offset_idx = 0; offset_idx <= kKernelOffsetCount; ++offset_idx) {
        for (int i = 0; i < kKernelSize; ++i) {
            float idx = i + offset_idx * kKernelSize;
            float window = var->kernel_window_storage_[idx];
            float pre_sinc = var->kernel_pre_sinc_storage_[idx];

            var->kernel_storage_[idx] = window * ((pre_sinc == 0) ? sinc_scale_factor
                                   : (sin(sinc_scale_factor * pre_sinc) / pre_sinc));
        }
    }

}

void Resample(SincResampler* var, int frames, float* destination) {
    int remaining_frames = frames;

    // Step (1) -- Prime the input buffer at the start of the input stream.
    if (!buffer_primed_ && remaining_frames) {
    }

}
