#ifndef COMMON_AUDIO_RESAMPLER_SINC_RESAMPLER_H_
#define COMMON_AUDIO_RESAMPLER_SINC_RESAMPLER_H_

#include <stdio.h>
#include <string.h>
#include "ns_common.h"

#define kKernelSize 32
#define kDefaultRequestSize 512
#define kKernelOffsetCount 32
#define kKernelStorageSize (kKernelSize * (kKernelOffsetCount + 1))

typedef struct
{ 
    float io_sample_rate_ratio_; // The ratio of input / output sample rates.
    float virtual_source_idx_; // An index on the source input buffer with sub-sample precision.  It must be
                                // double precision to avoid drift.
    bool buffer_primed_; // The buffer is primed once at the very beginning of processing.

    int request_frames_;
    int block_size_;
    int input_buffer_size_;

    float kernel_storage_[];
    float kernel_pre_sinc_storage_[];
    float kernel_window_storage_[];
    float input_buffer_[];


    ConvolveProc convolve_proc_;

    // Pointers to the various regions inside `input_buffer_`.  See the diagram at
    // the top of the .cc file for more information.
    float* r0_;
    float* const r1_;
    float* const r2_;
    float* r3_;
    float* r4_;
} SincResampler;

#endif  // COMMON_AUDIO_RESAMPLER_SINC_RESAMPLER_H_



      
  
