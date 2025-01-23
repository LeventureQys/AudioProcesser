#ifndef MODULES_AUDIO_PROCESSING_NS_WIENER_FILTER_H_
#define MODULES_AUDIO_PROCESSING_NS_WIENER_FILTER_H_

#include "ns_common.h"
#include "suppression_params.h"


typedef struct
{
    float spectrum_prev_process_[kFftSizeBy2Plus1];
    float initial_spectral_estimate_[kFftSizeBy2Plus1];
    float filter_[kFftSizeBy2Plus1];
} WienerFilter;


void WienerFilter_init(WienerFilter* var);
void WienerFilter_Update(WienerFilter* var, const SuppressionParams& suppression_params_, int32_t num_analyzed_frames,
    float* noise_spectrum, float* prev_noise_spectrum, float* parametric_noise_spectrum, float* signal_spectrum);
float ComputeOverallScalingFactor(const SuppressionParams& suppression_params_, int32_t num_analyzed_frames, float prior_speech_probability, 
    float energy_before_filtering, float energy_after_filtering);

#endif  // MODULES_AUDIO_PROCESSING_NS_WIENER_FILTER_H_
