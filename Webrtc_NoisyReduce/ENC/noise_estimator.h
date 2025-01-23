#ifndef MODULES_AUDIO_PROCESSING_NS_NOISE_ESTIMATOR_H_
#define MODULES_AUDIO_PROCESSING_NS_NOISE_ESTIMATOR_H_


#include "ns_common.h"
#include "quantile_noise_estimator.h"
#include "suppression_params.h"

typedef struct
{
    float white_noise_level_;
    float pink_noise_numerator_;
    float pink_noise_exp_;

    float prev_noise_spectrum_[kFftSizeBy2Plus1];
    float conservative_noise_spectrum_[kFftSizeBy2Plus1];
    float parametric_noise_spectrum_[kFftSizeBy2Plus1];
    float noise_spectrum_[kFftSizeBy2Plus1];

    QuantileNoiseEstimator quantile_noise_estimator_;
} NoiseEstimator;

void NoiseEstimator_init(NoiseEstimator* var);
void NoiseEstimator_PrepareAnalysis(NoiseEstimator* var);
void NoiseEstimator_PreUpdate(NoiseEstimator* var, const SuppressionParams& suppression_params_, 
    int32_t num_analyzed_frames, float* signal_spectrum, float signal_spectral_sum);
void NoiseEstimator_PostUpdate(NoiseEstimator* var, float* speech_probability, float* signal_spectrum);

#endif  // MODULES_AUDIO_PROCESSING_NS_QUANTILE_NOISE_ESTIMATOR_H_
