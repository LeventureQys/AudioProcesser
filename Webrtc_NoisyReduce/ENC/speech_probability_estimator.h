#ifndef MODULES_AUDIO_PROCESSING_NS_SPEECH_PROBABILITY_ESTIMATOR_H_
#define MODULES_AUDIO_PROCESSING_NS_SPEECH_PROBABILITY_ESTIMATOR_H_

#include "ns_common.h"
#include "signal_model_estimator.h"

typedef struct
{
    float prior_speech_prob_;
    float speech_probability_[kFftSizeBy2Plus1];

    SignalModelEstimator signal_model_estimator_;
} SpeechProbabilityEstimator;

void SpeechProbabilityEstimator_init(SpeechProbabilityEstimator* var);
void SpeechProbabilityEstimator_Update(SpeechProbabilityEstimator* var, int32_t num_analyzed_frames, float* prior_snr, float* post_snr, 
    float* conservative_noise_spectrum, float* signal_spectrum, float signal_spectral_sum, float signal_energy);

#endif  // MODULES_AUDIO_PROCESSING_NS_SPEECH_PROBABILITY_ESTIMATOR_H_
