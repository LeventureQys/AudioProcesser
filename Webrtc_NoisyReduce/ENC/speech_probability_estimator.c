#include <stdio.h>
#include <string.h>
#include "speech_probability_estimator.h"
#include <math.h>
#include "fast_math.h"


void SpeechProbabilityEstimator_init(SpeechProbabilityEstimator* var) {
    var->prior_speech_prob_ = 0.5f;
    memset(var->speech_probability_, 0, sizeof(float) * (kFftSizeBy2Plus1));

    SignalModelEstimator_init(var->signal_model_estimator_, kLtrFeatureThr);
}

void SpeechProbabilityEstimator_Update(SpeechProbabilityEstimator* var, int32_t num_analyzed_frames, float* prior_snr, float* post_snr, 
    float* conservative_noise_spectrum, float* signal_spectrum, float signal_spectral_sum, float signal_energy)
{
    // Update models.
    if (num_analyzed_frames < kLongStartupPhaseBlocks) {
        SignalModelEstimator_AdjustNormalization(var->signal_model_estimator_, num_analyzed_frames, signal_energy);
    }

    SignalModelEstimator_Update(var->signal_model_estimator_, prior_snr, post_snr, 
                    conservative_noise_spectrum, signal_spectrum, signal_spectral_sum, signal_energy);

    const SignalModel& model = var->signal_model_estimator_.features_;
    const PriorSignalModel& prior_model = var->signal_model_estimator_.prior_model_;

    // Width parameter in sigmoid map for prior model.
    float kWidthPrior0 = 4.f;
    // Width for pause region: lower range, so increase width in tanh map.
    float kWidthPrior1 = 2.f * kWidthPrior0;

    // Average LRT feature: use larger width in tanh map for pause regions.
    float width_prior = model.lrt < prior_model.lrt ? kWidthPrior1 : kWidthPrior0;

    // Compute indicator function: sigmoid map.
    float indicator0 = 0.5f * (tanh(width_prior * (model.lrt - prior_model.lrt)) + 1.f);

    // Spectral flatness feature: use larger width in tanh map for pause regions.
    width_prior = model.spectral_flatness > prior_model.flatness_threshold ? kWidthPrior1 : kWidthPrior0;

    // Compute indicator function: sigmoid map.
    float indicator1 = 0.5f * (tanh(1.f * width_prior *
                   (prior_model.flatness_threshold - model.spectral_flatness)) + 1.f);

    // For template spectrum-difference : use larger width in tanh map for pause regions.
    width_prior = model.spectral_diff < prior_model.template_diff_threshold ? kWidthPrior1 : kWidthPrior0;

    // Compute indicator function: sigmoid map.
    float indicator2 = 0.5f * (tanh(width_prior * 
                   (model.spectral_diff - prior_model.template_diff_threshold)) + 1.f);

    // Combine the indicator function with the feature weights.
    float ind_prior = prior_model.lrt_weighting * indicator0 +
                    prior_model.flatness_weighting * indicator1 +
                    prior_model.difference_weighting * indicator2;

    // Compute the prior probability.
    var->prior_speech_prob_ += 0.1f * (ind_prior - var->prior_speech_prob_);

    // Make sure probabilities are within range: keep floor to 0.01.
    var->prior_speech_prob_ = max_local(min_local(var->prior_speech_prob_, 1.f), 0.01f);

    // Final speech probability: combine prior model with LR factor:.
    float gain_prior = (1.f - var->prior_speech_prob_) / (var->prior_speech_prob_ + 0.0001f);

    float inv_lrt[kFftSizeBy2Plus1];
    ExpApproximationSignFlip(model.avg_log_lrt, inv_lrt);
    for (int i = 0; i < kFftSizeBy2Plus1; ++i) {
        var->speech_probability_[i] = 1.f / (1.f + gain_prior * inv_lrt[i]);
    }
}

