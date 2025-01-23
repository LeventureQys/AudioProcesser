#include "wiener_filter.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "fast_math.h"


void WienerFilter_init(WienerFilter* var) {
    memset(var->spectrum_prev_process_, 0, sizeof(float) * (kFftSizeBy2Plus1));
    memset(var->initial_spectral_estimate_, 0, sizeof(float) * (kFftSizeBy2Plus1));
    for (int i = 0; i < kFftSizeBy2Plus1; i++)
    {
        var->filter_[i] = 1.0;
    }
}

void WienerFilter_Update(WienerFilter* var, const SuppressionParams& suppression_params_, int32_t num_analyzed_frames,
    float* noise_spectrum, float* prev_noise_spectrum, float* parametric_noise_spectrum, float* signal_spectrum) 
{
    for (int i = 0; i < kFftSizeBy2Plus1; ++i) {
        // Previous estimate based on previous frame with gain filter.
        float prev_tsa = var->spectrum_prev_process_[i] / (prev_noise_spectrum[i] + 0.0001f) * filter_[i];
        // Current estimate.
        float current_tsa;
        if (signal_spectrum[i] > noise_spectrum[i]) {
            current_tsa = signal_spectrum[i] / (noise_spectrum[i] + 0.0001f) - 1.f;
        } else {
            current_tsa = 0.f;
        }
        // Directed decision estimate is sum of two terms: current estimate and previous estimate.
        float snr_prior = 0.98f * prev_tsa + (1.f - 0.98f) * current_tsa;
        float tmp = snr_prior / (suppression_params_.over_subtraction_factor + snr_prior);
        var->filter_[i] = max_local(min_local(tmp, 1.f), suppression_params_.minimum_attenuating_gain);
    }

    if (num_analyzed_frames < kShortStartupPhaseBlocks) {
        for (int i = 0; i < kFftSizeBy2Plus1; ++i) {
            var->initial_spectral_estimate_[i] += signal_spectrum[i];
            float filter_initial = var->initial_spectral_estimate_[i] - suppression_params_.over_subtraction_factor * parametric_noise_spectrum[i];
            filter_initial /= var->initial_spectral_estimate_[i] + 0.0001f;
            filter_initial = max_local(min_local(filter_initial, 1.f), suppression_params_.minimum_attenuating_gain);

            // Weight the two suppression filters.
            float kOnyByShortStartupPhaseBlocks = 1.f / kShortStartupPhaseBlocks;
            filter_initial *= kShortStartupPhaseBlocks - num_analyzed_frames;
            var->filter_[i] *= num_analyzed_frames;
            var->filter_[i] += filter_initial;
            var->filter_[i] *= kOnyByShortStartupPhaseBlocks;
        }
    }

    memcpy(var->spectrum_prev_process_, signal_spectrum, sizeof(float) * kFftSizeBy2Plus1);
}

float ComputeOverallScalingFactor(const SuppressionParams& suppression_params_, int32_t num_analyzed_frames, float prior_speech_probability, 
    float energy_before_filtering, float energy_after_filtering)
{
    if (!suppression_params_.use_attenuation_adjustment || num_analyzed_frames <= kLongStartupPhaseBlocks) {
        return 1.f;
    }

    float gain = SqrtFastApproximation(energy_after_filtering / (energy_before_filtering + 1.f));

    // Scaling for new version. Threshold in final energy gain factor calculation.
    float kBLim = 0.5f;
    float scale_factor1 = 1.f;
    if (gain > kBLim) {
        scale_factor1 = 1.f + 1.3f * (gain - kBLim);
        if (gain * scale_factor1 > 1.f) {
            scale_factor1 = 1.f / gain;
        }
    }

    float scale_factor2 = 1.f;
    if (gain < kBLim) {
        // Do not reduce scale too much for pause regions: attenuation here should be controlled by flooring.
        gain = max_local(gain, suppression_params_.minimum_attenuating_gain);
        scale_factor2 = 1.f - 0.3f * (kBLim - gain);

        // Combine both scales with speech/noise prob: note prior (prior_speech_probability) is not frequency dependent.
        return prior_speech_probability * scale_factor1 + (1.f - prior_speech_probability) * scale_factor2;
    }

}
