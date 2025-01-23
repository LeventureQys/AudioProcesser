#include "quantile_noise_estimator.h"
#include <math.h>
#include "fast_math.h"
//#include "user_task.h"

void QuantileNoiseEstimator_init(QuantileNoiseEstimator* var)
{
    memset(var->quantile_, 0, sizeof(float) * (kSimult * kFftSizeBy2Plus1));
    var->num_updates_ = 1;
    for (int i = 0; i < kSimult * kFftSizeBy2Plus1; i++)
    {
        var->log_quantile_[i] = 8.f;
    }

    for (int i = 0; i < kFftSizeBy2Plus1; i++)
    {
        var->quantile_[i] = 0.3f;
    }

    float kOneBySimult = 1.f / kSimult;
    for (int i = 0; i < kSimult; ++i) {
        var->counter_[i] = floor(kLongStartupPhaseBlocks * (i + 1.f) * kOneBySimult);
    }
}

void QuantileNoiseEstimator_Estimate(QuantileNoiseEstimator* var, float* signal_spectrum, float* noise_spectrum)
{
    float log_spectrum[kFftSizeBy2Plus1];
    LogApproximation(signal_spectrum, log_spectrum);

    int quantile_index_to_return = -1;
    // Loop over simultaneous estimates.
    for (int s = 0, k = 0; s < kSimult; ++s, k += kFftSizeBy2Plus1)
    {
        float one_by_counter_plus_1 = 1.f / (var->counter_[s] + 1.f);
        for (int i = 0, j = k; i < kFftSizeBy2Plus1; ++i, ++j) {
            // Update log quantile estimate.
            float delta = var->density_[j] > 1.f ? 40.f / var->density_[j] : 40.f;
            float multiplier = delta * one_by_counter_plus_1;
            float QUANTILE = 0.25f;

            if (log_spectrum[i] > var->log_quantile_[j]) {
                var->log_quantile_[j] += QUANTILE * multiplier;
            } else {
                var->log_quantile_[j] -= (1 - QUANTILE) * multiplier;
            }

            // Update density estimate.
            float kWidth = 0.01f;
            float kOneByWidthPlus2 = 1.f / (2.f * kWidth);
            float tmp = abs_local(log_spectrum[i] - var->log_quantile_[j]);
            if (tmp < kWidth) {
                var->density_[j] = (var->counter_[s] * var->density_[j] + kOneByWidthPlus2) * one_by_counter_plus_1;
            }
        }

        if (var->counter_[s] >= kLongStartupPhaseBlocks) {
            var->counter_[s] = 0;
            if (var->num_updates_ >= kLongStartupPhaseBlocks) {
                quantile_index_to_return = k;
            }
        }

        ++var->counter_[s];
    }

    // Sequentially update the noise during startup.
    if (var->num_updates_ < kLongStartupPhaseBlocks) {
        // Use the last "s" to get noise during startup that differ from zero.
        quantile_index_to_return = kFftSizeBy2Plus1 * (kSimult - 1);
        ++var->num_updates_;
    }

    if (quantile_index_to_return >= 0) {
        memcpy(log_spectrum, var->log_quantile_ + quantile_index_to_return, kFftSizeBy2Plus1 * sizeof(float));
        ExpApproximation(log_spectrum, var->quantile_);
    }
    memcpy(noise_spectrum, var->quantile_, kFftSizeBy2Plus1 * sizeof(float));


}
