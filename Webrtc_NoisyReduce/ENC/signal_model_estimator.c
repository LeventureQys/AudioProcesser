#include "signal_model_estimator.h"
#include "fast_math.h"

float ComputeSpectralDiff(float* conservative_noise_spectrum, float* signal_spectrum, float signal_spectral_sum, float diff_normalization);
void UpdateSpectralFlatness(float* signal_spectrum, float signal_spectral_sum, float* spectral_flatness);
void UpdateSpectralLrt(float* prior_snr, float* post_snr, float* avg_log_lrt, float* lrt);


// Computes the difference measure between input spectrum and a template/learned noise spectrum.
float ComputeSpectralDiff(float* conservative_noise_spectrum, float* signal_spectrum, float signal_spectral_sum, float diff_normalization) {
    // Compute average quantities.
    float noise_average = 0.f;
    for (int i = 0; i < kFftSizeBy2Plus1; ++i) {
        // Conservative smooth noise spectrum from pause frames.
        noise_average += conservative_noise_spectrum[i];
    }
    noise_average = noise_average * kOneByFftSizeBy2Plus1;
    float signal_average = signal_spectral_sum * kOneByFftSizeBy2Plus1;

    // Compute variance and covariance quantities.
    float covariance = 0.f;
    float noise_variance = 0.f;
    float signal_variance = 0.f;
    for (int i = 0; i < kFftSizeBy2Plus1; ++i) {
        float signal_diff = signal_spectrum[i] - signal_average;
        float noise_diff = conservative_noise_spectrum[i] - noise_average;
        covariance += signal_diff * noise_diff;
        noise_variance += noise_diff * noise_diff;
        signal_variance += signal_diff * signal_diff;
    }
    covariance *= kOneByFftSizeBy2Plus1;
    noise_variance *= kOneByFftSizeBy2Plus1;
    signal_variance *= kOneByFftSizeBy2Plus1;

    // Update of average magnitude spectrum.
    float spectral_diff = signal_variance - (covariance * covariance) / (noise_variance + 0.0001f);
    // Normalize.
    return spectral_diff / (diff_normalization + 0.0001f);

}

// Updates the spectral flatness based on the input spectrum.
void UpdateSpectralFlatness(float* signal_spectrum, float signal_spectral_sum, float* spectral_flatness) {
    // Compute log of ratio of the geometric to arithmetic mean (handle the log(0)
    // separately).
    float kAveraging = 0.3f;
    float avg_spect_flatness_num = 0.f;
    for (int i = 1; i < kFftSizeBy2Plus1; ++i) {
        if (signal_spectrum[i] == 0.f) {
            *spectral_flatness -= kAveraging * (*spectral_flatness);
            return;
        }
    }

    for (size_t i = 1; i < kFftSizeBy2Plus1; ++i) {
        avg_spect_flatness_num += LogApproximation(signal_spectrum[i]);
    }

    float avg_spect_flatness_denom = signal_spectral_sum - signal_spectrum[0];

    avg_spect_flatness_denom = avg_spect_flatness_denom * kOneByFftSizeBy2Plus1;
    avg_spect_flatness_num = avg_spect_flatness_num * kOneByFftSizeBy2Plus1;

    float spectral_tmp = ExpApproximation(avg_spect_flatness_num) / avg_spect_flatness_denom;
    // Time-avg update of spectral flatness feature.
    *spectral_flatness += kAveraging * (spectral_tmp - *spectral_flatness);
}

// Updates the log LRT measures.
void UpdateSpectralLrt(float* prior_snr, float* post_snr, float* avg_log_lrt, float* lrt) {
    for (size_t i = 0; i < kFftSizeBy2Plus1; ++i) {
        float tmp1 = 1.f + 2.f * prior_snr[i];
        float tmp2 = 2.f * prior_snr[i] / (tmp1 + 0.0001f);
        float bessel_tmp = (post_snr[i] + 1.f) * tmp2;
        avg_log_lrt[i] += .5f * (bessel_tmp - LogApproximation(tmp1) - avg_log_lrt[i]);
    }

    float log_lrt_time_avg_k_sum = 0.f;
    for (size_t i = 0; i < kFftSizeBy2Plus1; ++i) {
        log_lrt_time_avg_k_sum += avg_log_lrt[i];
    }
    *lrt = log_lrt_time_avg_k_sum * kOneByFftSizeBy2Plus1;
}

void SignalModelEstimator_init(SignalModelEstimator* var, float lrt_initial_value)
{
    var->diff_normalization_ = 0.f;
    var->signal_energy_sum_ = 0.f;
    var->histogram_analysis_counter_ = 500;
    PriorSignalModelEstimator_init(var->histograms_, var->prior_model_, lrt_initial_value);
    SignalModel_init(var->features_);
}

void SignalModelEstimator_AdjustNormalization(SignalModelEstimator* var, int32_t num_analyzed_frames, float signal_energy) {

    var->diff_normalization_ *= num_analyzed_frames;
    var->diff_normalization_ += signal_energy;
    var->diff_normalization_ /= (num_analyzed_frames + 1);
}

void SignalModelEstimator_Update(SignalModelEstimator* var, float* prior_snr, float* post_snr, 
    float* conservative_noise_spectrum, float* signal_spectrum, float signal_spectral_sum, float signal_energy)
{
    // Compute spectral flatness on input spectrum.
    UpdateSpectralFlatness(signal_spectrum, signal_spectral_sum, &var->features_.spectral_flatness);
    // Compute difference of input spectrum with learned/estimated noise spectrum.
    float spectral_diff = ComputeSpectralDiff(conservative_noise_spectrum, signal_spectrum, signal_spectral_sum, var->diff_normalization_);
    // Compute time-avg update of difference feature.
    var->features_.spectral_diff += 0.3f * (spectral_diff - var->features_.spectral_diff);
    var->signal_energy_sum_ += signal_energy;

    // Compute histograms for parameter decisions (thresholds and weights for
    // features). Parameters are extracted periodically.
    if (--var->histogram_analysis_counter_ > 0) {
        Histograms_Update(var->features_, var->histograms_);
    } else {
        // Compute model parameters.
        PriorSignalModelEstimator_Update(var->features_, var->histograms_, var->prior_model_);

        // Clear histograms for next update.
        Histograms_Clear(var->histograms_);

        var->histogram_analysis_counter_ = kFeatureUpdateWindowSize;

        // Update every window:
        // Compute normalization for the spectral difference for next estimation.
        var->signal_energy_sum_ = var->signal_energy_sum_ / kFeatureUpdateWindowSize;
        var->diff_normalization_ = 0.5f * (var->signal_energy_sum_ + var->diff_normalization_);
        var->signal_energy_sum_ = 0.f;
    }

}
