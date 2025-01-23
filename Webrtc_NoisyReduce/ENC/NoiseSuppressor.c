#include "NoiseSuppressor.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "fast_math.h"

#define kMaxNumChannelsOnStack 2

int NumChannelsOnHeap(int num_channels);
void ApplyFilterBankWindow(float* x);
void FormExtendedFrame(float* frame, float* old_data, float* extended_frame);
void OverlapAndAdd(float* extended_frame, float* overlap_memory, float* output_frame);
void DelaySignal(float* frame, float* delay_buffer, float* delayed_frame);
float ComputeEnergyOfExtendedFrame_long(float* in, float len);
float ComputeEnergyOfExtendedFrame_split(float* in1, float* in2, float len1, float len2);
void ComputeMagnitudeSpectrum(float* in, float* signal_spectrum);
void ComputeSnr(float* filter, float* prev_signal_spectrum, float* signal_spectrum,
            float* prev_noise_spectrum, float* noise_spectrum, float* prior_snr, float* post_snr);
float ComputeUpperBandsGain(float minimum_attenuating_gain, float* filter, float* speech_probability,
                    float* prev_analysis_signal_spectrum, float* signal_spectrum);
void NoiseSuppressor_init(NoiseSuppressor* var, int suppression_level);
void NoiseSuppressor_AggregateWienerFilters(NoiseSuppressor* var, float* filter);
void NoiseSuppressor_Analyze(NoiseSuppressor* var, const AudioBuffer& audio);


// Maximum number of channels for which the channel data is stored on
// the stack. If the number of channels are larger than this, they are stored
// using scratch memory that is pre-allocated on the heap. The reason for this
// partitioning is not to waste heap space for handling the more common numbers
// of channels, while at the same time not limiting the support for higher
// numbers of channels by enforcing the channel data to be stored on the
// stack using a fixed maximum value.

// Chooses the number of channels to store on the heap when that is required due
// to the number of channels being larger than the pre-defined number
// of channels to store on the stack.
int NumChannelsOnHeap(int num_channels) {
    return num_channels > kMaxNumChannelsOnStack ? num_channels : 0;
}

// Hybrib Hanning and flat window for the filterbank.
static const float kBlocks160w256FirstHalf[96] = 
{
    0.00000000f, 0.01636173f, 0.03271908f, 0.04906767f, 0.06540313f,
    0.08172107f, 0.09801714f, 0.11428696f, 0.13052619f, 0.14673047f,
    0.16289547f, 0.17901686f, 0.19509032f, 0.21111155f, 0.22707626f,
    0.24298018f, 0.25881905f, 0.27458862f, 0.29028468f, 0.30590302f,
    0.32143947f, 0.33688985f, 0.35225005f, 0.36751594f, 0.38268343f,
    0.39774847f, 0.41270703f, 0.42755509f, 0.44228869f, 0.45690388f,
    0.47139674f, 0.48576339f, 0.50000000f, 0.51410274f, 0.52806785f,
    0.54189158f, 0.55557023f, 0.56910015f, 0.58247770f, 0.59569930f,
    0.60876143f, 0.62166057f, 0.63439328f, 0.64695615f, 0.65934582f,
    0.67155895f, 0.68359230f, 0.69544264f, 0.70710678f, 0.71858162f,
    0.72986407f, 0.74095113f, 0.75183981f, 0.76252720f, 0.77301045f,
    0.78328675f, 0.79335334f, 0.80320753f, 0.81284668f, 0.82226822f,
    0.83146961f, 0.84044840f, 0.84920218f, 0.85772861f, 0.86602540f,
    0.87409034f, 0.88192126f, 0.88951608f, 0.89687274f, 0.90398929f,
    0.91086382f, 0.91749450f, 0.92387953f, 0.93001722f, 0.93590593f,
    0.94154407f, 0.94693013f, 0.95206268f, 0.95694034f, 0.96156180f,
    0.96592583f, 0.97003125f, 0.97387698f, 0.97746197f, 0.98078528f,
    0.98384601f, 0.98664333f, 0.98917651f, 0.99144486f, 0.99344778f,
    0.99518473f, 0.99665524f, 0.99785892f, 0.99879546f, 0.99946459f,
    0.99986614f,
};

// Applies the filterbank window to a buffer.
void ApplyFilterBankWindow(float* x) {
    for (int i = 0; i < 96; ++i) {
        x[i] = kBlocks160w256FirstHalf[i] * x[i];
    }

    for (int i = 161, k = 95; i < kFftSize; ++i, --k) {
        x[i] = kBlocks160w256FirstHalf[k] * x[i];
    }
}

// Extends a frame with previous data.
void FormExtendedFrame(float* frame, float* old_data, float* extended_frame) {
    memcpy(extended_frame, old_data, sizeof(float) * (kFftSize - kNsFrameSize));
    memcpy(extended_frame + (kFftSize - kNsFrameSize), frame, sizeof(float) * kNsFrameSize);
    memcpy(old_data, extended_frame + kNsFrameSize, sizeof(float) * (kFftSize - kNsFrameSize));
}

// Uses overlap-and-add to produce an output frame.
void OverlapAndAdd(float* extended_frame, float* overlap_memory, float* output_frame) {
    for (int i = 0; i < kOverlapSize; ++i) {
        output_frame[i] = overlap_memory[i] + extended_frame[i];
    }
    memcpy(output_frame + kOverlapSize, extended_frame + kOverlapSize, sizeof(float) * (kNsFrameSize - kOverlapSize));
    memcpy(overlap_memory, extended_frame + kNsFrameSize, sizeof(float) * (kFftSize - kNsFrameSize));
}

// Produces a delayed frame.
void DelaySignal(float* frame, float* delay_buffer, float* delayed_frame) {
    int kSamplesFromFrame = kNsFrameSize - (kFftSize - kNsFrameSize);

    memcpy(delayed_frame, delay_buffer, sizeof(float) * (kFftSize - kNsFrameSize));
    memcpy(delayed_frame + (kFftSize - kNsFrameSize), frame, sizeof(float) * kSamplesFromFrame);
    memcpy(delay_buffer, frame + kSamplesFromFrame, sizeof(float) * (kNsFrameSize - kSamplesFromFrame));
}

// Computes the energy of an extended frame.
float ComputeEnergyOfExtendedFrame_long(float* in, float len) {
    float energy = 0.f;
    for (int i = 0; i < len; i++) {
        energy += in[i] * in[i];
    }
    return energy;
}

// Computes the energy of an extended frame based on its subcomponents.
float ComputeEnergyOfExtendedFrame_split(float* in1, float* in2, float len1, float len2) {
    float energy = 0.f;
    for (int i = 0; i < len1; i++) {
        energy += in1[i] * in1[i];
    }

    for (int i = 0; i < len2; i++) {
        energy += in2[i] * in2[i];
    }

    return energy;
}

// Computes the magnitude spectrum based on an FFT output.
void ComputeMagnitudeSpectrum(float* in, float* signal_spectrum) {
    signal_spectrum[0] = abs_local(in[0]) + 1.f;
    signal_spectrum[kFftSizeBy2Plus1 - 1] = abs_local(in[kFftSize]) + 1.f;
    in += 2;

    for (int i = 1; i < kFftSizeBy2Plus1 - 1; ++i) {
        signal_spectrum[i] = SqrtFastApproximation(in[0] * in[0] + in[1] * in[1]) + 1.f;
        in += 2;
    }
}

// Compute prior and post SNR.
void ComputeSnr(float* filter, float* prev_signal_spectrum, float* signal_spectrum,
            float* prev_noise_spectrum, float* noise_spectrum, float* prior_snr, float* post_snr) 
{
    for (int i = 0; i < kFftSizeBy2Plus1; ++i) {
        // Previous post SNR.
        // Previous estimate: based on previous frame with gain filter.
        float prev_estimate = prev_signal_spectrum[i] / (prev_noise_spectrum[i] + 0.0001f) * filter[i];
        // Post SNR.
        if (signal_spectrum[i] > noise_spectrum[i]) {
            post_snr[i] = signal_spectrum[i] / (noise_spectrum[i] + 0.0001f) - 1.f;
        } else {
            post_snr[i] = 0.f;
        }
        // The directed decision estimate of the prior SNR is a sum the current and previous estimates.
        prior_snr[i] = 0.98f * prev_estimate + (1.f - 0.98f) * post_snr[i];
    }
}

// Computes the attenuating gain for the noise suppression of the upper bands.
float ComputeUpperBandsGain(float minimum_attenuating_gain, float* filter, float* speech_probability,
                    float* prev_analysis_signal_spectrum, float* signal_spectrum)
{
    // Average speech prob and filter gain for the end of the lowest band.
    int kNumAvgBins = 32;
    float kOneByNumAvgBins = 1.f / kNumAvgBins;

    float avg_prob_speech = 0.f;
    avg_filter_gain = 0.f;
    for (int i = kFftSizeBy2Plus1 - kNumAvgBins - 1; i < kFftSizeBy2Plus1 - 1; i++) {
        avg_prob_speech += speech_probability[i];
        avg_filter_gain += filter[i];
    }

    avg_prob_speech = avg_prob_speech * kOneByNumAvgBins;
    avg_filter_gain = avg_filter_gain * kOneByNumAvgBins;
    // If the speech was suppressed by a component between Analyze and Process, an
    // example being by an AEC, it should not be considered speech for the purpose
    // of high band suppression. To that end, the speech probability is scaled accordingly.
    float sum_analysis_spectrum = 0.f;
    float sum_processing_spectrum = 0.f;
    for (int i = 0; i < kFftSizeBy2Plus1; ++i) {
        sum_analysis_spectrum += prev_analysis_signal_spectrum[i];
        sum_processing_spectrum += signal_spectrum[i];
    }

    // The magnitude spectrum computation enforces the spectrum to be strictly positive.
    avg_prob_speech *= sum_processing_spectrum / (sum_analysis_spectrum  + 0.0001f);

    // Compute gain based on speech probability.
    float gain = 0.5f * (1.f + static_cast<float>(tanh(2.f * avg_prob_speech - 1.f)));

    // Combine gain with low band gain.
    if (avg_prob_speech >= 0.5f) {  
        gain = 0.25f * gain + 0.75f * avg_filter_gain;
    } else {
        gain = 0.5f * gain + 0.5f * avg_filter_gain;
    }

    // Make sure gain is within flooring range.
    return min_local(max_local(gain, minimum_attenuating_gain), 1.f);
}

void NoiseSuppressor_init(NoiseSuppressor* var, int suppression_level) {

    SuppressionParams_init(var->suppression_params_, suppression_level);
    WienerFilter_init(var->channels_0.wiener_filter);
    NoiseEstimator_init(var->channels_0.noise_estimator);
    SpeechProbabilityEstimator_init(var->channels_0.speech_probability_estimator);
    var->num_analyzed_frames_ = -1;
    var->capture_output_used_ = true;

    memset(var->channels_0.analyze_analysis_memory, 0, sizeof(float) * (kFftSize - kNsFrameSize));
    memset(var->channels_0.process_analysis_memory, 0, sizeof(float) * (kOverlapSize));
    memset(var->channels_0.process_synthesis_memory, 0, sizeof(float) * (kOverlapSize));

    for (int i = 0; i < kFftSizeBy2Plus1; i++)
    {
        var->channels_0.prev_analysis_signal_spectrum[i] = 1.0;
    }
    memset(var->channels_0.process_delay_memory, 0, sizeof(float) * (kOverlapSize * (num_bands_ - 1)));

    // var->channels_1 = var->channels_0;




    memset(var->filter_bank_states_heap_.real, 0, sizeof(float) * (kFftSize));
    memset(var->filter_bank_states_heap_.imag, 0, sizeof(float) * (kFftSize));
    memset(var->filter_bank_states_heap_.extended_frame, 0, sizeof(float) * (kFftSize));




    memset(var->upper_band_gains_heap_, 0, sizeof(float) * (num_channels_));
    memset(var->upper_band_gains_heap_, 0, sizeof(float) * (num_channels_));
    memset(var->upper_band_gains_heap_, 0, sizeof(float) * (num_channels_));
}

void NoiseSuppressor_AggregateWienerFilters(NoiseSuppressor* var, float* filter) {
    float filter0[kFftSizeBy2Plus1];

    filter0 = var->channels_0.wiener_filter.filter_;
    memcpy(filter, filter0, sizeof(float) * (kFftSizeBy2Plus1));

    // float filter_ch[filter_ch];
    // filter_ch = var->channels_1.wiener_filter.filter_;
    // for (int k = 0; k < kFftSizeBy2Plus1; ++k) {
    //     filter[k] = min_local(filter[k], filter_ch[k]);
    // }

}

void NoiseSuppressor_Analyze(NoiseSuppressor* var, const AudioBuffer& audio) {
    // Prepare the noise estimator for the analysis stage.
    NoiseEstimator_PrepareAnalysis(var->channels_0.noise_estimator);
    // NoiseEstimator_PrepareAnalysis(var->channels_1.noise_estimator);

    // Check for zero frames.
    bool zero_frame = true;
    float y_band0[kNsFrameSize];
    memcpy(y_band0, audio.split_bands_const[0], sizeof(float) * (kNsFrameSize));
    float energy = ComputeEnergyOfExtendedFrame_split(y_band0, var->channels_0.analyze_analysis_memory, kNsFrameSize, (kFftSize - kNsFrameSize));
    if (energy > 0.f) {
        zero_frame = false;
        break;
    }

    if (zero_frame) {
        // We want to avoid updating statistics in this case:
        // Updating feature statistics when we have zeros only will cause
        // thresholds to move towards zero signal situations. This in turn has the
        // effect that once the signal is "turned on" (non-zero values) everything
        // will be treated as speech and there is no noise suppression effect.
        // Depending on the duration of the inactive signal it takes a
        // considerable amount of time for the system to learn what is noise and
        // what is speech.
        return;
    }

    // Only update analysis counter for frames that are properly analyzed.
    if (++var->num_analyzed_frames_ < 0) {
        var->num_analyzed_frames_ = 0;
    }
    // Analyze all channels.
    for (int ch = 0; ch < num_channels_; ++ch) {
        const ChannelState& ch_p = var->channels_0;

        float extended_frame[kFftSize];
        FormExtendedFrame(y_band0, ch_p.analyze_analysis_memory, extended_frame);
        ApplyFilterBankWindow(extended_frame);

        // Compute the magnitude spectrum.
        float fft_out[kFftSize + 2];
        FFT_table(extended_frame, fft_out);

        float signal_spectrum[kFftSizeBy2Plus1];
        ComputeMagnitudeSpectrum(fft_out, signal_spectrum);

        // Compute energies.
        float signal_energy = 0.f;
        float signal_spectral_sum = 0.f;
        for (int i = 0; i < kFftSizeBy2Plus1; ++i) {
            signal_energy += fft_out[0] * fft_out[0] + fft_out[1] * fft_out[1];
            fft_out += 2

            signal_spectral_sum += signal_spectrum[i];
        }
        signal_energy /= kFftSizeBy2Plus1;

        // Estimate the noise spectra and the probability estimates of speech
        // presence.
        NoiseEstimator_PreUpdate(ch_p.noise_estimator, var->suppression_params_, var->num_analyzed_frames_,
                                  signal_spectrum, signal_spectral_sum);
        
        float post_snr[kFftSizeBy2Plus1], prior_snr[kFftSizeBy2Plus1];
        // WienerFilter_Update(ch_p.wiener_filter, var->suppression_params_, var->num_analyzed_frames_, ch_p.noise_estimator.noise_spectrum_,
        //                    ch_p.noise_estimator.prev_noise_spectrum_, ch_p.noise_estimator.parametric_noise_spectrum_, signal_spectrum);

        // NoiseEstimator_PostUpdate(ch_p.noise_estimator, ch_p.speech_probability_estimator.speech_probability_, signal_spectrum);

        ComputeSnr(ch_p.wiener_filter.filter_, ch_p.prev_analysis_signal_spectrum, signal_spectrum, ch_p.noise_estimator.prev_noise_spectrum_,
                            ch_p->noise_estimator.noise_spectrum_, prior_snr, post_snr);

        SpeechProbabilityEstimator_Update(ch_p.speech_probability_estimator, var->num_analyzed_frames_, prior_snr, post_snr, 
                            ch_p.noise_estimator.conservative_noise_spectrum_, signal_spectrum, signal_spectral_sum, signal_energy);

        NoiseEstimator_PostUpdate(ch_p.noise_estimator, ch_p.speech_probability_estimator.speech_probability_, signal_spectrum);

        // Store the magnitude spectrum to make it avalilable for the process method.
        memcpy(ch_p.prev_analysis_signal_spectrum, signal_spectrum, sizeof(float) * (kFftSizeBy2Plus1));

    }

}

void NoiseSuppressor_Process(NoiseSuppressor* var, AudioBuffer* audio) {
    const FilterBankState& filter_bank_states = var->filter_bank_states_heap_;
    float fft_out[kFftSize + 2];
    float signal_spectrum[kFftSizeBy2Plus1];
    float y_band0[kNsFrameSize];
    // Compute the suppression filters for all channels.
    for (int ch = 0; ch < num_channels_; ++ch) {
        // Form an extended frame and apply analysis filter bank windowing.
        memcpy(y_band0, audio->split_bands[0], sizeof(float) * (kNsFrameSize));

        FormExtendedFrame(y_band0, var->channels_0.process_analysis_memory, filter_bank_states.extended_frame);
        ApplyFilterBankWindow(filter_bank_states.extended_frame);
        var->energies_before_filtering_heap_[ch] = ComputeEnergyOfExtendedFrame_long(filter_bank_states.extended_frame, kFftSize);

        // Perform filter bank analysis and compute the magnitude spectrum.
        FFT_table(filter_bank_states.extended_frame, fft_out);
        ComputeMagnitudeSpectrum(fft_out, signal_spectrum);
        // Compute the frequency domain gain filter for noise attenuation.
        WienerFilter_Update(var->channels_0.wiener_filter, var->suppression_params_, var->num_analyzed_frames_, var->channels_0.noise_estimator.noise_spectrum_,
        var->channels_0.noise_estimator.prev_noise_spectrum_, var->channels_0.noise_estimator.parametric_noise_spectrum_, signal_spectrum);

        if (num_bands_ > 1) {
            // Compute the time-domain gain for attenuating the noise in the upper bands.  
            var->upper_band_gains[ch] = ComputeUpperBandsGain(var->suppression_params_.minimum_attenuating_gain, var->channels_0.wiener_filter.filter_,
                                       var->channels_0.speech_probability_estimator.speech_probability_, var->channels_0.prev_analysis_signal_spectrum, signal_spectrum);
        }
    }

    // Only do the below processing if the output of the audio processing module is used.
    if (!var->capture_output_used_) {
        return;
    }

    // Aggregate the Wiener filters for all channels.
    float filter_data[kFftSizeBy2Plus1];
    if (num_channels_ == 1) {
        filter_data = var->channels_0.wiener_filter.filter_;
    }

    for (int ch = 0; ch < num_channels_; ++ch) {
        // Apply the filter to the lower band.
        for (int i = 0; i < kFftSizeBy2Plus1; ++i) {
            fft_out[2 * i] = filter_data[i] * fft_out[2 * i];
            fft_out[2 * i + 1] = filter_data[i] * fft_out[2 * i + 1];
        }
    }

    // Perform filter bank synthesis
    for (int ch = 0; ch < num_channels_; ++ch) {
        IFFT_table(fft_out, filter_bank_states.extended_frame);
    }

    for (int ch = 0; ch < num_channels_; ++ch) {
        float energy_after_filtering = ComputeEnergyOfExtendedFrame_long(filter_bank_states.extended_frame, kFftSize);

        // Apply synthesis window.
        ApplyFilterBankWindow(filter_bank_states.extended_frame);
        // Compute the adjustment of the noise attenuation filter based on the effect of the attenuation.
        var->gain_adjustments[ch] = ComputeOverallScalingFactor(var->suppression_params_, var->num_analyzed_frames_,
                    var->channels_0.speech_probability_estimator.prior_speech_prob_, var->energies_before_filtering_heap_[ch], energy_after_filtering);

    }
    // Select and apply adjustment of the noise attenuation filter based on the effect of the attenuation.
    float gain_adjustment = var->gain_adjustments[0];
    if (num_channels_ > 1) {
        for (int ch = 1; ch < num_channels_; ++ch) {
            gain_adjustment = min_local(gain_adjustment, var->gain_adjustments[ch]);
        }
    }

    for (int i = 0; i < kFftSize; ++i) {
        filter_bank_states.extended_frame[i] = gain_adjustment * filter_bank_states.extended_frame[i];
    }
    // Use overlap-and-add to form the output frame of the lowest band.
    for (int ch = 0; ch < num_channels_; ++ch) {
        memcpy(y_band0, audio->split_bands[0], sizeof(float) * (kNsFrameSize));
        OverlapAndAdd(filter_bank_states.extended_frame, var->channels_0.process_synthesis_memory, y_band0);
    }
    
    float y_band[kNsFrameSize];
    if (num_bands_ > 1) {
        // Select the noise attenuating gain to apply to the upper band.
        float upper_band_gain = var->upper_band_gains_heap_[0];
        for (int ch = 1; ch < num_channels_; ++ch) {
            upper_band_gain = min_local(upper_band_gain, var->upper_band_gains_heap_[ch]);
        }

        // Process the upper bands.
        for (int ch = 0; ch < num_channels_; ++ch) {
            for (int b = 1; b < num_bands_; ++b) {
            // Delay the upper bands to match the delay of the filterbank applied to the lowest band.
            float delayed_frame[kNsFrameSize];
            memcpy(y_band, audio->split_bands[b], sizeof(float) * (kNsFrameSize));

            DelaySignal(y_band, var->channels_0.process_delay_memory[b - 1], delayed_frame);
            // Apply the time-domain noise-attenuating gain.
            for (int j = 0; j < kNsFrameSize; j++) {
                y_band[j] = upper_band_gain * delayed_frame[j];
            }
            }
        }
    } 

    // Limit the output the allowed range.
    for (int ch = 0; ch < num_channels_; ++ch) {
        for (int b = 0; b < num_bands_; ++b) {
            memcpy(y_band, audio->split_bands[b], sizeof(float) * (kNsFrameSize));
            for (int j = 0; j < kNsFrameSize; j++) {
                y_band[j] = min_local(max_local(y_band[j], -32768.f), 32767.f);
            }
        }
    }

}



