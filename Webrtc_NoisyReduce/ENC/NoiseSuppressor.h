#ifndef MODULES_AUDIO_PROCESSING_NS_NOISE_SUPPRESSOR_H_
#define MODULES_AUDIO_PROCESSING_NS_NOISE_SUPPRESSOR_H_

#include "audio_buffer.h"
#include "noise_estimator.h"
#include "ns_common.h"
#include "speech_probability_estimator.h"
#include "suppression_params.h"
#include "wiener_filter.h"

#define sample_rate_hz 16000
#define num_bands_ (sample_rate_hz / 16000)
#define num_channels_ 1


typedef struct
{
    SpeechProbabilityEstimator speech_probability_estimator;
    WienerFilter wiener_filter;
    NoiseEstimator noise_estimator;

    float prev_analysis_signal_spectrum[kFftSizeBy2Plus1];
    float analyze_analysis_memory[kFftSize - kNsFrameSize];
    float process_analysis_memory[kOverlapSize];
    float process_synthesis_memory[kOverlapSize];
    float process_delay_memory[num_bands_ - 1][kOverlapSize];
} ChannelState;

typedef struct
{
    float real[kFftSize];
    float imag[kFftSize];
    float extended_frame[kFftSize];
} FilterBankState;

typedef struct
{
    SuppressionParams suppression_params_;
    int num_analyzed_frames_;
    bool capture_output_used_;

    float upper_band_gains_heap_[num_channels_];
    float energies_before_filtering_heap_[num_channels_];
    float gain_adjustments_heap_[num_channels_];


    FilterBankState filter_bank_states_heap_;
    ChannelState channels_0;


} NoiseSuppressor;

#endif  // MODULES_AUDIO_PROCESSING_NS_NOISE_SUPPRESSOR_H_











    

