#include <stdio.h>
#include <string.h>
#include <math.h>
#include "splitting_filter.h"

#include "channel_buffer.h"
#include "signal_processing_library.h"

#define kSamplesPerBand 160
#define kTwoBandFilterSamplesPerFrame 320

void TwoBandsAnalysis(SplittingFilter* var, ChannelBuffer* data, ChannelBuffer* bands);
void TwoBandsSynthesis(SplittingFilter* var, ChannelBuffer* bands, ChannelBuffer* data);
void ThreeBandsAnalysis(SplittingFilter* var, ChannelBuffer* data, ChannelBuffer* bands);
void ThreeBandsSynthesis(SplittingFilter* var, ChannelBuffer* bands, ChannelBuffer* data);


void TwoBandsAnalysis(SplittingFilter* var, ChannelBuffer* data, ChannelBuffer* bands) {
    RTC_DCHECK_EQ(data->num_frames_, kTwoBandFilterSamplesPerFrame);

    for (int i = 0; i < data->num_channels_; ++i) {
        int16_t bands16[2][kSamplesPerBand], full_band16[kTwoBandFilterSamplesPerFrame];
        FloatS16ToS16(data->channels[0], kTwoBandFilterSamplesPerFrame, full_band16);
        WebRtcSpl_AnalysisQMF(full_band16, data->num_frames_, bands16[0], bands16[1],
                              var->two_bands_states_.analysis_state1, var->two_bands_states_.analysis_state2);
        S16ToFloatS16(bands16[0], kSamplesPerBand, bands->channels[0]);
        S16ToFloatS16(bands16[1], kSamplesPerBand, bands->channels[1]);
    }
}

void TwoBandsSynthesis(SplittingFilter* var, ChannelBuffer* bands, ChannelBuffer* data) {
    RTC_DCHECK_EQ(data->num_frames_, kTwoBandFilterSamplesPerFrame);

    for (int i = 0; i < data->num_channels_; ++i) {
        int16_t bands16[2][kSamplesPerBand], full_band16[kTwoBandFilterSamplesPerFrame];
        FloatS16ToS16(bands->channels[0], kSamplesPerBand, bands16[0]);
        FloatS16ToS16(bands->channels[1], kSamplesPerBand, bands16[1]);
        WebRtcSpl_SynthesisQMF(bands16[0], bands16[1], bands->num_frames_per_band_, full_band16,
                              var->two_bands_states_.synthesis_state1, var->two_bands_states_.synthesis_state2);
        S16ToFloatS16(full_band16, kTwoBandFilterSamplesPerFrame, data->channels[0]);
    }
}

void ThreeBandsAnalysis(SplittingFilter* var, ChannelBuffer* data, ChannelBuffer* bands) {
    RTC_DCHECK_EQ(data->num_frames_, kFullBandSize);
    RTC_DCHECK_EQ(bands->num_frames_, kFullBandSize);
    RTC_DCHECK_EQ(bands->num_bands_, kNumBands);
    RTC_DCHECK_EQ(bands->num_frames_per_band_, kSplitBandSize);

    for (int i = 0; i < bands->num_channels_; ++i) {
        ThreeBandFilterBank_Analysis(var->three_band_filter_banks_, data->channels_view, bands->bands_view);
    }
}

void ThreeBandsSynthesis(SplittingFilter* var, ChannelBuffer* bands, ChannelBuffer* data) {
    RTC_DCHECK_EQ(data->num_frames_, kFullBandSize);
    RTC_DCHECK_EQ(bands->num_frames_, kFullBandSize);
    RTC_DCHECK_EQ(bands->num_bands_, kNumBands);
    RTC_DCHECK_EQ(bands->num_frames_per_band_, kSplitBandSize);

    for (int i = 0; i < data->num_channels_; ++i) {
        ThreeBandFilterBank_Synthesis(var->three_band_filter_banks_, bands->bands_view, data->channels_view);
    }
}

void SplittingFilter_init(SplittingFilter* var, int num_channels, int num_bands) {
    var->num_bands_ = num_bands;
    if (num_bands == 3) {
        ThreeBandFilterBank_init(var->three_band_filter_banks_);
    }
    if (num_bands == 2) {
        memset(var->two_bands_states_.analysis_state1, 0, sizeof(float) * (kStateSize));
        memset(var->two_bands_states_.analysis_state2, 0, sizeof(float) * (kStateSize));
        memset(var->two_bands_states_.synthesis_state1, 0, sizeof(float) * (kStateSize));
        memset(var->two_bands_states_.synthesis_state2, 0, sizeof(float) * (kStateSize));
    }
}

void SplittingFilter_Analysis(SplittingFilter* var, ChannelBuffer* data, ChannelBuffer* bands) {
    RTC_DCHECK_EQ(var->num_bands_, bands->num_bands_);
    RTC_DCHECK_EQ(data->num_channels_, bands->num_channels_);
    RTC_DCHECK_EQ(data->num_frames_, bands->num_frames_per_band_ * bands->num_bands_);
    if (bands->num_bands_ == 2) {
        TwoBandsAnalysis(data, bands);
    } else if (bands->num_bands_ == 3) {
        ThreeBandsAnalysis(data, bands);
    }
}

void SplittingFilter_Synthesis(SplittingFilter* var, ChannelBuffer* bands, ChannelBuffer* data) {
    RTC_DCHECK_EQ(var->num_bands_, bands->num_bands_);
    RTC_DCHECK_EQ(data->num_channels_, bands->num_channels_);
    RTC_DCHECK_EQ(data->num_frames_, bands->num_frames_per_band_ * bands->num_bands_);
    if (bands->num_bands_ == 2) {
        TwoBandsAnalysis(bands, data);
    } else if (bands->num_bands_ == 3) {
        ThreeBandsAnalysis(bands, data);
    }
}


