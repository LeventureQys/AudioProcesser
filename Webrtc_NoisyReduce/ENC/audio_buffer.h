#ifndef MODULES_AUDIO_PROCESSING_AUDIO_BUFFER_H_
#define MODULES_AUDIO_PROCESSING_AUDIO_BUFFER_H_

#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include "channel_buffer.h"
#include "stream_config.h"
#include "splitting_filter.h"

#define kSplitBandSize 160
#define kMaxSampleRate 384000
#define kMaxSplitFrameLength 160
#define kMaxNumBands 3

enum Band { kBand0To8kHz = 0, kBand8To16kHz = 1, kBand16To24kHz = 2 };

typedef struct
{ 
    int input_rate;
    int input_num_frames_;
    int input_num_channels_;

    int buffer_rate;
    int buffer_num_frames_;
    int buffer_num_channels_;

    int output_rate;
    int output_num_frames_;
    int output_num_channels_;

    int num_channels_;
    int num_bands_;
    int num_split_frames_;

    ChannelBuffer data_;
    ChannelBuffer split_data_;
    SplittingFilter splitting_filter_;

    bool downmix_by_averaging_;
    int channel_for_downmixing_;

} AudioBuffer;








