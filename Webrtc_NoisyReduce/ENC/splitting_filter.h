#ifndef MODULES_AUDIO_PROCESSING_SPLITTING_FILTER_H_
#define MODULES_AUDIO_PROCESSING_SPLITTING_FILTER_H_

#include <stdio.h>
#include <string.h>
#include "ns_common.h"
#include "channel_buffer.h"
#include "three_band_filter_bank.h"

#define kStateSize 6

typedef struct
{
    int analysis_state1[kStateSize];
    int analysis_state2[kStateSize];
    int synthesis_state1[kStateSize];
    int synthesis_state2[kStateSize];
} TwoBandsStates;

// Splitting filter which is able to split into and merge from 2 or 3 frequency
// bands. The number of channels needs to be provided at construction time.
//
// For each block, Analysis() is called to split into bands and then Synthesis()
// to merge these bands again. The input and output signals are contained in
// ChannelBuffers and for the different bands an array of ChannelBuffers is
// used.

typedef struct
{
    int num_bands_;
    TwoBandsStates two_bands_states_;
    ThreeBandFilterBank  three_band_filter_banks_;
} SplittingFilter;

void SplittingFilter_init(SplittingFilter* var, int num_channels, int num_bands);
void SplittingFilter_Analysis(SplittingFilter* var, ChannelBuffer* data, ChannelBuffer* bands);
void SplittingFilter_Synthesis(SplittingFilter* var, ChannelBuffer* bands, ChannelBuffer* data);

#endif  // MODULES_AUDIO_PROCESSING_SPLITTING_FILTER_H_
