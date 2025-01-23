
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "channel_buffer.h"
#include "audio_buffer.h"

#define kSamplesPer32kHzChannel 320
#define kSamplesPer48kHzChannel 480
#define kMaxSamplesPerChannel (kMaxSampleRate / 100)

int NumBandsFromFramesPerChannel(int num_frames) {
    if (num_frames == kSamplesPer32kHzChannel) {
        return 2;
    } 
    if (num_frames == kSamplesPer48kHzChannel) {
        return 3;
    }
    return 1;
}

void AudioBuffer_init(AudioBuffer* var, int input_rate, int input_num_channels, int buffer_rate, int buffer_num_channels,
                        int output_rate, int output_num_channels) 
{
    var->input_num_frames_ = input_rate / 100;
    var->input_num_channels_ = input_num_channels;
    var->buffer_num_frames_ = buffer_rate / 100;
    var->buffer_num_channels_ = buffer_num_channels;
    var->output_num_frames_ = output_rate / 100;
    var->output_num_channels_ = 0;

    var->num_channels_ = buffer_num_channels;
    var->num_bands_ = NumBandsFromFramesPerChannel(var->buffer_num_frames_);
    var->num_split_frames_ = var->buffer_num_frames_ / var->num_bands_;

    var->downmix_by_averaging_ = true;
    var->channel_for_downmixing_ = 0;

    ChannelBuffer_init(var->data_, var->buffer_num_frames_, var->buffer_num_channels_, var->num_bands_);
    if (var->num_bands_ > 1) {
        ChannelBuffer_init(var->split_data_, var->buffer_num_frames_, var->buffer_num_channels_, var->num_bands_);
        SplittingFilter_init(var->splitting_filter_, var->buffer_num_channels_, var->num_bands_);
    }

}

void set_downmixing_to_specific_channel(int channel) {
    var->downmix_by_averaging_ = false;
    var->channel_for_downmixing_ = min_local(channel, var->input_num_channels_ - 1);
}

void set_downmixing_by_averaging() {
    var->downmix_by_averaging_ = true;
}

void AudioBuffer_CopyFrom(AudioBuffer* var, float* stacked_data, const StreamConfig& stream_config) {
    stream_config.num_frames_ = var->input_num_frames_;
    stream_config.num_channels_ = var->input_num_channels_;
    RestoreNumChannels(var);

    bool downmix_needed = var->input_num_channels_ > 1 && var->num_channels_ == 1;

    if (downmix_needed) {
        float downmix[kMaxSamplesPerChannel];
        if (var->downmix_by_averaging_) {
            float kOneByNumChannels = 1.f / var->input_num_channels_;
            for (int i = 0; i < var->input_num_frames_; ++i) {
                float value = stacked_data[0][i];
                for (int j = 1; j < var->input_num_channels_; ++j) {
                    value += stacked_data[j][i];
                }
                downmix[i] = value * kOneByNumChannels;
            }
        }
        float* downmixed_data = var->downmix_by_averaging_ ? downmix : stacked_data[var->channel_for_downmixing_];
        float* data_to_convert = downmixed_data;
        FloatToFloatS16(data_to_convert, var->buffer_num_frames_, var->data_.channels[0]);
    } else {
        for (int i = 0; i < var->num_channels_; ++i) {
            FloatToFloatS16(stacked_data[i], var->buffer_num_frames_, var->data_.channels[i]);
        }

    }
}

void AudioBuffer_CopyTotwo(AudioBuffer* var, float* stacked_data, const StreamConfig& stream_config) {
    stream_config.num_frames_ = var->output_num_frames_;

    for (int i = 0; i < var->num_channels_; ++i) {
        FloatS16ToFloat(var->data_.channels[i], var->buffer_num_frames_, stacked_data[i]);
    }

    for (int i = var->num_channels_; i < stream_config.num_channels_; ++i) {
        memcpy(stacked_data[i], stacked_data[0], var->output_num_frames_ * sizeof(**stacked_data));
    }

}

void AudioBuffer_CopyToone(AudioBuffer* var) {
    var->buffer_num_frames_ = var->output_num_frames_;

    for (int i = 0; i < var->num_channels_; ++i) {
        memcpy(buffer->channels()[i], data_->channels()[i], var->buffer_num_frames_ * sizeof(**buffer->channels()));
    }

    for (int i = var->num_channels_; i < buffer->num_channels(); ++i) {
        memcpy(buffer->channels()[i], buffer->channels()[0], var->output_num_frames_ * sizeof(**buffer->channels()));
    }

}

void RestoreNumChannels(AudioBuffer* var) {
    var->num_channels_ = var->buffer_num_channels_;
    var->data_.num_channels_ = var->buffer_num_channels_;
    var->split_data_.num_channels_ = var->buffer_num_channels_;
}

void set_num_channels(AudioBuffer* var, int num_channels) {
    var->buffer_num_channels_ = num_channels;
    var->num_channels_ = num_channels;
    var->data_.num_channels_ = num_channels;
    var->split_data_.num_channels_ = num_channels;
}

void AudioBuffer_CopyFrom_two(AudioBuffer* var, int16_t* interleaved_data, const StreamConfig& stream_config) {
    stream_config.num_channels_ = var->input_num_channels_;
    stream_config.num_frames_ = var->input_num_frames_;
    RestoreNumChannels(var);

    int16_t* interleaved = interleaved_data;
    if (var->num_channels_ == 1) {
        if (var->input_num_channels_ == 1) {
            S16ToFloatS16(interleaved, var->input_num_frames_, var->data_.channels[0]);
        } else {
            float float_buffer[kMaxSamplesPerChannel];
            float* downmixed_data = var->data_.channels[0];
            if (var->downmix_by_averaging_) {
                for (int j = 0, k = 0; j < var->input_num_frames_; ++j) {
                    int32_t sum = 0;
                    for (int i = 0; i < var->input_num_channels_; ++i, ++k) {
                        sum += interleaved[k];
                    }
                    downmixed_data[j] = sum / (int16_t)(var->input_num_channels_);
                }
            } else {
                for (int j = 0, k = var->channel_for_downmixing_; j < var->input_num_frames_; ++j, k += var->input_num_channels_) {
                    downmixed_data[j] = interleaved[k];
                }
            }
        }
    } else {
        for (int i = 0; i < var->num_channels_; ++i) {
            for (int j = 0, k = i; j < var->input_num_frames_; ++j, k += var->num_channels_) {
                var->data_.channels[i][j] = interleaved[k];
            }
        }

    }

}

void AudioBuffer_CopyTo_two(AudioBuffer* var, const StreamConfig& stream_config, int16_t* interleaved_data) {
    int config_num_channels = stream_config.num_channels;
    stream_config.num_frames = var->output_num_frames_;

    int16_t* interleaved = interleaved_data;
    if (var->num_channels_ == 1) {
        float float_buffer[kMaxSamplesPerChannel];
        float* deinterleaved = var->data_.channels[0];

        if (var->config_num_channels == 1) {
            for (int j = 0; j < var->output_num_frames_; ++j) {
                interleaved[j] = FloatS16ToS16(deinterleaved[j]);
            }
        } else {
            for (int i = 0, k = 0; i < var->output_num_frames_; ++i) {
                float tmp = FloatS16ToS16(deinterleaved[i]);
                for (int j = 0; j < config_num_channels; ++j, ++k) {
                    interleaved[k] = tmp;
                }
            }
        }
    } else {
        for (int i = 0; i < var->num_channels_; ++i) {
            for (int k = 0, j = i; k < var->output_num_frames_; ++k, j += config_num_channels) {
                interleaved[j] = FloatS16ToS16(var->data_.channels[i][k]);
            }
        }

        for (int i = var->num_channels_; i < config_num_channels; ++i) {
            for (int j = 0, k = i, n = var->num_channels_; j < var->output_num_frames_; ++j, k += config_num_channels, n += config_num_channels) {
                interleaved[k] = interleaved[n];
            }
        }
    }
}

void SplitIntoFrequencyBands(AudioBuffer* var) {
    SplittingFilter_Analysis(var->splitting_filter_, var->data_, var->split_data_);
}

void MergeFrequencyBands(AudioBuffer* var) {
    SplittingFilter_Synthesis(var->splitting_filter_, var->split_data_, var->data_);
}

void ExportSplitChannelData(AudioBuffer* var, int channel, int16_t* split_band_data) {
    for (int k = 0; k < var->num_bands_; ++k) {
        float* band_data = split_bands_const(channel)[k];

        for (int i = 0; i < var->num_split_frames_; ++i) {
            split_band_data[k][i] = FloatS16ToS16(band_data[i]);
        }
    }
}

void ImportSplitChannelData(AudioBuffer* var, int channel, int16_t* split_band_data) {
    for (size_t k = 0; k < var->num_bands_; ++k) {
        float* band_data = split_bands(channel)[k];

        for (size_t i = 0; i < var->num_split_frames_; ++i) {
            band_data[i] = split_band_data[k][i];
        }
    }
}




