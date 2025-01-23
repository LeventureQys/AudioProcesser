#pragma once

// APM processes audio in chunks of about 10 ms. See GetFrameSize() for
// details.
static constexpr int kChunkSizeMs = 10;

// Returns floor(sample_rate_hz/100): the number of samples per channel used
// as input and output to the audio processing module in calls to
// ProcessStream, ProcessReverseStream, AnalyzeReverseStream, and
// GetLinearAecOutput.
//
// This is exactly 10 ms for sample rates divisible by 100. For example:
//  - 48000 Hz (480 samples per channel),
//  - 44100 Hz (441 samples per channel),
//  - 16000 Hz (160 samples per channel).
//
// Sample rates not divisible by 100 are received/produced in frames of
// approximately 10 ms. For example:
//  - 22050 Hz (220 samples per channel, or ~9.98 ms per frame),
//  - 11025 Hz (110 samples per channel, or ~9.98 ms per frame).
// These nondivisible sample rates yield lower audio quality compared to
// multiples of 100. Internal resampling to 10 ms frames causes a simulated
// clock drift effect which impacts the performance of (for example) echo
// cancellation.
static int AudioProcessing_GetFrameSize(int sample_rate_hz) { return sample_rate_hz / 100; }

class StreamConfig {
public:
	// sample_rate_hz: The sampling rate of the stream.
	// num_channels: The number of audio channels in the stream.
	StreamConfig(int sample_rate_hz = 0, size_t num_channels = 0)
		: sample_rate_hz_(sample_rate_hz),
		num_channels_(num_channels),
		num_frames_(calculate_frames(sample_rate_hz)) {}

	void set_sample_rate_hz(int value) {
		sample_rate_hz_ = value;
		num_frames_ = calculate_frames(value);
	}
	void set_num_channels(size_t value) { num_channels_ = value; }

	int sample_rate_hz() const { return sample_rate_hz_; }

	// The number of channels in the stream.
	size_t num_channels() const { return num_channels_; }

	size_t num_frames() const { return num_frames_; }
	size_t num_samples() const { return num_channels_ * num_frames_; }

	bool operator==(const StreamConfig& other) const {
		return sample_rate_hz_ == other.sample_rate_hz_ &&
			num_channels_ == other.num_channels_;
	}

	bool operator!=(const StreamConfig& other) const { return !(*this == other); }

private:
	static size_t calculate_frames(int sample_rate_hz) {
		return static_cast<size_t>(AudioProcessing_GetFrameSize(sample_rate_hz));
	}

	int sample_rate_hz_;
	size_t num_channels_;
	size_t num_frames_;
};