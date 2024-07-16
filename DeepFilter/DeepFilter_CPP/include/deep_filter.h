
#define DEEP_FILTER_MAJOR 0
#define DEEP_FILTER_MINOR 3
#define DEEP_FILTER_PATCH 2


#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <new>

struct DFState;

extern "C" {

/// Create a DeepFilterNet Model
///
/// Args:
///     - path: File path to a DeepFilterNet tar.gz onnx model
///     - atten_lim: Attenuation limit in dB.
///
/// Returns:
///     - DF state doing the full processing: stft, DNN noise reduction, istft.
DFState *df_create(const char *path, float atten_lim);

/// Get DeepFilterNet frame size in samples.
uintptr_t df_get_frame_length(DFState *st);

/// Get DeepFilterNet frame size in samples.
///
/// Args:
///     - lim_db: New attenuation limit in dB.
void df_set_atten_lim(DFState *st, float lim_db);

/// Processes a chunk of samples.
///
/// Args:
///     - df_state: Created via df_create()
///     - input: Input buffer of length df_get_frame_length()
///     - output: Output buffer of length df_get_frame_length()
///
/// Returns:
///     - Local SNR of the current frame.
float df_process_frame(DFState *st, float *input, float *output);

/// Free a DeepFilterNet Model
void df_free(DFState *model);

} // extern "C"