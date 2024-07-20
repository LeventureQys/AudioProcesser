#ifndef DEEP_FILTER_H
#define DEEP_FILTER_H

#include <stdio.h>
#include <stdint.h>

typedef struct DFState DFState;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * Create a DeepFilterNet Model
 *
 * Args:
 *     - path: File path to a DeepFilterNet tar.gz onnx model
 *     - atten_lim: Attenuation limit in dB.
 *
 * Returns:
 *     - DF state doing the full processing: stft, DNN noise reduction, istft.
 */
DFState *df_create(const char *path, float atten_lim);

/**
 * Get DeepFilterNet frame size in samples.
 */
uintptr_t df_get_frame_length(DFState *st);

/**
 * Get DeepFilterNet frame size in samples.
 *
 * Args:
 *     - lim_db: New attenuation limit in dB.
 */
void df_set_atten_lim(DFState *st, float lim_db);

/**
 * Processes a chunk of f32 samples.
 *
 * Args:
 *     - df_state: Created via df_create()
 *     - input: Input buffer of length df_get_frame_length()
 *     - output: Output buffer of length df_get_frame_length()
 *
 * Returns:
 *     - Local SNR of the current frame.
 */
float df_process_frame(DFState *st, float *input, float *output);

/**
 * Processes a chunk of i16 samples.
 *
 * Args:
 *     - df_state: Created via df_create()
 *     - input: Input buffer of length df_get_frame_length()
 *     - output: Output buffer of length df_get_frame_length()
 *
 * Returns:
 *     - Local SNR of the current frame.
 */
float df_process_frame_i16(DFState *st, short *input, short *output);

/**
 * Free a DeepFilterNet Model
 */
void df_free(DFState *model);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif /* DEEP_FILTER_H */
