#ifndef __MCU_FFT_H__
#define __MCU_FFT_H__

/*******************************************************************************
 * API
 ******************************************************************************/
// #if defined(__cplusplus)
// extern "C" {
// #endif /* _cplusplus */
typedef   signed           int int32_t;
typedef unsigned           int uint32_t;

void FFT_table(float *arr, float *result);
void IFFT_table(float *arr, float *result);
void FFT_table_TS(float *arr, float *result);
void IFFT_table_TS(float *arr, float *result);

// #if defined(__cplusplus)
// }
// #endif
/*! @}*/
#endif /* __MCU_FFT_H__ */
