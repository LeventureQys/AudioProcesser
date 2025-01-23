#ifndef MODULES_AUDIO_PROCESSING_NS_FAST_MATH_H_
#define MODULES_AUDIO_PROCESSING_NS_FAST_MATH_H_

// Sqrt approximation.
float SqrtFastApproximation(float f);

// Log base conversion log(x) = log2(x)/log2(e).
float LogApproximation(float x);
void LogApproximation(float* x, float* y);

// 2^x approximation.
float Pow2Approximation(float p);

// x^p approximation.
float PowApproximation(float x, float p);

// e^x approximation.
float ExpApproximation(float x);
void ExpApproximation(float* x, float* y);
void ExpApproximationSignFlip(float* x, float* y);

#endif  // MODULES_AUDIO_PROCESSING_NS_FAST_MATH_H_
