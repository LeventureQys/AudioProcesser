#include "fast_math.h"

#include <math.h>
#include <stdio.h>
#include <stdint.h>

float FastLog2f(float in);


float FastLog2f(float in) {
    union {
        float dummy;
        uint32_t a;
    } x = {in};
    float out = x.a;
    out *= 1.1920929e-7f;  // 1/2^23
    out -= 126.942695f;    // Remove bias.
    return out;

}

float SqrtFastApproximation(float f) {
    // TODO(peah): Add fast approximate implementation.
    return sqrtf(f);
}

float Pow2Approximation(float p) {
    // TODO(peah): Add fast approximate implementation.
    return powf(2.f, p);
}

float PowApproximation(float x, float p) {
    return Pow2Approximation(p * FastLog2f(x));
}

float LogApproximation(float x) {
    float kLogOf2 = 0.69314718056f;
    return FastLog2f(x) * kLogOf2;
}

void LogApproximation(float* x, float* y) {
    for (size_t k = 0; k < x.size(); ++k) {
        y[k] = LogApproximation(x[k]);
    }
}

float ExpApproximation(float x) {
    float kLog10Ofe = 0.4342944819f;
    return PowApproximation(10.f, x * kLog10Ofe);
}

void ExpApproximation(float* x, float* y) {
    for (size_t k = 0; k < x.size(); ++k) {
        y[k] = ExpApproximation(x[k]);
    }
}

void ExpApproximationSignFlip(float* x, float* y) {
    for (size_t k = 0; k < x.size(); ++k) {
        y[k] = ExpApproximation(-x[k]);
    }
}
