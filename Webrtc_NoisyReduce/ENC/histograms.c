#include "histograms.h"
#include <math.h>


void Histograms_Clear(Histograms* var) {
    memset(var->lrt_, 0, sizeof(int) * (kHistogramSize));
    memset(var->spectral_flatness_, 0, sizeof(int) * (kHistogramSize));
    memset(var->spectral_diff_, 0, sizeof(int) * (kHistogramSize));
}

void Histograms_Update(const SignalModel& features_, Histograms* var) {
    float kOneByBinSizeLrt = 1.f / kBinSizeLrt;
    if (features_.lrt < kHistogramSize * kBinSizeLrt && features_.lrt >= 0.f) {
        ++var->lrt_[kOneByBinSizeLrt * features_.lrt];
    }

    float kOneByBinSizeSpecFlat = 1.f / kBinSizeSpecFlat;
    if (features_.spectral_flatness < kHistogramSize * kBinSizeSpecFlat &&
        features_.spectral_flatness >= 0.f) {
        ++var->spectral_flatness_[features_.spectral_flatness * kOneByBinSizeSpecFlat];
    }

    float kOneByBinSizeSpecDiff = 1.f / kBinSizeSpecDiff;
    if (features_.spectral_diff < kHistogramSize * kBinSizeSpecDiff &&
        features_.spectral_diff >= 0.f) {
        ++var->spectral_diff_[features_.spectral_diff * kOneByBinSizeSpecDiff];
    }

}

