#ifndef MODULES_AUDIO_PROCESSING_NS_SUPPRESSION_PARAMS_H_
#define MODULES_AUDIO_PROCESSING_NS_SUPPRESSION_PARAMS_H_

#include "ns_common.h"
#include "signal_model.h"


#define kHistogramSize 1000

typedef struct
{
    int lrt_[kHistogramSize];
    int spectral_flatness_[kHistogramSize];
    int spectral_diff_[kHistogramSize];

} Histograms;

  void Histograms_Clear(Histograms* var);
  void Histograms_Update(const SignalModel& features_, Histograms* var);

  #endif  // MODULES_AUDIO_PROCESSING_NS_QUANTILE_NOISE_ESTIMATOR_H_
