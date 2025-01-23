#ifndef MODULES_AUDIO_PROCESSING_NS_SUPPRESSION_PARAMS_H_
#define MODULES_AUDIO_PROCESSING_NS_SUPPRESSION_PARAMS_H_


#include <math.h>
#include "ns_common.h"
//#include "user_task.h"


typedef struct
{
  float lrt;
  float spectral_diff;
  float spectral_flatness;
  // Log LRT factor with time-smoothing.
  float avg_log_lrt[kFftSizeBy2Plus1];
} SignalModel;

void SignalModel_init(SignalModel* var);

#endif  // MODULES_AUDIO_PROCESSING_NS_QUANTILE_NOISE_ESTIMATOR_H_
