#ifndef MODULES_AUDIO_PROCESSING_NS_PRIOR_SIGNAL_MODEL_ESTIMATOR_H_
#define MODULES_AUDIO_PROCESSING_NS_PRIOR_SIGNAL_MODEL_ESTIMATOR_H_

#include <math.h>
#include "ns_common.h"
//#include "user_task.h"

#define kSimult 3

typedef struct
{
    float density_[kSimult * kFftSizeBy2Plus1];
    float log_quantile_[kSimult * kFftSizeBy2Plus1];
    float quantile_[kFftSizeBy2Plus1];
    int counter_[kSimult];
    int num_updates_;

} QuantileNoiseEstimator;

// Estimate noise.
void QuantileNoiseEstimator_init(QuantileNoiseEstimator* var);
void QuantileNoiseEstimator_Estimate(QuantileNoiseEstimator* var, float* signal_spectrum, float* noise_spectrum);

#endif  // MODULES_AUDIO_PROCESSING_NS_QUANTILE_NOISE_ESTIMATOR_H_
