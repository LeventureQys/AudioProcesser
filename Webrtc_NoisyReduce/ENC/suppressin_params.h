#ifndef MODULES_AUDIO_PROCESSING_NS_SUPPRESSION_PARAMS_H_
#define MODULES_AUDIO_PROCESSING_NS_SUPPRESSION_PARAMS_H_

#include "ns_common.h"

typedef struct
{
    float over_subtraction_factor;
    float minimum_attenuating_gain;
    bool use_attenuation_adjustment;
} SuppressionParams;

void SuppressionParams_init(SuppressionParams* var, int suppression_level);

#endif  // MODULES_AUDIO_PROCESSING_NS_QUANTILE_NOISE_ESTIMATOR_H_
