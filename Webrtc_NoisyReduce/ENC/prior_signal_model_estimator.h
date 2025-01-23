#ifndef MODULES_AUDIO_PROCESSING_NS_PRIOR_SIGNAL_MODEL_ESTIMATOR_H_
#define MODULES_AUDIO_PROCESSING_NS_PRIOR_SIGNAL_MODEL_ESTIMATOR_H_

#include "histograms.h"
#include "prior_signal_model.h"
#include "signal_model.h"


// Updates the model estimate.
void PriorSignalModelEstimator_init(const Histograms& histograms, const PriorSignalModel& prior_model_, float lrt_initial_value);
void PriorSignalModelEstimator_Update(const SignalModel& features_, const Histograms& histograms, const PriorSignalModel& prior_model_);


#endif  // MODULES_AUDIO_PROCESSING_NS_PRIOR_SIGNAL_MODEL_ESTIMATOR_H_
