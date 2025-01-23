#ifndef MODULES_AUDIO_PROCESSING_NS_SIGNAL_MODEL_ESTIMATOR_H_
#define MODULES_AUDIO_PROCESSING_NS_SIGNAL_MODEL_ESTIMATOR_H_


#include "histograms.h"
#include "ns_common.h"
#include "prior_signal_model.h"
#include "prior_signal_model_estimator.h"
#include "signal_model.h"

#define kOneByFftSizeBy2Plus1 (1.f / kFftSizeBy2Plus1)


typedef struct
{
    float diff_normalization_;
    float signal_energy_sum_;
    int histogram_analysis_counter_;

    SignalModel features_;
    Histograms histograms_;
    PriorSignalModel prior_model_;

} SignalModelEstimator;


void SignalModelEstimator_init(SignalModelEstimator* var, float lrt_initial_value);
void SignalModelEstimator_AdjustNormalization(SignalModelEstimator* var, int32_t num_analyzed_frames,float signal_energy);
void SignalModelEstimator_Update(SignalModelEstimator* var, float* prior_snr, float* post_snr, 
    float* conservative_noise_spectrum, float* signal_spectrum, float signal_spectral_sum, float signal_energy);

#endif  // MODULES_AUDIO_PROCESSING_NS_SIGNAL_MODEL_ESTIMATOR_H_
