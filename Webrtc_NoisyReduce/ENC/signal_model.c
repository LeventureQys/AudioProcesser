#include<stdio.h>
#include <string.h>
#include "signal_model.h"


void SignalModel_init(SignalModel* var) {
    float kSfFeatureThr = 0.5f;

    var->lrt = kLtrFeatureThr;
    var->spectral_flatness = kSfFeatureThr;
    var->spectral_diff = kSfFeatureThr;

    int i;
    for (i = 0; i < kFftSizeBy2Plus1; i++)
    {
        var->avg_log_lrt[i] = kLtrFeatureThr;
    }
}


