#include "prior_signal_model.h"
//#include "user_task.h"

void PriorSignalModel_init(PriorSignalModel* var, float lrt_initial_value) {
    var->flatness_threshold = .5f;
    var->template_diff_threshold = .5f;
    var->lrt_weighting = 1.f;
    var->flatness_weighting = 0.f;
    var->difference_weighting = 0.f;
    
    var->lrt = lrt_initial_value;

}
