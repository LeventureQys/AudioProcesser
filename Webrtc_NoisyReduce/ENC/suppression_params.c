#include "suppression_params.h"
//#include "user_task.h"

void SuppressionParams_init(SuppressionParams* var, int suppression_level) {
    switch (suppression_level) {
        // 6 dB attenuation.
        case 0:
        var->over_subtraction_factor = 1.f;
        var->minimum_attenuating_gain = 0.5f;
        var->use_attenuation_adjustment = false;
        break;
        // 12 dB attenuation.
        case 1:
        var->over_subtraction_factor = 1.f;
        var->minimum_attenuating_gain = 0.25f;
        var->use_attenuation_adjustment = true;
        break;
        // 18 dB attenuation.
        case 2:
        var->over_subtraction_factor = 1.1f;
        var->minimum_attenuating_gain = 0.125f;
        var->use_attenuation_adjustment = true;
        break;
        // 21 dB attenuation.
        case 3:
        var->over_subtraction_factor = 1.25f;
        var->minimum_attenuating_gain = 0.09f;
        var->use_attenuation_adjustment = true;
        break;
    }

}
