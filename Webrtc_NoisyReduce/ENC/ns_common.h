#ifndef MODULES_AUDIO_PROCESSING_NS_SUPPRESSION_PARAMS_H_
#define MODULES_AUDIO_PROCESSING_NS_SUPPRESSION_PARAMS_H_


#include <math.h>
//#include "user_task.h"

typedef   signed           int int32_t;
typedef unsigned           int uint32_t;
typedef unsigned          char uint8_t;

#define M_PI 3.14159265358979323846264338327950288
#define epsilon 0.000001
#define kFftSize 256
#define kFftSizeBy2Plus1  (kFftSize / 2 + 1)
#define kNsFrameSize 160
#define kOverlapSize (kFftSize - kNsFrameSize)

#define kShortStartupPhaseBlocks 50
#define kLongStartupPhaseBlocks 200
#define kFeatureUpdateWindowSize 500

#define kLtrFeatureThr 0.5f
#define kBinSizeLrt 0.1f
#define kBinSizeSpecFlat 0.05f
#define kBinSizeSpecDiff 0.1f

#define max_local(a,b) ((a>b)? a:b)
#define min_local(a,b) ((a<b)? a:b)
#define abs_local(a)  ((a>0)? a:(-a))

#define RTC_DCHECK_EQ(a, b) assert((a) == (b))
#define RTC_DCHECK_LE(a, b) assert((a) <= (b))

#endif  // MODULES_AUDIO_PROCESSING_NS_QUANTILE_NOISE_ESTIMATOR_H_
