#pragma once

#include <assert.h>

#define RTC_CHECK(condition) assert(condition)

#define RTC_CHECK_EQ(a, b) assert((a) == (b))
#define RTC_CHECK_NE(a, b) assert((a) != (b))
#define RTC_CHECK_LE(a, b) assert((a) <= (b))
#define RTC_CHECK_LT(a, b) assert((a) < (b))
#define RTC_CHECK_GE(a, b) assert((a) >= (b))
#define RTC_CHECK_GT(a, b) assert((a) > (b))

#define RTC_DCHECK(condition) assert(condition)

#define RTC_DCHECK_EQ(a, b) assert((a) == (b))
#define RTC_DCHECK_NE(a, b) assert((a) != (b))
#define RTC_DCHECK_LE(a, b) assert((a) <= (b))
#define RTC_DCHECK_LT(a, b) assert((a) < (b))
#define RTC_DCHECK_GE(a, b) assert((a) >= (b))
#define RTC_DCHECK_GT(a, b) assert((a) > (b)) 

#define RTC_DCHECK_NOTREACHED() RTC_DCHECK(false)

#ifdef __cplusplus

#include <iostream>

namespace rtc {

// Performs the integer division a/b and returns the result. CHECKs that the
// remainder is zero.
template <typename T>
inline T CheckedDivExact(T a, T b) {
	if ((a % b) != 0)
		std::cout << a << " is not evenly divisible by " << b << std::endl;
	RTC_CHECK_EQ((a % b), 0);
	return a / b;
}

}  // namespace rtc

#endif //__cplusplus