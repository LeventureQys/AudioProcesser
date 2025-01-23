#pragma once

#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

// Number of bars on the indicator.
// Note that the number of elements is specified because we are indexing it
// in the range of 0-32
const int8_t permutation[33] =
{ 0,1,2,3,4,4,5,5,5,5,6,6,6,6,6,7,7,7,7,8,8,8,9,9,9,9,9,9,9,9,9,9,9 };

// Maximum absolute value of word16 vector. C version for generic platforms.
static int16_t _MaxAbsValueW16C(const int16_t* vector, size_t length) {
	size_t i = 0;
	int absolute = 0, maximum = 0;

	assert(length);

	for (i = 0; i < length; i++) {
		absolute = abs((int)vector[i]);

		if (absolute > maximum) {
			maximum = absolute;
		}
	}

	// Guard the case for abs(-32768).
	if (maximum > INT16_MAX) {
		maximum = INT16_MAX;
	}

	return (int16_t)maximum;
}

static int8_t Voice_ComputeLevel(const int16_t* vector, size_t length)
{
	static int16_t _absMax = 0;
	int16_t absValue(0);

	// Check speech level (works for 2 channels as well)
	absValue = _MaxAbsValueW16C(
		vector,
		length);

	if (absValue > _absMax)
		_absMax = absValue;

	// Highest value for a int16_t is 0x7fff = 32767
	// Divide with 1000 to get in the range of 0-32 which is the range of
	// the permutation vector
	int32_t position = _absMax / 1000;

	// Make it less likely that the bar stays at position 0. I.e. only if
	// its in the range 0-250 (instead of 0-1000)
	if ((position == 0) && (_absMax > 250))
	{
		position = 1;
	}

	int8_t _currentLevel =  permutation[position];

	// Decay the absolute maximum (divide by 4)
	_absMax >>= 2;

	return _currentLevel;
}