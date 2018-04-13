#include "Timer.h"

namespace nn {

	Timer::Timer() : beg(clock::now())
	{
	}

	void Timer::reset()
	{
		beg = clock::now();
	}

	float Timer::elapsedMillis() const
	{
		return std::chrono::duration_cast<milliseconds_type>(clock::now() - beg).count();
	}
}