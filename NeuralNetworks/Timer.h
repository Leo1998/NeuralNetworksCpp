#pragma once

#include <chrono>

namespace nn {

	class Timer {
	public:
		Timer();
		void reset();
		float elapsedMillis() const;
		inline float elapsedSeconds() const { return elapsedMillis() / 1000.0f; }

	private:
		typedef std::chrono::high_resolution_clock clock;
		typedef std::chrono::duration<float, std::milli> milliseconds_type;

		std::chrono::time_point<clock> beg;
	};
}