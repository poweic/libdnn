#ifndef __PERF_H_
#define __PERF_H_

#include <helper_timer.h>

namespace perf {
  class Timer {
    public:
      Timer(): timer(NULL) { sdkCreateTimer(&timer); }
      ~Timer() { sdkDeleteTimer(&timer); }

      void start()	{ timer->start(); }
      void stop()	{ timer->stop();  }
      void reset()	{ timer->reset(); }

      float getTime() { return timer->getTime(); }
      void elapsed() {
	this->stop();
	printf("\33[33m[Info]\33[0m Done. Elasped time: \33[32m%.2f\33[0m secs\n", this->getTime() / 1000);
      }

    private:
      StopWatchInterface* timer;
  };
};

#endif
