#ifndef __PERF_H_
#define __PERF_H_

#include <color.h>
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
	printf("[Done] Elasped time: "GREEN"%.2f"COLOREND" secs\n", this->getTime() / 1000);
      }

    private:
      StopWatchInterface* timer;
  };
};

#endif
