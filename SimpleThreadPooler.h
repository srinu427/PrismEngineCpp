#pragma once
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <atomic>
#include <functional>

class SimpleThreadPooler {
public:
	SimpleThreadPooler(uint32_t max_threads);
	~SimpleThreadPooler();

	template<typename F, typename ... Fargs>
	void add_task(F&& f, Fargs&& ...args) {
		at_lock.lock();
		_rem_tasks[rtj] = std::function<void()>([f, args...]{ f(args...); });
		rtj = (rtj + 1) % MAX_TASKS;
		at_lock.unlock();
		
	};
	void do_work(size_t tid);
	void run();
	void stop();
	void wait_till_done();
private:
	uint32_t MAX_TASKS = 1000;
	uint32_t thread_limit;
	std::vector<std::thread*> _wthreads;
	std::vector<bool> _thread_running;
	std::vector<std::function<void()>> _rem_tasks;
	uint32_t rti = 0, rtj = 0;
	std::mutex at_lock;
	std::atomic_bool stop_work = false;
};
