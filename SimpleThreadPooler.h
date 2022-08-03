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
		wait_lock.lock();
		_rem_tasks[rtj] = std::function<void()>([f, args...]{ f(args...); });
		_task_being_read[rtj] = false;
		rtj = (rtj + 1) % MAX_TASKS;
		wait_lock.unlock();
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
	std::atomic<bool>* _task_being_read;
	uint32_t rtj = 0;
	std::mutex at_lock;
	std::mutex wait_lock;
	std::atomic<bool> stop_work = false;
};
