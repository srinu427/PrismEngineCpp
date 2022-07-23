#include "SimpleThreadPooler.h"
#include "iostream"

SimpleThreadPooler::SimpleThreadPooler(uint32_t max_threads)
{
	thread_limit = max_threads;
	_wthreads.resize(thread_limit);
	_thread_running.resize(thread_limit);
	_rem_tasks.resize(MAX_TASKS);
}

SimpleThreadPooler::~SimpleThreadPooler()
{
	stop();
}

//template<typename F, typename ...Fargs>
//inline void SimpleThreadPooler::add_task(F&& f, Fargs && ...args)
//{
//	at_lock.lock();
//	_rem_tasks.push(std::function<void()>([f, args...]{ f(args...); }));
//	at_lock.unlock();
//}

void SimpleThreadPooler::do_work(size_t tid)
{
	while (!stop_work) {
		at_lock.lock();
		if (rti != rtj) {
			std::function<void()> t = _rem_tasks[rti];
			rti = (rti + 1) % MAX_TASKS;
			_thread_running[tid] = true;
			at_lock.unlock();
			t();
		}
		else {
			_thread_running[tid] = false;
			at_lock.unlock();
		}
	}
}

void SimpleThreadPooler::run()
{
	stop_work = false;
	for (uint32_t i = 0; i < thread_limit; i++) {
		_wthreads[i] = new std::thread(&SimpleThreadPooler::do_work, this, i);
	}
}

void SimpleThreadPooler::stop()
{
	wait_till_done();
	stop_work = true;
	for (uint32_t i = 0; i < thread_limit; i++) {
		if (_wthreads[i] != NULL && _wthreads[i]->joinable()) {
			_wthreads[i]->join();
			delete _wthreads[i];
		}
	}

}

void SimpleThreadPooler::wait_till_done()
{
	while (true) {
		at_lock.lock();
		bool is_empty = rti == rtj;
		if (is_empty) {
			for (size_t i = 0; i < thread_limit; i++) {
				if (_thread_running[i]) is_empty = false;
			}
		}
		at_lock.unlock();
		if (is_empty) {
			return;
		}
	}
}
