#include "SimpleThreadPooler.h"
#include "iostream"

SimpleThreadPooler::SimpleThreadPooler(uint32_t max_threads)
{
	thread_limit = max_threads;
	_wthreads.resize(thread_limit);
	_thread_running.resize(thread_limit);
	_rem_tasks.resize(MAX_TASKS);
	_task_being_read = new std::atomic<bool>[MAX_TASKS];
}

SimpleThreadPooler::~SimpleThreadPooler()
{
	stop();
	delete _task_being_read, MAX_TASKS;
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
	uint32_t myrti = 0;
	while (!stop_work) {
		if (myrti != rtj) {
			bool no_work = false;
			if (!_task_being_read[myrti]) {
				_task_being_read[myrti] = true;
				std::function<void()> t = _rem_tasks[myrti];
				_thread_running[tid] = true;
				t();
				_thread_running[tid] = false;
			}
			else{
				myrti = (myrti + 1) % MAX_TASKS;
			}
		}
		else {
			_thread_running[tid] = false;
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
	wait_lock.lock();
	while (true) {
		bool is_done = true;
		for (size_t i = 0; i < thread_limit; i++) {
			if (_thread_running[i]) {
				is_done = false;
				break;
			}
		}
		if (is_done) {
			wait_lock.unlock();
			return;
		}
	}
	wait_lock.unlock();
}
