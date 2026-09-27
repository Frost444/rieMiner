// Host-only scheduling is compiled by MSVC, not by the CUDA device frontend.
#include <functional>
#include <future>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <stdexcept>
class R259Executor {
  std::mutex mutex_;
  std::condition_variable cv_;
  std::packaged_task<void()> job_;
  std::thread thread_;
  bool stop_ = false;
public:
  void call(std::function<void()> work) {
    std::packaged_task<void()> job(std::move(work));
    auto result = job.get_future();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!thread_.joinable()) thread_ = std::thread([this] {
        for (;;) {
          std::packaged_task<void()> next;
          {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return stop_ || job_.valid(); });
            if (stop_) return;
            next = std::move(job_);
          }
          next();
        }
      });
      if (job_.valid() || stop_) throw std::runtime_error("R259 producer ownership");
      job_ = std::move(job);
    }
    cv_.notify_one();
    result.get();
  }
  void stop() noexcept {
    { std::lock_guard<std::mutex> lock(mutex_); stop_ = true; }
    cv_.notify_one();
    if (thread_.joinable()) thread_.join();
  }
  ~R259Executor() { stop(); }
};
static R259Executor executor;
void r259_host_call(std::function<void()> work) { executor.call(std::move(work)); }
void r259_host_stop() { executor.stop(); }
