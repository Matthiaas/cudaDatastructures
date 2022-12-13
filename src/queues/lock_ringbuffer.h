#ifndef LOCK_RINBUFFER_QUEUE_H
#define LOCK_RINBUFFER_QUEUE_H

#include <functional>
#include <iostream>
#include <mutex>

namespace queues {

template <typename T, size_t SIZE>
class LockRingBuffer {
 public:
  typedef T data_type;
  static constexpr bool can_run_on_gpu = false;
  static constexpr bool can_run_on_cpu = true;

  static_assert(SIZE > 0, "Size must be greater than 0");
  static_assert((SIZE & (SIZE - 1)) == 0, "Size must be a power of 2");
  static_assert(sizeof(T) == 8, "Size of T must be equal to 8 bytes");

  LockRingBuffer() : head_(0), tail_(0) {}
  ~LockRingBuffer() {}

  bool push(T value, bool insert) {
    if (!insert) {
      return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (head_ == tail_ + SIZE) {
      return false;
    }
    buffer[head_ % SIZE].data = value;
    head_++;
    return true;
  }

  bool pop(T* res, bool remove) {
    if (!remove) {
      return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (tail_ == head_) {
      return false;
    }
    *res = buffer[tail_ % SIZE].data;
    tail_++;
    return true;
  }

 private:
  struct Node {
    T data;
  };

  Node buffer[SIZE];
  volatile uint32_t head_;
  volatile uint32_t tail_;

  std::mutex mutex_;
};

}  // namespace queues

#endif  // !BROKER_QUEUE_H
