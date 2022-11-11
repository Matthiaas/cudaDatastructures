#ifndef EPOCH_BASED_H
#define EPOCH_BASED_H

#include <cinttypes>

// Inspire by https://github.com/mpoeter/emr/blob/390ee0c3b92b8ad0adb897177202e1dd2c53a1b7/include/emr/epoch_based_impl.hpp

template <typename T>
struct LimboListNode {
    T* value;
    LimboListNode<T>* next;
};

template <typename T>
struct LocalThreadData {
    __device__ LocalThreadData() : 
            local_epoch_entries_(0),
             local_epoch_(0), 
             is_in_crtitcal_section_(false) {
        for (int i = 0; i < 2; ++i) {
            limbo_list_[i] = nullptr;
        }
    }
    LimboListNode<T>* limbo_list_[3];
    uint32_t local_epoch_entries_;
    volatile uint32_t local_epoch_;
    volatile bool is_in_crtitcal_section_;
};

namespace default_settings {
    __device__ uint32_t getGlobalThreadId() {
        return blockIdx.x * blockDim.x + threadIdx.x;
    }

    template <typename T>
    __device__ void deleteObject(T* ptr) {
        delete ptr;
    }
}

typedef uint32_t (*GetGlobalThreadId_t)(); 

template <
    typename T, 
    std::size_t UPDATE_THRESHOLD,
    void (*DeleteFun)(T*) = default_settings::deleteObject, 
    GetGlobalThreadId_t GetGlobalThreadId = default_settings::getGlobalThreadId>
class EpochBasedReclamation {
public:
    __device__ EpochBasedReclamation(uint32_t global_thread_count) 
            : global_epoch_(0), global_thread_count_(global_thread_count) {
        local_thread_data_ = new LocalThreadData<T>[global_thread_count];
    }
    __device__ ~EpochBasedReclamation() {}

    __device__ void Setup() {
        uint32_t global_thread_id = GetGlobalThreadId();
        local_thread_data_[global_thread_id] = LocalThreadData<T>();
    }
    
    __device__ void Retire(T* element) {
        uint32_t global_thread_id = GetGlobalThreadId();
        LocalThreadData<T>& local_thread_data = local_thread_data_[global_thread_id];
        LimboListNode<T>* limbo_node = new LimboListNode<T>();
        limbo_node->value = element;
        limbo_node->next = 
            local_thread_data.limbo_list_[local_thread_data.local_epoch_];
        local_thread_data.limbo_list_[local_thread_data.local_epoch_] = limbo_node;

    }

    __device__ void EnterCriticalSection() {
        uint32_t global_thread_id = GetGlobalThreadId();
        LocalThreadData<T>& local_thread_data = local_thread_data_[global_thread_id];
        local_thread_data.is_in_crtitcal_section_ = true;
        __threadfence();
        uint32_t cur_global_epoch = global_epoch_;
        // printf("cur_global_epoch: %d, local_epoch: %d\n", cur_global_epoch, local_thread_data.local_epoch_);
        if (local_thread_data.local_epoch_ != cur_global_epoch) {
            // Some other thread has already updated the global epoch
            local_thread_data.local_epoch_ = global_epoch_;
            // printf("cur_global_epoch: %d, local_epoch: %d\n", cur_global_epoch, local_thread_data.local_epoch_);

        } else if (local_thread_data.local_epoch_entries_++ >= UPDATE_THRESHOLD) {
            // printf("Thread %d: local_epoch_entries_ = %d\n", global_thread_id, local_thread_data.local_epoch_entries_);
            if (!TryUpdateGlobalEpoch(cur_global_epoch)) {
                return;
            }
            
            local_thread_data.local_epoch_ = global_epoch_;
        } else {
            // No need to update the global epoch
            return;
        }
        local_thread_data.local_epoch_entries_ = 0;
        // Before we can use the new epoch, we need to free our limbo list.
        LimboListNode<T>* limbo_list = 
            local_thread_data.limbo_list_[local_thread_data.local_epoch_];
        local_thread_data.limbo_list_[local_thread_data.local_epoch_] = nullptr;
        while (limbo_list != nullptr) {
            LimboListNode<T>* next = limbo_list->next;
            DeleteFun(limbo_list->value);
            delete limbo_list;
            limbo_list = next;
        }
    }

    __device__ bool TryUpdateGlobalEpoch(uint32_t cur_global_epoch) {
        uint32_t new_epoch = (cur_global_epoch + 1) % MAX_EPOCH;
        uint32_t old_epoch = (cur_global_epoch + MAX_EPOCH - 1) % MAX_EPOCH;
        bool can_update = true;
        for (uint32_t i = 0; i < global_thread_count_ && can_update; ++i) {
            const auto& local_thread_data = local_thread_data_[i];
            can_update &= !local_thread_data.is_in_crtitcal_section_ || 
                            local_thread_data.local_epoch_ != old_epoch;
            // printf("Thread %d: is_in_crtitcal_section_ = %d, local_epoch_ = %d, old_epoch = %d\n", i, local_thread_data.is_in_crtitcal_section_, local_thread_data.local_epoch_, old_epoch);
        }
        // printf("can_update = %d\n", can_update);
        if (!can_update) {
            return false;
        }
        if (global_epoch_ == cur_global_epoch) {
            __threadfence();
            global_epoch_ = new_epoch;
            // printf("Update global epoch from %d to %d\n", cur_global_epoch, new_epoch);
        }
        return true;
    }

    __device__ void LeaveCriticalSection() {
        __threadfence();
        uint32_t global_thread_id = GetGlobalThreadId();
        local_thread_data_[global_thread_id].is_in_crtitcal_section_ = false;
    }

    __device__ size_t GetGlobalEpoch() const {
        return global_epoch_;
    }

    __device__ size_t GetLocalEpoch() const {
        uint32_t global_thread_id = GetGlobalThreadId();
        return local_thread_data_[global_thread_id].local_epoch_;
    }

private:
    LocalThreadData<T>* local_thread_data_;
    size_t global_thread_count_;
    volatile uint32_t global_epoch_;
    static constexpr uint32_t MAX_EPOCH = 3;

};



#endif /* EPOCH_BASED_H */
