#ifndef STORGE_LAYOUT_H
#define STORGE_LAYOUT_H


template <typename KeyType, typename ValueType, bool UseBuckets, size_t CooperativeGroupSize, typename KeyRead>
class BucketizedLayout {
    
public:
    static constexpr size_t bucket_size = UseBuckets ? 
                (CooperativeGroupSize * KeyRead::key_count) : 1;

            
    void init(size_t key_num) {
        if(UseBuckets) {
            bucket_num_ = key_num / bucket_size;
            bucket_num_ =  warpcore::detail::get_valid_capacity(bucket_num_, 1);
        } else {
            bucket_num_ =  warpcore::detail::get_valid_capacity(key_num, CooperativeGroupSize);
        }
        cudaMalloc(&storage_, sizeof(StorageType) * bucket_num_);CUERR
        cudaMemset(storage_, 0, sizeof(StorageType) * bucket_num_);CUERR
    }

    void destroy() {
        cudaFree(storage_);CUERR
    }

    __device__ KeyType* GetCurKeyPos(size_t bucket_id, 
                      const cg::thread_block_tile<CooperativeGroupSize>& group) const noexcept {
        if(UseBuckets) {
            return &storage_[bucket_id].keys[group.thread_rank() * KeyRead::key_count];
        } else {
            bucket_id = (bucket_id + group.thread_rank()) % bucket_num_;
            return &storage_[bucket_id].keys[0];
        }
    }

    __device__ ValueType* GetCurValuePos(size_t bucket_id, 
                      const cg::thread_block_tile<CooperativeGroupSize>& group) const noexcept {
        if(UseBuckets) {
            return &storage_[bucket_id].values[group.thread_rank() * KeyRead::key_count];
        } else {
            bucket_id = (bucket_id + group.thread_rank()) % bucket_num_;
            return &storage_[bucket_id].values[0];
        }
    }

    size_t GetCapacity() const {
        return bucket_num_ * bucket_size;
    }

    __device__ size_t GetBucketNum() const {
        return bucket_num_;
    }


private:
    
    struct Bucket {
        KeyType keys[bucket_size];
        ValueType values[bucket_size];
    };
    
    using StorageType =  Bucket;

    StorageType * storage_;
    size_t bucket_num_;
};

template <typename KeyType, typename ValueType, bool UseBuckets, size_t CooperativeGroupSize, typename KeyRead>
class ContiguousLayout {
    static constexpr size_t bucket_size = UseBuckets ? 
            (CooperativeGroupSize * KeyRead::key_count) : 1;
public:
    void init(size_t key_num) {
        if(UseBuckets) {
            bucket_num_ = key_num / bucket_size;
            bucket_num_ =  warpcore::detail::get_valid_capacity(bucket_num_, 1);
        } else {
            bucket_num_ =  warpcore::detail::get_valid_capacity(key_num, CooperativeGroupSize);
        }
        cudaMalloc(&keys, sizeof(KeyType) * bucket_num_ * bucket_size);CUERR
        cudaMalloc(&values, sizeof(ValueType) * bucket_num_ * bucket_size);CUERR
    }

    void destroy() {
        cudaFree(keys);CUERR
        cudaFree(values);CUERR
    }

    __device__ KeyType* GetCurKeyPos(size_t bucket_id, 
                           const cg::thread_block_tile<CooperativeGroupSize>& group) const noexcept {
        if(UseBuckets) {
            // This is guaranteed to be aligned and not go out of bounds
            return &keys[bucket_id * bucket_size + group.thread_rank() * KeyRead::key_count];
        } else {
            bucket_id = (bucket_id + group.thread_rank()) % bucket_num_;
            return &keys[bucket_id];
        }
    }

    __device__ ValueType* GetCurValuePos(size_t bucket_id, 
                             const cg::thread_block_tile<CooperativeGroupSize>& group) const noexcept {
        if(UseBuckets) {
            // This is guaranteed to be aligned and not go out of bounds
            return &values[bucket_id * bucket_size + group.thread_rank() * KeyRead::key_count];
        } else {
            bucket_id = (bucket_id + group.thread_rank()) % bucket_num_;
            return &values[bucket_id];
        }
    }

    size_t GetCapacity() const {
        return bucket_num_ * bucket_size;
    }

    __device__ size_t GetBucketNum() const {
        return bucket_num_;
    }


private:
    KeyType* keys;
    ValueType* values;
    size_t bucket_num_;
};


#endif