#include "Kernels.cuh"
#include "ProbingPolicies.cuh"
#include "VectroizedReadPolicies.cuh"


template<class KeyType>
using DefaultVectorizedReadPolicy = StandardReadPolicy<KeyType>;


template <
    typename KeyType, 
    typename ValueType,
    KeyType EmptyKey, 
    KeyType TombstoneKey,
    bool UseBuckets,
    size_t CooperativeGroupSize,
    typename ProbingPolicy, 
    template <typename> typename VectrizedReadPolicyTemplate 
>
class MyHashTableDeviceImpl {
public:
    static constexpr size_t cooperative_group_size = CooperativeGroupSize;
    using key_type = KeyType;
    using value_type = ValueType;

    using KeyRead = VectrizedReadPolicyTemplate<KeyType>;
    using ValueRead = VectrizedReadPolicyTemplate<ValueType>;

    void init(uint64_t key_num) {
        bucket_num_ = key_num / StorageType::bucket_size;
        cudaMalloc(&storage_, sizeof(StorageType) * bucket_num_);CUERR
        cudaMemset(storage_, 0, sizeof(StorageType) * bucket_num_);CUERR
    }

    void destroy() {
        cudaFree(storage_);CUERR
    }

    __device__ __forceinline__ constexpr bool IsEmpty(const KeyType key) const noexcept {
        return key == EmptyKey;
    }


    __device__ __forceinline__ void 
    GetKeyAndValuePos(key_type** cur_key_pos, value_type** cur_value_pos, const size_t hash,
                      const cg::thread_block_tile<CooperativeGroupSize>& group) const noexcept {
        if constexpr (UseBuckets) {
            uint64_t bucket_id = hash % bucket_num_;
            *cur_key_pos = &storage_[bucket_id].keys[group.thread_rank() * KeyRead::key_count];
            *cur_value_pos = &storage_[bucket_id].values[group.thread_rank() * KeyRead::key_count];
        } else {
            uint64_t bucket_id = (hash + group.thread_rank()) % bucket_num_;
            *cur_key_pos = &storage_[bucket_id].keys[0];
            *cur_value_pos = &storage_[bucket_id].values[0];

            // printf("value: %d at position %lld hash: %lld\n", **cur_value_pos, bucket_id, hash);
        }
    }

    __device__ bool retrieve(const KeyType key, ValueType& value_out, 
                             const cg::thread_block_tile<CooperativeGroupSize>& group) const noexcept {
        ProbingPolicy probing_policy;
        for (size_t hash = probing_policy.begin(key); hash != probing_policy.end(); hash = probing_policy.next()) {
            key_type* cur_key_pos;
            value_type* cur_value_pos;

            GetKeyAndValuePos(&cur_key_pos, &cur_value_pos, hash, group);
            uint64_t bucket_id = (hash + group.thread_rank()) % bucket_num_;

            typename KeyRead::ArrayType cur_keys = KeyRead::read(cur_key_pos);
            bool hit = false;
            bool found_empty_slot = false;
            constexpr size_t key_count = KeyRead::key_count;
            for (size_t i = 0; i < key_count; ++i) {
                if (cur_keys.data[i] == key) {
                    value_out = cur_value_pos[i];
                    hit = true;
                    break;
                } else if (cur_keys.data[i] == EmptyKey) {
                    found_empty_slot = true;
                }
            }

            const auto hit_mask = group.ballot(hit || empty_slot_mask);
            if(hit_mask) {
                return hit;
            }

        }
        
        return false;
    }


    __device__ bool insert(
            const key_type& key, 
            const value_type& value,
            const cg::thread_block_tile<CooperativeGroupSize>& group) {
        
       ProbingPolicy probing_policy;
       for (size_t hash = probing_policy.begin(key); hash != probing_policy.end(); hash = probing_policy.next()) {
            key_type* cur_key_pos;
            value_type* cur_value_pos;
            GetKeyAndValuePos(&cur_key_pos, &cur_value_pos, hash, group);
            key_type cur_key = *cur_key_pos;

            typename KeyRead::ArrayType cur_keys = KeyRead::read(cur_key_pos);
            bool hit = false;
            constexpr size_t key_count = KeyRead::key_count;
            
            for (size_t i = 0; i < key_count; ++i) {
                if (cur_keys.data[i] == key) {
                    cur_value_pos[i] = value;
                    hit = true;
                    break;
                }
            }
            const auto hit_mask = group.ballot(hit);

            if(hit_mask) {
                return true;
            }

            auto empty_mask = group.ballot(IsEmpty(cur_key));
            
            bool success = false;
            bool duplicate = false;

            while (empty_mask) {
                const auto leader = ffs(empty_mask) - 1;
                if (group.thread_rank() == leader) {
                    for (size_t i = 0; i < key_count; ++i) {
                        const auto old = atomicCAS(&cur_key_pos[i], EmptyKey, key);

                        success = (old == cur_key);
                        duplicate = (old == key);

                        if (success || duplicate) {
                            cur_value_pos[i] = value;
                            // printf( "insert key: %d at %lld\n", key, i);
                            break;
                        }
                    }
                }

                if (group.any(duplicate)) {
                    return true;
                }
                if (group.any(success)) {
                    return true;
                }

                empty_mask ^= 1UL << leader;
            }
        }


        return false;

    }

    __device__ void print() {
        for (int i = 0; i < bucket_num_; i++) {
            if constexpr (UseBuckets) {
                for (int j = 0; j < StorageType::bucket_size; j++) {
                    printf("key: %d value: %d at position %d, %d\n", storage_[i].keys[j], storage_[i].values[j], i, j);
                }
            } else {
                printf("key: %d value: %d at position %d\n", storage_[i].keys[0], storage_[i].values[0], i);
            }
        }
    }

private:

    template<size_t BucketSize>
    struct Bucket {
        static constexpr size_t bucket_size = BucketSize;
        KeyType keys[BucketSize];
        ValueType values[BucketSize];
    };
    static constexpr size_t bucket_size = CooperativeGroupSize * KeyRead::key_count;
    using StorageType = std::conditional_t<UseBuckets, Bucket<bucket_size>, Bucket<1>>;
    StorageType * storage_;
    size_t bucket_num_;

};

