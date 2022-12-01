#include "Kernels.cuh"
#include "ProbingPolicies.cuh"
#include "StorageLayout.cuh"
#include "VectroizedReadPolicies.cuh"
#include "warpcore/base.cuh"

template <class KeyType>
using DefaultVectorizedReadPolicy = StandardReadPolicy<KeyType>;

template <
    typename KeyType, 
    typename ValueType,
    KeyType EmptyKey, 
    KeyType TombstoneKey,
    bool UseBuckets,
    size_t CooperativeGroupSize,
    typename ProbingPolicy, 
    typename StoreageLayout,
    typename KeyRead 
>
class MyHashTableDeviceImpl {
public:
 static constexpr size_t cooperative_group_size = CooperativeGroupSize;
 static constexpr size_t KeyReadSize = KeyRead::key_count;
 using key_type = KeyType;
 using value_type = ValueType;

 void init(uint64_t key_num) { storage_.init(key_num); }

 void destroy() { storage_.destroy(); }

 size_t GetCapacity() const { return storage_.GetCapacity(); }

 __device__ __forceinline__ constexpr bool IsEmpty(
     const KeyType key) const noexcept {
   return key == EmptyKey;
 }

 __device__ bool retrieve(
     const KeyType key, ValueType& value_out,
     const cg::thread_block_tile<CooperativeGroupSize>& group) const noexcept {
   ProbingPolicy probing_policy(storage_.GetBucketNum());
   for (size_t pos = probing_policy.begin(key); pos != probing_policy.end();
        pos = probing_policy.next()) {
     key_type* cur_key_pos = storage_.GetCurKeyPos(pos, group);

     // GetKeyAndValuePos(&cur_key_pos, &cur_value_pos, pos, group);
     typename KeyRead::ArrayType cur_keys = KeyRead::read(cur_key_pos);
     bool hit = false;
     bool found_empty_slot = false;
     for (size_t i = 0; i < KeyReadSize; ++i) {
       if (cur_keys.data[i] == key) {
         value_type* cur_value_pos = storage_.GetCurValuePos(pos, group);
        //  value_out = cur_value_pos[i];
         hit = true;
         break;
       } else if (cur_keys.data[i] == EmptyKey) {
         found_empty_slot = true;
       }
     }

     const auto hit_mask = group.ballot(hit || found_empty_slot);
     if (hit_mask) {
       return hit;
     }
   }

   return false;
 }

 __device__ bool insert(
     const key_type& key, const value_type& value,
     const cg::thread_block_tile<CooperativeGroupSize>& group) {
   ProbingPolicy probing_policy(storage_.GetBucketNum());
   for (size_t pos = probing_policy.begin(key); pos != probing_policy.end();
        pos = probing_policy.next()) {
     key_type* cur_key_pos = storage_.GetCurKeyPos(pos, group);
     typename KeyRead::ArrayType cur_keys = KeyRead::read(cur_key_pos);
     bool hit = false;
     bool found_empty_key = false;
     for (size_t i = 0; i < KeyReadSize; ++i) {
       if (cur_keys.data[i] == key) {
         value_type* cur_value_pos = storage_.GetCurValuePos(pos, group);
        //  cur_value_pos[i] = value;
         hit = true;
         break;
       } else if (IsEmpty(cur_keys.data[i])) {
         found_empty_key = true;
       }
     }
     const auto hit_mask = group.ballot(hit);

     if (hit_mask) {
       return true;
     }

     auto empty_mask = group.ballot(found_empty_key);

     bool success = false;
     bool duplicate = false;

     while (empty_mask) {
       const auto leader = ffs(empty_mask) - 1;
       if (group.thread_rank() == leader) {
         for (size_t i = 0; i < KeyReadSize; ++i) {
           const auto old = atomicCAS(&cur_key_pos[i], EmptyKey, key);

           success = (old == EmptyKey);
           duplicate = (old == key);

           if (success || duplicate) {
             value_type* cur_value_pos = storage_.GetCurValuePos(pos, group);
            //  cur_value_pos[i] = value;
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

    // __device__ void print() {
    //     for (int i = 0; i < bucket_num_; i++) {
    //         if constexpr (UseBuckets) {
    //             for (int j = 0; j < StorageType::bucket_size; j++) {
    //                 printf("key: %d value: %d at position %d, %d\n", storage_[i].keys[j], storage_[i].values[j], i, j);
    //             }
    //         } else {
    //             printf("key: %d value: %d at position %d\n", storage_[i].keys[0], storage_[i].values[0], i);
    //         }
    //     }
    // }



private:
 StoreageLayout storage_;
};

