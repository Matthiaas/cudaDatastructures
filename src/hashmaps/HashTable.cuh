#include "HashTableDeviceImpl.cuh"

template <
    typename KeyType, 
    typename ValueType,
    KeyType EmptyKey, 
    KeyType TombstoneKey,
    auto &HashFunction,
    template <typename,auto&,bool,size_t> typename ProbingPolicyTemplate,
    bool UseBuckets = true,
    size_t CooperativeGroupSize = 16,
    template <typename> typename VectrizedReadPolicyTemplate = DefaultVectorizedReadPolicy
>
class MyHashTable
{
public:
    using ProbingPolicy = ProbingPolicyTemplate<KeyType, HashFunction, UseBuckets, CooperativeGroupSize>;
    static_assert(UseBuckets == true || 
        std::is_same_v<VectrizedReadPolicyTemplate<KeyType>, StandardReadPolicy<KeyType>>, 
        "Vectorized read policy is only supported with buckets");

    using key_type = KeyType;
    using value_type = ValueType;

    MyHashTable(uint64_t key_num)  {
        impl_.init(key_num);
    }
    ~MyHashTable() {
        impl_.destroy();
    }

    void insert(
            const key_type * const keys,
            const value_type * const values,
            const size_t count) {
        constexpr size_t block_size = 1024;
        const size_t block_count = (count * CooperativeGroupSize + block_size - 1) / block_size;
        insert_kernel<<<block_count, block_size>>>(keys, values, count, impl_);
        CUERR
    }

    void retrieve(
            const key_type * const keys,
            const size_t count,
            value_type * const values_out) {
        constexpr size_t block_size = 1024;
        const size_t block_count = (count * CooperativeGroupSize  + block_size - 1) / block_size;
        retrieve_kernel<<<block_count, block_size>>>(keys, values_out, count, impl_);
        CUERR
    }

    void print() {
        print_kernel<<<1, 1>>>(impl_);
        CUERR
    }

    size_t capacity() const noexcept
    {
        return capacity_;
    }  

    void init() const noexcept {}

    double load_factor() const noexcept
    {
        return static_cast<double>(inserted_elements_) / capacity_;
    }

    
private:    
    MyHashTableDeviceImpl<
        KeyType, 
        ValueType, 
        EmptyKey, 
        TombstoneKey, 
        UseBuckets, 
        CooperativeGroupSize,
        ProbingPolicy,
        VectrizedReadPolicyTemplate> impl_;
    uint64_t capacity_;
    uint64_t inserted_elements_; 
};
