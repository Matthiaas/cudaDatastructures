#include "HashTableDeviceImpl.cuh"

template <typename KeyType, typename ValueType, 
          KeyType EmptyKey, KeyType TombstoneKey,
          auto &HashFunction,
          template <typename, auto &, bool, size_t>
          typename ProbingPolicyTemplate,
          template <typename, typename, bool, size_t, typename>
          typename StoragePolicyTemplate,
          bool UseBuckets = true, 
          size_t CooperativeGroupSize = 16,
          template <typename> typename VectrizedReadPolicyTemplate =
              DefaultVectorizedReadPolicy>
class MyHashTable {
 public:
  static constexpr bool can_adjust_blocksize = true;
  using KeyRead = VectrizedReadPolicyTemplate<KeyType>;

  using ProbingPolicy = ProbingPolicyTemplate<KeyType, HashFunction, UseBuckets,
                                              CooperativeGroupSize>;
  using StoragePolicy = StoragePolicyTemplate<KeyType, ValueType, UseBuckets,
                                              CooperativeGroupSize, KeyRead>;
  static_assert(UseBuckets == true ||
                    std::is_same_v<KeyRead, StandardReadPolicy<KeyType>>,
                "Vectorized read policy is only supported with buckets");

  using key_type = KeyType;
  using value_type = ValueType;

  MyHashTable(uint64_t key_num, uint64_t block_size = 512)
      : inserted_elements_(0), block_size_(block_size) {
    impl_.init(key_num);
  }
  ~MyHashTable() { impl_.destroy(); }

  static std::string GetName() {
    return std::string("MyHashTable") + std::string("|") +
        ProbingPolicy::GetName() + std::string("|usebuckets:") +
        std::to_string(UseBuckets) + std::string("|groupsize:") +
        std::to_string(CooperativeGroupSize) + std::string("|layout:") +
        StoragePolicy::GetName() + std::string("|VecRead:") +
        std::to_string(VectrizedReadPolicyTemplate<KeyType>::key_count);
  }

  void insert(const key_type *const keys, const value_type *const values,
              const size_t count) {
    inserted_elements_ += count;
    const size_t block_count =
        (count * CooperativeGroupSize + block_size_ - 1) / block_size_;
    insert_kernel<<<block_count, block_size_>>>(keys, values, count, impl_);
    CUERR
  }

  void retrieve(const key_type *const keys, const size_t count,
                value_type *const values_out) {
    const size_t block_count =
        (count * CooperativeGroupSize + block_size_ - 1) / block_size_;
    retrieve_kernel<<<block_count, block_size_>>>(keys, values_out, count,
                                                  impl_);
    CUERR
  }

  void print() {
    print_kernel<<<1, 1>>>(impl_);
    CUERR
  }

  size_t capacity() const noexcept { return impl_.GetCapacity(); }

  void init() const noexcept {}

  double load_factor() const noexcept {
    return static_cast<double>(inserted_elements_) / capacity();
  }

 private:
  MyHashTableDeviceImpl<KeyType, ValueType, EmptyKey, TombstoneKey, UseBuckets,
                        CooperativeGroupSize, ProbingPolicy, StoragePolicy,
                        KeyRead>
      impl_;
  uint64_t inserted_elements_;
  uint64_t block_size_;
};
