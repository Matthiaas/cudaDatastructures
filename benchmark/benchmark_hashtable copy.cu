

#include "warpcore/single_value_hash_table.cuh"
#include "hash_table/benchmark_common.cuh"
#include "hash_table/dycuckoo_wrapper.cuh"

#include <hashmaps/HashTable.cuh>
#include <type_traits>

__device__ size_t hash_function(std::uint32_t x) {
  x ^= x >> 16;
  x *= 0x85ebca6b;
  x ^= x >> 13;
  x *= 0xc2b2ae35;
  x ^= x >> 16;
  return x;
}

template <typename T, typename Index>
__global__ void printArray(T* array, Index size) {
  for (Index i = 0; i < size; i++) {
    printf("%d ", array[i]);
  }
  printf("\n");
}

template <bool CountCollisions,class HashTable>
HOSTQUALIFIER INLINEQUALIFIER void single_value_benchmark(
    const typename HashTable::key_type* keys_d, const uint64_t max_keys,
    std::vector<uint64_t> input_sizes, std::vector<float> load_factors,
    std::vector<uint64_t> block_sizes, bool print_headers = true,
    uint8_t iters = 1,
    std::chrono::milliseconds thermal_backoff =
        std::chrono::milliseconds(100)) {
  using key_t = typename HashTable::key_type;
  using value_t = typename HashTable::value_type;

  value_t* values_d = nullptr;
  cudaMalloc(&values_d, sizeof(value_t) * max_keys);
  CUERR
  cudaMemcpy(values_d, keys_d, sizeof(value_t) * max_keys,
             cudaMemcpyDeviceToDevice);

  const auto max_input_size =
      *std::max_element(input_sizes.begin(), input_sizes.end());
  const auto min_load_factor =
      *std::min_element(load_factors.begin(), load_factors.end());

  if (max_input_size > max_keys) {
    std::cerr << "Maximum input size exceeded." << std::endl;
    exit(1);
  }

  // if constexpr (!std::is_same_v<HashTable, DycuckooHashTableWrapper>)
  // {
  //     if(!sufficient_memory_oa<HashTable>(max_input_size / min_load_factor))
  //     {
  //         std::cerr << "Not enough GPU memory." << std::endl;
  //         exit(1);
  //     }
  // }
  auto benchmark_hashtable = [&](HashTable& hash_table, uint64_t size,
                                 float load, uint64_t block_size) {
    Output<key_t, value_t> output;
    output.sample_size = size;
    output.key_capacity = hash_table.capacity();

    
    cudaMemcpy(values_d, keys_d, sizeof(value_t) * max_keys,
               cudaMemcpyDeviceToDevice);

    output.insert_ms = benchmark_insert(hash_table, keys_d, values_d, size,
                                        iters, thermal_backoff);

    cudaMemset(values_d, 0, sizeof(value_t) * size);
    // hash_table.print();

    output.query_ms = benchmark_query(hash_table, keys_d, values_d, size, iters,
                                      thermal_backoff);

    cudaDeviceSynchronize();
    CUERR
    // printArray<<<1,1>>>(values_d, size);

    cudaDeviceSynchronize();
    CUERR

    output.key_load_factor = hash_table.load_factor();
    output.density = output.key_load_factor;
    // output.status = hash_table.pop_status();
    std::cout << HashTable::GetName() << " " << load << " " << block_size; 
    if constexpr (CountCollisions) {
      const auto [a, b] = hash_table.GetCollisionCount();
      std::cout << " " << a << " " << b;
    } else {
      std::cout << " 0 0";
    }
    output.print_csv();
  };
  for (auto block_size : block_sizes) {
    for (auto size : input_sizes) {
      for (auto load : load_factors) {
        for (size_t i = 0; i < iters; i++) {
          const std::uint64_t capacity = size / load;
          if (block_size == 0) {
            // This means take the 'default' block size
            HashTable hash_table(capacity);
            benchmark_hashtable(hash_table, size, load, block_size);
          } else {
            HashTable hash_table(capacity, block_size);
            benchmark_hashtable(hash_table, size, load, block_size);
          }
        }
      }
    }
  }

  cudaFree(values_d);
  CUERR
}

using KeyType = std::uint32_t;
using ValueType = std::uint32_t;

template <bool CountCollisions, typename... Args>
void single_value_benchmarks(const KeyType* keys_d, const uint64_t max_keys,
                             std::vector<uint64_t> input_sizes,
                             std::vector<float> load_factors,
                             std::vector<uint64_t> block_sizes,
                             bool print_headers = true, uint8_t iters = 1,
                             std::chrono::milliseconds thermal_backoff =
                                 std::chrono::milliseconds(100)) {
  (single_value_benchmark<CountCollisions, Args>(keys_d, max_keys, input_sizes, load_factors,
                                block_sizes, print_headers, iters,
                                thermal_backoff),
   ...);
}

using namespace warpcore;
using warpcore_hash_table_t = SingleValueHashTable<
        KeyType,
        ValueType,
        defaults::empty_key<KeyType>(),
        defaults::tombstone_key<KeyType>(),
        defaults::probing_scheme_t<KeyType, 8>,
        storage::key_value::AoSStore<KeyType, ValueType>>;


using dycuckoo_hash_table_t = DycuckooHashTableWrapper;

template <size_t CG_size, template <typename T> typename ReadPolicy, bool CountCollisions = false>
struct HashTablesWithCG{


using lin_prob_no_bucket_standard_read = MyHashTable<
    KeyType, ValueType,
    0,0,
    hash_function,
    LinearProbingPolicy,
    BucketizedLayout,
    false,CG_size, 
    ReadPolicy,
    CountCollisions>;

using exp_prob_no_bucket_standard_read = MyHashTable<
    KeyType, ValueType,
    0,0,
    hash_function,
    QuadraticProbingPolicy,
    BucketizedLayout,
    false,CG_size, 
    StandardReadPolicy,
    CountCollisions>;

using double_prob_no_bucket_standard_read = MyHashTable<
    KeyType, ValueType,
    0,0,
    hash_function,
    DoubleHashinglProbingPolicy,
    BucketizedLayout,
    false,CG_size, 
    StandardReadPolicy,
    CountCollisions>;

using lin_prob_bucket_standard_read = MyHashTable<
    KeyType, ValueType,
    0,0,
    hash_function,
    LinearProbingPolicy,
    BucketizedLayout,
    true,CG_size, 
    StandardReadPolicy,
    CountCollisions>;

using exp_prob_bucket_standard_read = MyHashTable<
    KeyType, ValueType,
    0,0,
    hash_function,
    QuadraticProbingPolicy,
    BucketizedLayout,
    true,CG_size, 
    StandardReadPolicy,
    CountCollisions>;

using double_prob_bucket_standard_read = MyHashTable<
    KeyType, ValueType,
    0,0,
    hash_function,
    DoubleHashinglProbingPolicy,
    BucketizedLayout,
    true,CG_size, 
    StandardReadPolicy,
    CountCollisions>;

};

template <template <typename T> typename ReadPolicy, bool count_collisions>
void run_all(const KeyType* keys_d, const uint64_t max_keys,
             std::vector<uint64_t> input_sizes, std::vector<float> load_factors,
             std::vector<uint64_t> block_sizes, bool print_headers = true,
             uint8_t iters = 1,
             std::chrono::milliseconds thermal_backoff =
                 std::chrono::milliseconds(100)) {
single_value_benchmarks<
    count_collisions,
    typename HashTablesWithCG<
        1, ReadPolicy, count_collisions>::lin_prob_no_bucket_standard_read,
    typename HashTablesWithCG<
        2, ReadPolicy, count_collisions>::lin_prob_no_bucket_standard_read,
    typename HashTablesWithCG<
        4, ReadPolicy, count_collisions>::lin_prob_no_bucket_standard_read,
    typename HashTablesWithCG<
        8, ReadPolicy, count_collisions>::lin_prob_no_bucket_standard_read,
    typename HashTablesWithCG<
        16, ReadPolicy, count_collisions>::lin_prob_no_bucket_standard_read,
    typename HashTablesWithCG<
        32, ReadPolicy, count_collisions>::lin_prob_no_bucket_standard_read,

    typename HashTablesWithCG<2, ReadPolicy,
                              count_collisions>::lin_prob_bucket_standard_read,
    typename HashTablesWithCG<2, ReadPolicy,
                              count_collisions>::lin_prob_bucket_standard_read,
    typename HashTablesWithCG<4, ReadPolicy,
                              count_collisions>::lin_prob_bucket_standard_read,
    typename HashTablesWithCG<8, ReadPolicy,
                              count_collisions>::lin_prob_bucket_standard_read,
    typename HashTablesWithCG<16, ReadPolicy,
                              count_collisions>::lin_prob_bucket_standard_read,
    typename HashTablesWithCG<32, ReadPolicy,
                              count_collisions>::lin_prob_bucket_standard_read,

    typename HashTablesWithCG<
        1, ReadPolicy, count_collisions>::double_prob_no_bucket_standard_read,
    typename HashTablesWithCG<
        2, ReadPolicy, count_collisions>::double_prob_no_bucket_standard_read,
    typename HashTablesWithCG<
        4, ReadPolicy, count_collisions>::double_prob_no_bucket_standard_read,
    typename HashTablesWithCG<
        8, ReadPolicy, count_collisions>::double_prob_no_bucket_standard_read,
    typename HashTablesWithCG<
        16, ReadPolicy, count_collisions>::double_prob_no_bucket_standard_read,
    typename HashTablesWithCG<
        32, ReadPolicy, count_collisions>::double_prob_no_bucket_standard_read,

    typename HashTablesWithCG<
        1, ReadPolicy, count_collisions>::double_prob_bucket_standard_read,
    typename HashTablesWithCG<
        2, ReadPolicy, count_collisions>::double_prob_bucket_standard_read,
    typename HashTablesWithCG<
        4, ReadPolicy, count_collisions>::double_prob_bucket_standard_read,
    typename HashTablesWithCG<
        8, ReadPolicy, count_collisions>::double_prob_bucket_standard_read,
    typename HashTablesWithCG<
        16, ReadPolicy, count_collisions>::double_prob_bucket_standard_read,
    typename HashTablesWithCG<
        32, ReadPolicy, count_collisions>::double_prob_bucket_standard_read>(
    keys_d, max_keys, input_sizes, load_factors, block_sizes);
}

template <template <typename T> typename ReadPolicy, bool count_collisions>
void run_all_only_buckets(const KeyType* keys_d, const uint64_t max_keys,
             std::vector<uint64_t> input_sizes, std::vector<float> load_factors,
             std::vector<uint64_t> block_sizes, bool print_headers = true,
             uint8_t iters = 1,
             std::chrono::milliseconds thermal_backoff =
                 std::chrono::milliseconds(100)) {
single_value_benchmarks<
    count_collisions,
    typename HashTablesWithCG<2, ReadPolicy,
                              count_collisions>::lin_prob_bucket_standard_read,
    typename HashTablesWithCG<2, ReadPolicy,
                              count_collisions>::lin_prob_bucket_standard_read,
    typename HashTablesWithCG<4, ReadPolicy,
                              count_collisions>::lin_prob_bucket_standard_read,
    typename HashTablesWithCG<8, ReadPolicy,
                              count_collisions>::lin_prob_bucket_standard_read,
    typename HashTablesWithCG<16, ReadPolicy,
                              count_collisions>::lin_prob_bucket_standard_read,
    typename HashTablesWithCG<32, ReadPolicy,
                              count_collisions>::lin_prob_bucket_standard_read,

    typename HashTablesWithCG<
        1, ReadPolicy, count_collisions>::double_prob_bucket_standard_read,
    typename HashTablesWithCG<
        2, ReadPolicy, count_collisions>::double_prob_bucket_standard_read,
    typename HashTablesWithCG<
        4, ReadPolicy, count_collisions>::double_prob_bucket_standard_read,
    typename HashTablesWithCG<
        8, ReadPolicy, count_collisions>::double_prob_bucket_standard_read,
    typename HashTablesWithCG<
        16, ReadPolicy, count_collisions>::double_prob_bucket_standard_read,
    typename HashTablesWithCG<
        32, ReadPolicy, count_collisions>::double_prob_bucket_standard_read>(
    keys_d, max_keys, input_sizes, load_factors, block_sizes);
}

int main(int argc, char* argv[]) {
  const uint64_t max_keys = 1ul << 26;

  uint64_t dev_id = 0;
  if (argc > 2) dev_id = std::atoi(argv[2]);
  cudaSetDevice(dev_id);
  CUERR

  KeyType* keys_d = nullptr;
  if (argc > 1)
    keys_d = load_keys<KeyType>(argv[1], max_keys);
  else
    keys_d = generate_keys<KeyType>(max_keys, 1);

  auto load_factors =
      std::vector<float> {0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99};
  auto block_sizes =
      std::vector<uint64_t>{512};  // 32, 64, 128, 256, 512, 1024};
  auto max_keys_arr = std::vector<uint64_t>{max_keys};

  // We can not run the warpcore_hash_table_t with different block sizes.
  // So we run it with the default block size.
  single_value_benchmarks<false,
      dycuckoo_hash_table_t,
      warpcore_hash_table_t>(
          keys_d, max_keys, max_keys_arr, load_factors, {0});

  constexpr bool count_collisions = true;

  run_all<StandardReadPolicy, count_collisions>(
      keys_d, max_keys, max_keys_arr, load_factors, block_sizes);
  run_all_only_buckets<Vectroized2ReadPolicy, count_collisions>(
      keys_d, max_keys, max_keys_arr, load_factors, block_sizes);
  run_all_only_buckets<Vectroized4ReadPolicy, count_collisions>(
      keys_d, max_keys, max_keys_arr, load_factors, block_sizes);

  cudaFree(keys_d);
  CUERR
}
