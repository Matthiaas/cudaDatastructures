#include "benchmark_common.cuh"

#include "../benchmark_utils.cuh"



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

  auto benchmark_hashtable = [&](HashTable& hash_table, uint64_t size,
                                 float load, uint64_t block_size) {
    Output<key_t, value_t> output;
    output.sample_size = size;
    output.key_capacity = hash_table.capacity();

    output.insert_ms = benchmark_insert(hash_table, keys_d, values_d, size,
                                        iters, thermal_backoff);

    cudaMemset(values_d, 0, sizeof(value_t) * size);

    output.query_ms = benchmark_query(hash_table, keys_d, values_d, size, iters,
                                      thermal_backoff);

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


template <bool CountCollisions, typename... Args, typename KeyType>
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
