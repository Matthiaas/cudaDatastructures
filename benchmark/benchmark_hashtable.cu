

#include "warpcore/single_value_hash_table.cuh"
#include "hash_table/dycuckoo_wrapper.cuh"
#include "hash_table/gpu_hash_map_benchmark.cuh"

#include "oneapi/tbb/concurrent_hash_map.h"

#include "type_traits.h"

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

using KeyType = std::uint32_t;
using ValueType = std::uint32_t;

using namespace warpcore;
using warpcore_hash_table_t = SingleValueHashTable<
        KeyType,
        ValueType,
        defaults::empty_key<KeyType>(),
        defaults::tombstone_key<KeyType>(),
        defaults::probing_scheme_t<KeyType, 8>,
        storage::key_value::AoSStore<KeyType, ValueType>>;


using dycuckoo_hash_table_t = DycuckooHashTableWrapper;

template <typename T, T val> 
struct ValueContainer{ 
 static constexpr T const value = val;
};

template <template <typename T> typename ReadPolicy> 
struct ReadPolicyContainer {

};

template <template <typename, auto &, bool, size_t>
          typename ProbingPolicyTemplate> 
struct ProbingPolicyContainer {

};

template <template <typename, typename, bool, size_t, typename>
          typename StoragePolicyTemplate>
struct StoragePolicyContainer {

};

using CGSizes = mb::set<
  ValueContainer<size_t, 1>,
  ValueContainer<size_t, 2>,
  ValueContainer<size_t, 4>,
  ValueContainer<size_t, 8>,
  ValueContainer<size_t, 16>,
  ValueContainer<size_t, 32>>;

using BucketConfigs = mb::set<
  ValueContainer<bool, true>,
  ValueContainer<bool, false>>;

using OnlyBucketsConfig = mb::set<
  ValueContainer<bool, true>>;

using ReadPolicies = mb::set<
  ReadPolicyContainer<StandardReadPolicy>
>;

using VecReadPolicies = mb::set<
  ReadPolicyContainer<Vectroized2ReadPolicy>,
  ReadPolicyContainer<Vectroized4ReadPolicy>
>;

using ProbingPolicies = mb::set<
  ProbingPolicyContainer<LinearProbingPolicy>,
  ProbingPolicyContainer<DoubleHashinglProbingPolicy>
>;

using VecReadStorageLayouts = mb::set<
  StoragePolicyContainer<GroupLayout>,
  StoragePolicyContainer<ContiguousLayout>,
>;

using StorageLayouts = mb::set<
  StoragePolicyContainer<GroupLayout>,
  StoragePolicyContainer<ContiguousLayout>,
  StoragePolicyContainer<ContiguousKeyValLayout>
>;

using ConfigCrossProduct1 =
    mb::cross_product<CGSizes, BucketConfigs, ReadPolicies, ProbingPolicies,
                      StorageLayouts>;
using ConfigCrossProduct2 =
    mb::cross_product<CGSizes, OnlyBucketsConfig, VecReadPolicies, ProbingPolicies,
                      VecReadStorageLayouts>;

template <typename A, typename B,typename C, typename D,typename E>
struct Runner;

template <size_t CooperativeGroupSize, bool UseBuckets,
          template <typename> typename VectrizedReadPolicyTemplate,
          template <typename, auto&, bool, size_t>
          typename ProbingPolicyTemplate,
          template <typename, typename, bool, size_t, typename>
          typename StoragePolicyTemplate>
struct Runner<ValueContainer<size_t, CooperativeGroupSize>,
              ValueContainer<bool, UseBuckets>,
              ReadPolicyContainer<VectrizedReadPolicyTemplate>,
              ProbingPolicyContainer<ProbingPolicyTemplate>,
              StoragePolicyContainer<StoragePolicyTemplate>> {
 void operator()(const KeyType* keys_d, const uint64_t max_keys,
                 std::vector<uint64_t> input_sizes,
                 std::vector<float> load_factors,
                 std::vector<uint64_t> block_sizes, bool print_headers = true,
                 uint8_t iters = 1,
                 std::chrono::milliseconds thermal_backoff =
                     std::chrono::milliseconds(100)) {
    constexpr bool count_collisions = false;
    using Map = MyHashTable<KeyType, ValueType, 0, 0, hash_function,
                            ProbingPolicyTemplate, StoragePolicyTemplate,
                            UseBuckets, CooperativeGroupSize,
                            VectrizedReadPolicyTemplate, count_collisions>;
    single_value_benchmark<count_collisions, Map>(
        keys_d, max_keys, input_sizes, load_factors, block_sizes, print_headers,
        iters, thermal_backoff);
 }
};

void run_gpu_benchmark(int argc, char* argv[], size_t max_keys) {
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

  mb::for_each<ConfigCrossProduct1, Runner>(keys_d, max_keys, max_keys_arr, load_factors, block_sizes);
  mb::for_each<ConfigCrossProduct2, Runner>(keys_d, max_keys, max_keys_arr, load_factors, block_sizes);

 
  cudaFree(keys_d);
  CUERR
}

void run_cpu_benchmark(int argc, char* argv[], size_t max_keys) {
  std::vector<KeyType> keys(max_keys, 0);
  std::vector<ValueType> values(max_keys, 0);
  
  for (int i = 0; i < max_keys; ++i) {
    keys[i] = i;
    values[i] = i;
  }

  auto thread_fun = [&](int tid, int num_threads, 
                        tbb::concurrent_hash_map<KeyType, ValueType>& tbb_map) {
    // Let each thread insert a contiguous range of keys.
    int start = tid * max_keys / num_threads;
    int end = (tid + 1) * max_keys / num_threads;
    for (int i = start; i < end; ++i) {
      tbb_map.insert({keys[i], values[i]});
    }
  };

  auto load_factors =
      std::vector<float> {0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99};

  for (auto load_factor : load_factors) {
    for (size_t thread_count = 1; thread_count <= 32; thread_count *= 2) {
      Output<KeyType, ValueType> output;
      output.sample_size = max_keys;
      output.key_capacity = max_keys / load_factor;
      output.key_load_factor = load_factor;
      output.density = output.key_load_factor;

      std::vector<std::thread> threads;
      tbb::concurrent_hash_map<KeyType, ValueType> tbb_map(max_keys / load_factor );
      
      auto start = std::chrono::high_resolution_clock::now();
      for (size_t i = 0; i < thread_count; ++i) {
        threads.emplace_back(thread_fun, i, thread_count, std::ref(tbb_map));
      }
      for (auto& t : threads) t.join();
      auto end = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
      output.insert_ms = duration / 1000.0;

      start = std::chrono::high_resolution_clock::now();
      // Query all keys.
      for (int i = 0; i < max_keys; ++i) {
        tbb::concurrent_hash_map<KeyType, ValueType>::accessor a;
        tbb_map.find(a, keys[i]);
        values[i] = a->second;
      }
      end = std::chrono::high_resolution_clock::now();
      duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
      output.query_ms = duration / 1000.0;

      std::cout << "CPUTbb" << " " << load_factor << " " << thread_count << " 0 0"; 
      output.print_csv();
    
      threads.clear();
    }
  }
}

int main(int argc, char* argv[]) {
  const uint64_t max_keys = 1ul << 26;
  
  // run_gpu_benchmark(argc, argv, max_keys);
  run_cpu_benchmark(argc, argv, max_keys);

  
}
