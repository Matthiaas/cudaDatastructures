

#include "warpcore/single_value_hash_table.cuh"
#include "hash_table/benchmark_common.cuh"
#include "hash_table/dycuckoo_wrapper.cuh"

#include <hashmaps/HashTable.cuh>

__device__  size_t hash_function(std::uint32_t x) {
    x ^= x >> 16;
    x *= 0x85ebca6b;
    x ^= x >> 13;
    x *= 0xc2b2ae35;
    x ^= x >> 16;
    return x;
    
}

template <typename T>
__global__ void printArray(T *array, int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", array[i]);
    }
    printf("\n");
}

template<class HashTable>
HOSTQUALIFIER INLINEQUALIFIER
void single_value_benchmark(
    const typename HashTable::key_type * keys_d,
    const uint64_t max_keys,
    std::vector<uint64_t> input_sizes,
    std::vector<float> load_factors,
    bool print_headers = true,
    uint8_t iters = 1,
    std::chrono::milliseconds thermal_backoff = std::chrono::milliseconds(100))
{
    using key_t = typename HashTable::key_type;
    using value_t = typename HashTable::value_type;

    value_t* values_d = nullptr;
    cudaMalloc(&values_d, sizeof(value_t)*max_keys); CUERR
   

    const auto max_input_size =
        *std::max_element(input_sizes.begin(), input_sizes.end());
    const auto min_load_factor =
        *std::min_element(load_factors.begin(), load_factors.end());

    if(max_input_size > max_keys)
    {
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

    for(auto size : input_sizes)
    {
        for(auto load : load_factors)
        {
            const std::uint64_t capacity = size / load;

            HashTable hash_table(capacity);

            Output<key_t,value_t> output;
            output.sample_size = size;
            output.key_capacity = hash_table.capacity();

            output.insert_ms = benchmark_insert(
                hash_table, keys_d, values_d, size,
                iters, thermal_backoff);

                std::cout<< "inserted" << std::endl;

            output.query_ms = benchmark_query(
                hash_table, keys_d, values_d, size,
                iters, thermal_backoff);

            cudaDeviceSynchronize(); CUERR         

            output.key_load_factor = hash_table.load_factor();
            output.density = output.key_load_factor;
            //output.status = hash_table.pop_status();

            if(print_headers)
                output.print_with_headers();
            else
                output.print_without_headers();
        }
    }

    cudaFree(values_d); CUERR
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

template <size_t CG_size>
struct HashTablesWithCG{
using lin_prob_no_bucket_standard_read = MyHashTable<
    KeyType, ValueType,
    0,0,
    hash_function,
    LinearProbingPolicy,
    false,CG_size, 
    StandardReadPolicy>;

using exp_prob_no_bucket_standard_read = MyHashTable<
    KeyType, ValueType,
    0,0,
    hash_function,
    QuadraticProbingPolicy,
    false,CG_size, 
    StandardReadPolicy>;

using double_prob_no_bucket_standard_read = MyHashTable<
    KeyType, ValueType,
    0,0,
    hash_function,
    DoubleHashinglProbingPolicy,
    false,CG_size, 
    StandardReadPolicy>;

using lin_prob_bucket_standard_read = MyHashTable<
    KeyType, ValueType,
    0,0,
    hash_function,
    LinearProbingPolicy,
    true,CG_size, 
    StandardReadPolicy>;

using exp_prob_bucket_standard_read = MyHashTable<
    KeyType, ValueType,
    0,0,
    hash_function,
    QuadraticProbingPolicy,
    true,CG_size, 
    StandardReadPolicy>;

using double_prob_bucket_standard_read = MyHashTable<
    KeyType, ValueType,
    0,0,
    hash_function,
    DoubleHashinglProbingPolicy,
    true,CG_size, 
    StandardReadPolicy>;

};


int main(int argc, char* argv[])
{


    const uint64_t max_keys = 1UL << 28;

    const bool print_headers = true;

    uint64_t dev_id = 0;
    if(argc > 2) dev_id = std::atoi(argv[2]);
    cudaSetDevice(dev_id); CUERR

    KeyType * keys_d = nullptr;
    if(argc > 1)
        keys_d = load_keys<KeyType>(argv[1], max_keys);
    else
        keys_d = generate_keys<KeyType>(max_keys, 1);

    

    float load_factor = 0.8;

    single_value_benchmark<dycuckoo_hash_table_t>(
        keys_d, max_keys, {max_keys}, {load_factor}, print_headers);

    single_value_benchmark<warpcore_hash_table_t>(
        keys_d, max_keys, {max_keys}, {load_factor}, print_headers);



    single_value_benchmark<HashTablesWithCG<1>::lin_prob_no_bucket_standard_read>(
        keys_d, max_keys, {max_keys}, {load_factor}, print_headers);

    single_value_benchmark<HashTablesWithCG<2>::lin_prob_no_bucket_standard_read>(
        keys_d, max_keys, {max_keys}, {load_factor}, print_headers);

    single_value_benchmark<HashTablesWithCG<4>::lin_prob_no_bucket_standard_read>(
        keys_d, max_keys, {max_keys}, {load_factor}, print_headers);

    single_value_benchmark<HashTablesWithCG<8>::lin_prob_no_bucket_standard_read>(
        keys_d, max_keys, {max_keys}, {load_factor}, print_headers);

    single_value_benchmark<HashTablesWithCG<16>::lin_prob_no_bucket_standard_read>(
        keys_d, max_keys, {max_keys}, {load_factor}, print_headers);

    single_value_benchmark<HashTablesWithCG<32>::lin_prob_no_bucket_standard_read>(
        keys_d, max_keys, {max_keys}, {load_factor}, print_headers);

    


    single_value_benchmark<HashTablesWithCG<1>::lin_prob_bucket_standard_read>(
        keys_d, max_keys, {max_keys}, {load_factor}, print_headers);

    single_value_benchmark<HashTablesWithCG<2>::lin_prob_bucket_standard_read>(
        keys_d, max_keys, {max_keys}, {load_factor}, print_headers);

    single_value_benchmark<HashTablesWithCG<4>::lin_prob_bucket_standard_read>(
        keys_d, max_keys, {max_keys}, {load_factor}, print_headers);

    single_value_benchmark<HashTablesWithCG<8>::lin_prob_bucket_standard_read>(
        keys_d, max_keys, {max_keys}, {load_factor}, print_headers);

    single_value_benchmark<HashTablesWithCG<16>::lin_prob_bucket_standard_read>(
        keys_d, max_keys, {max_keys}, {load_factor}, print_headers);

    single_value_benchmark<HashTablesWithCG<32>::lin_prob_bucket_standard_read>(
        keys_d, max_keys, {max_keys}, {load_factor}, print_headers);

    
    single_value_benchmark<HashTablesWithCG<1>::double_prob_no_bucket_standard_read>(
        keys_d, max_keys, {max_keys}, {load_factor}, print_headers);

    single_value_benchmark<HashTablesWithCG<2>::double_prob_no_bucket_standard_read>(
        keys_d, max_keys, {max_keys}, {load_factor}, print_headers);

    single_value_benchmark<HashTablesWithCG<4>::double_prob_no_bucket_standard_read>(
        keys_d, max_keys, {max_keys}, {load_factor}, print_headers);

    single_value_benchmark<HashTablesWithCG<8>::double_prob_no_bucket_standard_read>(
        keys_d, max_keys, {max_keys}, {load_factor}, print_headers);

    single_value_benchmark<HashTablesWithCG<16>::double_prob_no_bucket_standard_read>(
        keys_d, max_keys, {max_keys}, {load_factor}, print_headers);

    single_value_benchmark<HashTablesWithCG<32>::double_prob_no_bucket_standard_read>(
        keys_d, max_keys, {max_keys}, {load_factor}, print_headers);

    


    single_value_benchmark<HashTablesWithCG<1>::double_prob_bucket_standard_read>(
        keys_d, max_keys, {max_keys}, {load_factor}, print_headers);

    single_value_benchmark<HashTablesWithCG<2>::double_prob_bucket_standard_read>(
        keys_d, max_keys, {max_keys}, {load_factor}, print_headers);

    single_value_benchmark<HashTablesWithCG<4>::double_prob_bucket_standard_read>(
        keys_d, max_keys, {max_keys}, {load_factor}, print_headers);

    single_value_benchmark<HashTablesWithCG<8>::double_prob_bucket_standard_read>(
        keys_d, max_keys, {max_keys}, {load_factor}, print_headers);

    single_value_benchmark<HashTablesWithCG<16>::double_prob_bucket_standard_read>(
        keys_d, max_keys, {max_keys}, {load_factor}, print_headers);

    single_value_benchmark<HashTablesWithCG<32>::double_prob_bucket_standard_read>(
        keys_d, max_keys, {max_keys}, {load_factor}, print_headers);

    
    // single_value_benchmark<HashTablesWithCG<1>::exp_prob_no_bucket_standard_read>(
    //     keys_d, max_keys, {max_keys}, {load_factor}, print_headers);

    // single_value_benchmark<HashTablesWithCG<2>::exp_prob_no_bucket_standard_read>(
    //     keys_d, max_keys, {max_keys}, {load_factor}, print_headers);

    // single_value_benchmark<HashTablesWithCG<4>::exp_prob_no_bucket_standard_read>(
    //     keys_d, max_keys, {max_keys}, {load_factor}, print_headers);

    // single_value_benchmark<HashTablesWithCG<8>::exp_prob_no_bucket_standard_read>(
    //     keys_d, max_keys, {max_keys}, {load_factor}, print_headers);

    // single_value_benchmark<HashTablesWithCG<16>::exp_prob_no_bucket_standard_read>(
    //     keys_d, max_keys, {max_keys}, {load_factor}, print_headers);

    // single_value_benchmark<HashTablesWithCG<32>::exp_prob_no_bucket_standard_read>(
    //     keys_d, max_keys, {max_keys}, {load_factor}, print_headers);


    // single_value_benchmark<HashTablesWithCG<1>::exp_prob_bucket_standard_read>(
    //     keys_d, max_keys, {max_keys}, {load_factor}, print_headers);

    // single_value_benchmark<HashTablesWithCG<2>::exp_prob_bucket_standard_read>(
    //     keys_d, max_keys, {max_keys}, {load_factor}, print_headers);

    // single_value_benchmark<HashTablesWithCG<4>::exp_prob_bucket_standard_read>(
    //     keys_d, max_keys, {max_keys}, {load_factor}, print_headers);

    // single_value_benchmark<HashTablesWithCG<8>::exp_prob_bucket_standard_read>(
    //     keys_d, max_keys, {max_keys}, {load_factor}, print_headers);

    // single_value_benchmark<HashTablesWithCG<16>::exp_prob_bucket_standard_read>(
    //     keys_d, max_keys, {max_keys}, {load_factor}, print_headers);

    // single_value_benchmark<HashTablesWithCG<32>::exp_prob_bucket_standard_read>(
    //     keys_d, max_keys, {max_keys}, {load_factor}, print_headers);



    cudaFree(keys_d); CUERR
}


// static constexpr size_t roundUp( size_t x, size_t y) {
//     return ((x + y - 1) / y) * y;
// }

// int main(int argc, char* argv[]) {
//     using KeyType = std::uint32_t;
//     using value_t = std::uint32_t;

//     const uint64_t max_keys = 1UL << 10;


//     uint64_t dev_id = 0;
//     if(argc > 2) dev_id = std::atoi(argv[2]);
//     cudaSetDevice(dev_id); CUERR

//     KeyType * keys_d = nullptr;
//     if(argc > 1)
//         keys_d = load_keys<KeyType>(argv[1], max_keys);
//     else
//         keys_d = generate_keys<KeyType>(max_keys, 1);

//     using my_hash_t = MyHashTable<
//         KeyType, value_t,
//         0,0,
//         hash_function,
//         QuadraticProbingPolicy,
//         true,8, 
//         Vectroized2ReadPolicy>;


//     my_hash_t hash_table(roundUp(max_keys / 0.8, 128));
//     CUERR

//     hash_table.insert(keys_d, keys_d, max_keys);
//     cudaDeviceSynchronize(); CUERR
//     // hash_table.print(); CUERR
//     cudaDeviceSynchronize(); CUERR
//     std::cout << "inserted" << std::endl;
//     value_t* values_d = nullptr;
//     cudaMalloc(&values_d, sizeof(value_t)*max_keys); CUERR
//     cudaMemset(values_d, 0, sizeof(value_t)*max_keys); CUERR
//     hash_table.retrieve(keys_d, max_keys, values_d);

//     cudaDeviceSynchronize(); CUERR
//     printArray<<<1,1>>>(values_d, max_keys);
//     cudaDeviceSynchronize(); CUERR

    

// }