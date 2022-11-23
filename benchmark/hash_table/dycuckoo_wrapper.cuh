
#include "dycuckoo/dynamicHash/core/dynamic_hash.cuh"
#include "dycuckoo/dynamicHash/data/data_layout.cuh"


class DycuckooHashTableWrapper
{


    cnmemDevice_t device;
public:
    using data_t = DataLayout<>::data_t;
    using key_type = DataLayout<>::key_t;
    using value_type = DataLayout<>::value_t;
    using index_type = std::uint64_t;
    using entry_t = DataLayout<>::entry_t;
    using bucket_t = DataLayout<>::bucket_t;
    using cuckoo_t = DataLayout<>::cuckoo_t;
    using error_table_t = DataLayout<>::error_table_t;
    static constexpr key_type empty_key = DataLayout<>::empty_key;
    static constexpr value_type empty_val = DataLayout<>::empty_val;
    static constexpr uint32_t bucket_size = DataLayout<>::bucket_size;
    static constexpr uint8_t table_num = DataLayout<>::table_num;

    static constexpr uint32_t thread_num = 512;
    static constexpr uint32_t block_num = 512;

    static std::string GetName() { return "DycuckooHashTableWrapper"; }

    DycuckooHashTableWrapper(uint64_t);
    ~DycuckooHashTableWrapper();

    void insert(
        const key_type * const keys_in,
        const value_type * const values_in,
        const index_type num_in);

     void retrieve(
        const key_type * const keys_in,
        const index_type num_in,
        value_type * const values_out);  

    index_type capacity() const noexcept
    {
        return capacity_;
    }  

    void init() const noexcept {}

    double load_factor() const noexcept
    {
        return static_cast<double>(inserted_elements_) / capacity_;
    }

private:
    uint64_t capacity_;
    uint64_t inserted_elements_;
    cuckoo_t *host_cuckoo_table;
    error_table_t* host_error_table;
};

DycuckooHashTableWrapper::DycuckooHashTableWrapper(uint64_t init_kv_num) : 
        inserted_elements_(0) {
    ///cnmem init
    memset(&device, 0, sizeof(device));
    device.size = (size_t)4*1024*1024*1024; /// more =(size_t) (0.95*props.totalGlobalMem);
    cnmemInit(1, &device, CNMEM_FLAGS_DEFAULT);
    checkCudaErrors(cudaGetLastError());
        
    uint64_t s = init_kv_num / (table_num * bucket_size);
    s = ch::nextPrime(s);
    uint32_t s_bucket = (s & 1) ? s + 1: s;
    capacity_ = s_bucket * table_num * bucket_size;

    host_cuckoo_table = (cuckoo_t *) malloc(sizeof(cuckoo_t));
    cuckoo_t::device_table_mem_init(*host_cuckoo_table, s_bucket);
    DynamicHash::meta_data_to_device(*host_cuckoo_table);

    checkCudaErrors(cudaGetLastError());

    //error table
    host_error_table = new error_table_t;
    host_error_table->device_mem_init();
    DynamicHash::meta_data_to_device(*host_error_table);
}

DycuckooHashTableWrapper::~DycuckooHashTableWrapper() {
        for(uint32_t i = 0; i < table_num; i++){
            cnmemFree(host_cuckoo_table->table_group[i], 0);
        }
        free(host_cuckoo_table);

        cnmemFree(host_error_table->error_keys, 0);
        cnmemFree(host_error_table->error_values, 0);
        free(host_error_table);

        cnmemRelease();
    }

void DycuckooHashTableWrapper::insert(
        const key_type * const keys_in,
        const value_type * const values_in,
        const index_type num_in) {
    inserted_elements_ += num_in;
    constexpr size_t block_size = 512;
    const size_t block_count = (num_in * 16 + block_size - 1) / block_size;
    DynamicHash::cuckoo_insert<<< block_count, block_size >>> (
        // Lets hope they not do soemting stupid.
        const_cast<key_type*>(keys_in), const_cast<value_type*>(values_in), num_in);   
}

void DycuckooHashTableWrapper::retrieve(
        const key_type * const keys_in,
        const index_type num_in,
        value_type * const values_out) {
    constexpr size_t block_size = 512;
    const size_t block_count = (num_in * 16 + block_size - 1) / block_size;
    DynamicHash::cuckoo_search <<< block_count, block_size >>> (
        const_cast<key_type*>(keys_in), values_out, num_in);
        
}