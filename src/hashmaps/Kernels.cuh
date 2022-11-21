

template <typename HashTable>
__global__ 
void insert_kernel(
        const typename HashTable::key_type* keys, 
        const typename HashTable::value_type* values, 
        size_t count,
        HashTable call_on) {
    const size_t tid = helpers::global_thread_id();
    const size_t gid = tid / HashTable::cooperative_group_size;
    const auto group =
        cg::tiled_partition<HashTable::cooperative_group_size>(cg::this_thread_block());

    if(gid < count) {
        call_on.insert(keys[gid], values[gid], group);
    }
}

template <typename HashTable>
__global__ 
void retrieve_kernel(
        const typename HashTable::key_type* keys, 
        typename HashTable::value_type* values, 
        size_t count,
        HashTable call_on) {
    const size_t tid = helpers::global_thread_id();
    const size_t gid = tid / HashTable::cooperative_group_size;
    const auto group =
        cg::tiled_partition<HashTable::cooperative_group_size>(cg::this_thread_block());

    if(gid < count) {
        call_on.retrieve(keys[gid], values[gid], group);
    }
}

template <typename HashTable>
__global__ void print_kernel(HashTable call_on) {
    call_on.print();
}