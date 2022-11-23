
template<class KeyType, auto &HashFunction, bool UseBuckets, size_t CooperativeGroupSize = 16>
struct LinearProbingPolicy {

    static std::string GetName() {
        return "LinearProbingPolicy";
    }

    __device__  LinearProbingPolicy(size_t bucket_num) : bucket_num_(bucket_num) {}

    __device__ ~LinearProbingPolicy() {}

    __device__ size_t begin(const KeyType key) noexcept {
        pos_ = HashFunction(key) % bucket_num_;
        // printf("hash %lld from key %d\n", pos_, key);
        return pos_;
    }

    __device__ size_t next() noexcept {
        pos_ = (pos_ + pos_inc) % bucket_num_;
        return pos_;
    }

    __device__ size_t end() const noexcept {
        return ~0;
    }

private:
    static constexpr size_t pos_inc = std::conditional_t<
        UseBuckets, 
        std::integral_constant<size_t, 1>, 
        std::integral_constant<size_t, CooperativeGroupSize>>::value;
    size_t pos_;
    size_t bucket_num_;
};

template<class KeyType, auto &HashFunction, bool UseBuckets, size_t CooperativeGroupSize>
struct QuadraticProbingPolicy {

    __device__  QuadraticProbingPolicy(size_t bucket_num) :
     bucket_num_(bucket_num) {}

    __device__ ~QuadraticProbingPolicy() {}

    __device__ size_t begin(const KeyType key) noexcept {
        start_ = HashFunction(key) % bucket_num_;
        i_ = 1;
        // printf("hash %lld from key %d\n", pos_, key);
        return start_;
    }

    __device__ size_t next() noexcept {
        size_t pos = start_ + i_ + i_ * i_;
        return pos;
    }

    __device__ size_t end() const noexcept {
        return ~0;
    }

private:
    static constexpr size_t pos_inc = std::conditional_t<
        UseBuckets, 
        std::integral_constant<size_t, 1>, 
        std::integral_constant<size_t, CooperativeGroupSize>>::value;
    size_t start_;
    size_t bucket_num_;
    size_t i_;

};


template<class KeyType, auto &HashFunction, bool UseBuckets, size_t CooperativeGroupSize>
struct DoubleHashinglProbingPolicy {

    static std::string GetName() {
        return "DoubleHashinglProbingPolicy";
    }

    __device__  DoubleHashinglProbingPolicy(size_t bucket_num) : 
        bucket_num_(bucket_num) {}

    __device__ ~DoubleHashinglProbingPolicy() {}

    __device__ size_t begin(const KeyType key) noexcept {
        pos_ = HashFunction(key) % bucket_num_;
        if constexpr (UseBuckets) {
            base_ = (HashFunction(key + 1) % (bucket_num_ - 1) + 1);
        } else {
            base_ = (HashFunction(key + 1) % (bucket_num_ / CooperativeGroupSize - 1) + 1) * CooperativeGroupSize;
        }
        // printf("hash %lld from key %d\n", pos_, key);
        return pos_;
    }

    __device__ size_t next() noexcept {
        pos_ = (pos_ + pos_inc * base_) % bucket_num_;
        return pos_;
    }

    __device__ size_t end() const noexcept {
        return ~0;
    }

private:
    static constexpr size_t pos_inc = std::conditional_t<
        UseBuckets, 
        std::integral_constant<size_t, 1>, 
        std::integral_constant<size_t, CooperativeGroupSize>>::value;
    size_t pos_;
    size_t base_;
    size_t bucket_num_;

};
