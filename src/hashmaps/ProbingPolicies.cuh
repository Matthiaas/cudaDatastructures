
template<class KeyType, auto &HashFunction, bool UseBuckets, size_t CooperativeGroupSize = 16>
struct LinearProbingPolicy {

    __device__  LinearProbingPolicy() {}

    __device__ ~LinearProbingPolicy() {}

    __device__ size_t begin(const KeyType key) noexcept {
        pos_ = HashFunction(key);
        // printf("hash %lld from key %d\n", pos_, key);
        return pos_;
    }

    __device__ size_t next() noexcept {
        pos_ = (pos_ + pos_inc);
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
};

template<class KeyType, auto &HashFunction, bool UseBuckets, size_t CooperativeGroupSize>
struct ExponentialProbingPolicy {

    __device__  ExponentialProbingPolicy() {}

    __device__ ~ExponentialProbingPolicy() {}

    __device__ size_t begin(const KeyType key) noexcept {
        pos_ = HashFunction(key);
        scaler_ = 1;
        // printf("hash %lld from key %d\n", pos_, key);
        return pos_;
    }

    __device__ size_t next() noexcept {
        pos_ = (pos_ + pos_inc * scaler_);
        scaler_ *= 2;
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
    size_t scaler_;

};
