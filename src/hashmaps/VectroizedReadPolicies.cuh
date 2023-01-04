#ifndef VECTORIZED_READ_POLICY_H
#define VECTORIZED_READ_POLICY_H

template <typename E, size_t S>
struct Array {
  E data[S];
};

template <class T>
struct StandardReadPolicy {
  static constexpr size_t key_count = 1;
  using ArrayType = Array<T, key_count>;

  __device__ static constexpr ArrayType read(T* key) noexcept { return {*key}; }
};

template <class T>
struct Vectroized2ReadPolicy {
  static constexpr size_t key_count = 2;
  using ArrayType = Array<T, key_count>;
  static_assert(sizeof(T) == 4, "T must be 4 or 8 bytes");

  using T2 = std::conditional_t<sizeof(T) == 4, int2, int4>;
  static_assert(std::is_same_v<T2, int2>, "T must be int2");
  __device__ static constexpr ArrayType read(T* key) noexcept {
    if constexpr (std::is_same_v<T2, int2>) {
      T2 key2 = *reinterpret_cast<T2*>(key);
      return {static_cast<T>(key2.x), static_cast<T>(key2.y)};
    } else {
      // Todo: support int64
      printf("Not implemented");
    }
  }
};

template <class T>
struct Vectroized4ReadPolicy {
  static constexpr size_t key_count = 4;
  using ArrayType = Array<T, key_count>;
  static_assert(sizeof(T) == 4, "T must be 4 bytes");

  using T4 = int4;
  __device__ static constexpr ArrayType read(T* key) noexcept {
    T4 key4 = *reinterpret_cast<T4*>(key);
    return {static_cast<T>(key4.x), static_cast<T>(key4.y),
            static_cast<T>(key4.z), static_cast<T>(key4.w)};
  }
};

#endif  // VECTORIZED_READ_POLICY_H
