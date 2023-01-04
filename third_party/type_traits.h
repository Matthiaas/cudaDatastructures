/* Copyright (C) 2022 Matthias Bungeroth - All Rights Reserved
 */

namespace mb {

template <typename... T>
struct set {};

template <typename T, typename U>
struct concat;

template <typename... T, typename... U>
struct concat<set<T...>, set<U...>> {
    using type = set<T..., U...>;
};


namespace internal {

    template <typename CurList, typename... Sets>
    struct cross_product_helper;

    template <typename... T>
    struct cross_product_helper<set<T...>> {
        using type = set<set<T...>>;
    };

    template <typename... T, typename U, typename... Sets>
    struct cross_product_helper<set<T...>, set<U>, Sets...> {
        using type = typename cross_product_helper<set<T..., U>, Sets...>::type;
    };

    template <typename... T, typename U, typename... Us, typename... Sets>
    struct cross_product_helper<set<T...>, set<U,Us...>, Sets...> {
        using next = typename cross_product_helper<set<T..., U>, Sets...>::type;
        using skip = typename cross_product_helper<set<T...>, set<Us...>, Sets...>::type;
        using type = typename concat<next,skip>::type;
    };

    template < template<typename...> typename F, typename... Args, typename... fArgs>
    void for_each(set<>, fArgs...) {
    }

    template < template<typename...> typename F, typename... Args, typename... S, typename... fArgs>
    void for_each(set<set<Args...>, S...>, fArgs... fargs) {
        F<Args...>()(fargs...);
        for_each<F>(set<S...> {},fargs...);
    }

}

template <typename... Sets>
struct cross_product {
    using type = typename internal::cross_product_helper<set<>, Sets...>::type;
};

template <class CP, template<typename...> typename Runner, typename... Args>
void for_each(Args... args) {
    typename CP::type x;
    internal::for_each<Runner>(x, args...);
}

// Usage:
// template<typename... Args>
// struct Runner {
//     void operator()() {
//         CallPrint<Args...>();
//     }
// };



// using cp = cross_product<set<A,B>, set<A,B>>;
// using res = set<
// set<A,A>,
// set<A,B>,
// set<B,A>,
// set<B,B>
// >; 

}