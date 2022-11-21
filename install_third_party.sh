# Install Warpcore
# git clone https://github.com/sleeepyjack/warpcore.git
# mv warpcore/include/warpcore/ third_party/warpcore
# rm -rf warpcore 
# # Install Warpcore Helpers
# mkdir third_party/helpers
# git clone https://gitlab.rlp.net/pararch/hpc_helpers.git
# mv hpc_helpers/include/cuda_helpers.cuh third_party/helpers/cuda_helpers.cuh
# mv hpc_helpers/include/packed_types.cuh third_party/helpers/packed_types.cuh
# mv hpc_helpers/include/hpc_helpers.h third_party/helpers/hpc_helpers.h
# mv hpc_helpers/include/type_helpers.h third_party/helpers/type_helpers.h
# mv hpc_helpers/include/io_helpers.h third_party/helpers/io_helpers.h
rm -rf hpc_helpers

# # Install dycuckoo
cd third_party
git clone https://github.com/zhuqiweigit/DyCuckoo.git dycuckoo
cd ..
# Remove some unneeded includes that I dont know where they come from
sed -i '/#include <helper_functions.h>/d' third_party/dycuckoo/dynamicHash/core/dynamic_cuckoo.cuh
sed -i '/#include <helper_functions.h>/d' third_party/dycuckoo/dynamicHash/core/static_cuckoo.cuh
sed -i '/#include <helper_functions.h>/d' third_party/dycuckoo/dynamicHash/core/dynamic_hash.cuh
sed -i '/#include <helper_functions.h>/d' third_party/dycuckoo/dynamicHash/test/dynamic_test.cu
sed -i '/#include <helper_functions.h>/d' third_party/dycuckoo/dynamicHash/test/static_test.cu

sed -i '/#include <helper_functions.h>/d' third_party/dycuckoo/dynamicHash_lock/core/dynamic_cuckoo.cuh
sed -i '/#include <helper_functions.h>/d' third_party/dycuckoo/dynamicHash_lock/core/static_cuckoo.cuh
sed -i '/#include <helper_functions.h>/d' third_party/dycuckoo/dynamicHash_lock/core/dynamic_hash.cuh
sed -i '/#include <helper_functions.h>/d' third_party/dycuckoo/dynamicHash_lock/test/dynamic_test.cu
sed -i '/#include <helper_functions.h>/d' third_party/dycuckoo/dynamicHash_lock/test/static_test.cu



# Install dycuckoo helpers
cd third_party
wget https://raw.githubusercontent.com/NVIDIA/cuda-samples/master/Common/helper_cuda.h
wget https://raw.githubusercontent.com/NVIDIA/cuda-samples/master/Common/helper_string.h

rm dycuckoo/dynamicHash_lock -r
