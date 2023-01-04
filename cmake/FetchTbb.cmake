include(ExternalProject)

ExternalProject_Add(
  oneTBBDownload
  GIT_REPOSITORY "https://github.com/oneapi-src/oneTBB.git"
  GIT_TAG "v2021.8.0"
  
  UPDATE_COMMAND ""
  PATCH_COMMAND ""
  
  SOURCE_DIR "${CMAKE_SOURCE_DIR}/externals/oneTBB"
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${GLOBAL_OUTPUT_PATH}/oneTBB -DTBB_TEST=OFF
  
  TEST_COMMAND ""
)

add_library(oneTBB SHARED IMPORTED)
set_target_properties(oneTBB PROPERTIES IMPORTED_LOCATION ${GLOBAL_OUTPUT_PATH}/oneTBB/lib/libtbb.so)
