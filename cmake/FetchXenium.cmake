include(FetchContent)
set(FETCHCONTENT_QUIET ON)

message(STATUS "Cloning External Project: Xenium")
get_filename_component(FC_BASE "${PROJECT_SOURCE_DIR}/externals"
                REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")
set(FETCHCONTENT_BASE_DIR ${FC_BASE})

FetchContent_Declare(
  xenium
    GIT_REPOSITORY https://github.com/mpoeter/xenium.git
    GIT_TAG        master
)

set(XENIUM_INCLUDE_DIR 
    ${FETCHCONTENT_BASE_DIR}/xenium-src)