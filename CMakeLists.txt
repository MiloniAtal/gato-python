cmake_minimum_required(VERSION 3.6)
enable_language(CUDA)
find_package(PythonInterp 3.6 REQUIRED)
find_package(PythonLibs 3.6 REQUIRED)
include(CheckLanguage)
check_language(CUDA)

include_directories(
    ${PYTHON_INCLUDE_DIRS}
    ./pybind11/include
	./GLASS
	./GBD-PCG/include
    )

link_directories(
    /usr/local/cuda/lib64
    )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DKNOT_POINTS=${K} -DSTATE_SIZE=${S} -DCONTROL_SIZE=${C} -DNX=${NX} -DNC=${NC} -lcublas")
add_library(gpu_library SHARED
  gpu_library.cu)

target_link_libraries(gpu_library
  ${PYTHON_LIBRARIES}
  cudart
  cublas)

set_target_properties(gpu_library PROPERTIES PREFIX "")
