cmake_minimum_required(VERSION 3.10)
project(sgla_tracker LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

# TensorRT (typical Jetson paths)
set(TENSORRT_INCLUDE_DIR /usr/include/aarch64-linux-gnu)
set(TENSORRT_LIB_DIR /usr/lib/aarch64-linux-gnu)

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
)

link_directories(${TENSORRT_LIB_DIR})

add_executable(sgla_tracker infer_trt_vid.cpp)

target_link_libraries(sgla_tracker
    nvinfer
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
)
