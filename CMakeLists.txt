cmake_minimum_required(VERSION 3.20)
project(WaferVision VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp
)

# Source files
set(CORE_SOURCES
    src/cpp/core/ImageProcessor.cpp
    src/cpp/core/WaferData.cpp
    src/cpp/core/InspectionMetrics.cpp
)

set(ALGORITHM_SOURCES
    src/cpp/algorithms/WaferAlignment.cpp
    src/cpp/algorithms/DefectDetection.cpp
    src/cpp/algorithms/Metrology.cpp
    src/cpp/algorithms/EdgeReduction.cpp
    src/cpp/algorithms/PatternProcessing.cpp
)

set(CUDA_SOURCES
    src/cpp/cuda/GpuImageProcessor.cu
    src/cpp/cuda/GpuDefectDetection.cu
)

set(UTIL_SOURCES
    src/cpp/utils/Logger.cpp
    src/cpp/utils/ConfigManager.cpp
)

# Create library
add_library(wafervision_core SHARED
    ${CORE_SOURCES}
    ${ALGORITHM_SOURCES}
    ${UTIL_SOURCES}
)

target_link_libraries(wafervision_core
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
)

# Create CUDA library
add_library(wafervision_cuda SHARED
    ${CUDA_SOURCES}
)

target_link_libraries(wafervision_cuda
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    wafervision_core
)

# Main executable
add_executable(wafervision_cli
    src/cpp/main.cpp
)

target_link_libraries(wafervision_cli
    wafervision_core
    wafervision_cuda
)

# Python bindings
find_package(pybind11 REQUIRED)
pybind11_add_module(wafervision_python
    src/python/bindings/python_bindings.cpp
)

target_link_libraries(wafervision_python PRIVATE
    wafervision_core
    wafervision_cuda
)

# Installation
install(TARGETS wafervision_core wafervision_cuda wafervision_cli
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
)

install(DIRECTORY src/cpp/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)
