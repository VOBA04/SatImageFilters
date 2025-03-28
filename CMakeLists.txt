cmake_minimum_required(VERSION 3.20)

project(SatImageFilters LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# -fsanitize=address Санитайзер мешает cuda
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g  -Wall -Wextra -pedantic")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -Wall -Wno-deprecated-gpu-targets")
set(SOURCE_DIR "${CMAKE_SOURCE_DIR}/src")
set(INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")
set(TESTS_DIR "${CMAKE_SOURCE_DIR}/tests")
set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 72 75 80 86 87 89 90 100 101 120)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_definitions(-DPROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

find_package(TIFF REQUIRED)
# find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)
enable_testing()

if (NOT TIFF_FOUND)
    message(FATAL_ERROR "LibTIFF library not found")
endif()

include_directories(${SOURCE_DIR})
include_directories(${INCLUDE_DIR})
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

file(GLOB_RECURSE SOURCES "${SOURCE_DIR}/*.cpp" "${SOURCE_DIR}/*.cu")
file(GLOB_RECURSE TEST_SOURCES "${TESTS_DIR}/*.cpp")

add_executable(${PROJECT_NAME} ${SOURCES})
add_executable(test_project ${TEST_SOURCES})
# target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} LANGUAGE CXX)
# include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} TIFF::TIFF ${CUDA_LIBRARIES})
# target_link_libraries(test_project TIFF::TIFF gtest gtest_main pthread ${OpenCV_LIBRARIES})
message(STATUS "TIFF_INCLUDE_DIRS: ${TIFF_INCLUDE_DIRS}")
message(STATUS "TIFF_LIBRARIES: ${TIFF_LIBRARIES}")
message(STATUS "CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
