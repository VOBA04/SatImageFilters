cmake_minimum_required(VERSION 3.10)

project(SatImageFilters LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fsanitize=address -Wall -Wextra -pedantic")
set(SOURCE_DIR "${CMAKE_SOURCE_DIR}/src")
set(INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")
set(CUDA_SOURCE_DIR "${CMAKE_SOURCE_DIR}/cuda")

add_definitions(-DPROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

find_package(TIFF REQUIRED)
enable_language(CUDA)

if (NOT TIFF_FOUND)
    message(FATAL_ERROR "LibTIFF library not found")
endif()


include_directories(${SOURCE_DIR})
include_directories(${INCLUDE_DIR})
include_directories(${CUDA_INCLUDE_DIR})

file(GLOB_RECURSE SOURCES "${SOURCE_DIR}/*.cpp" "${CUDA_SOURCE_DIR}/*.cu")

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} TIFF::TIFF ${CUDA_LIBRARIES})
message(STATUS "TIFF_INCLUDE_DIRS: ${TIFF_INCLUDE_DIRS}")
message(STATUS "TIFF_LIBRARIES: ${TIFF_LIBRARIES}")

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)