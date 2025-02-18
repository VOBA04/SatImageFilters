cmake_minimum_required(VERSION 3.10)

project(SatImageFilters)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(SOURCE_DIR "${CMAKE_SOURCE_DIR}/src")
set(INCLUSE_DIR "${CMAKE_SOURCE_DIR}/include")

find_package(TIFF REQUIRED)

if (NOT TIFF_FOUND)
    message(FATAL_ERROR "LibTIFF library not found")
endif()

file(GLOB_RECURSE SOURCES "${SOURCE_DIR}/*.cpp")
include_directories(${SOURCE_DIR})
include_directories(${INCLUSE_DIR})


add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} TIFF::TIFF)
message(STATUS "TIFF_INCLUDE_DIRS: ${TIFF_INCLUDE_DIRS}")
message(STATUS "TIFF_LIBRARIES: ${TIFF_LIBRARIES}")