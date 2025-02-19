cmake_minimum_required(VERSION 3.10)

project(SatImageFilters)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fsanitize=address -Wall -Wextra -pedantic")
set(SOURCE_DIR "${CMAKE_SOURCE_DIR}/src")
set(INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")

add_definitions(-DPROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

find_package(TIFF REQUIRED)

if (NOT TIFF_FOUND)
    message(FATAL_ERROR "LibTIFF library not found")
endif()

file(GLOB_RECURSE SOURCES "${SOURCE_DIR}/*.cpp")
include_directories(${SOURCE_DIR})
include_directories(${INCLUDE_DIR})


add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} TIFF::TIFF)
message(STATUS "TIFF_INCLUDE_DIRS: ${TIFF_INCLUDE_DIRS}")
message(STATUS "TIFF_LIBRARIES: ${TIFF_LIBRARIES}")