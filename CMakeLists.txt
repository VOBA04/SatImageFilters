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
set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 72 75 80 86)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_definitions(-DPROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

find_package(TIFF REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(GTest REQUIRED)
enable_language(CUDA)
enable_testing()

add_test(NAME unit_tests COMMAND unit_tests)

if (NOT TIFF_FOUND)
    message(FATAL_ERROR "LibTIFF library not found")
endif()
if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV library not found")
endif()
if (NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "CUDA library not found")
endif()
if (NOT GTest_FOUND)
    message(FATAL_ERROR "GTest library not found")
endif()

include_directories(${SOURCE_DIR})
include_directories(${INCLUDE_DIR})
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${GTEST_INCLUDE_DIRS})

file(GLOB_RECURSE SOURCES "${SOURCE_DIR}/*.cpp" "${SOURCE_DIR}/*.cu")
list(REMOVE_ITEM SOURCES "${SOURCE_DIR}/speedtest.cpp")
file(GLOB_RECURSE SPEEDTEST_SOURCES "${SOURCE_DIR}/*.cpp" "${SOURCE_DIR}/*.cu")
list(REMOVE_ITEM SPEEDTEST_SOURCES "${SOURCE_DIR}/main.cpp")
file(GLOB_RECURSE TEST_SOURCES "${TESTS_DIR}/*.cpp" "${SOURCE_DIR}/*.cpp" "${SOURCE_DIR}/*.cu")
list(REMOVE_ITEM TEST_SOURCES "${SOURCE_DIR}/main.cpp")
list(REMOVE_ITEM TEST_SOURCES "${SOURCE_DIR}/speedtest.cpp") 

add_executable(${PROJECT_NAME} ${SOURCES})
add_executable(speedtest ${SPEEDTEST_SOURCES})
add_executable(unit_tests ${TEST_SOURCES})

if(CMAKE_BUILD_TYPE MATCHES Debug)
    target_compile_options(unit_tests PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:--coverage;-fprofile-arcs;-ftest-coverage>
    )
    target_link_options(unit_tests PRIVATE --coverage)
    message(STATUS "Coverage flags enabled for unit_tests (CXX): --coverage -fprofile-arcs -ftest-coverage")
endif()

target_link_libraries(${PROJECT_NAME} TIFF::TIFF ${CUDA_LIBRARIES})
target_link_libraries(speedtest TIFF::TIFF ${CUDA_LIBRARIES})
target_link_libraries(unit_tests PRIVATE TIFF::TIFF gtest gtest_main pthread ${OpenCV_LIBRARIES})

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(speedtest PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_custom_target(coverage
    COMMAND lcov --zerocounters --directory .
    COMMAND ${CMAKE_COMMAND} --build . --target unit_tests
    COMMAND ctest -C Debug -R ^unit_tests$
    COMMAND geninfo ${CMAKE_BINARY_DIR} --base-directory ${CMAKE_SOURCE_DIR} --output-filename coverage.info --ignore-errors source,gcov,empty --rc geninfo_unexecuted_blocks=1
    COMMAND lcov --ignore-errors unused,source,empty --remove coverage.info '/usr/*' '*/tests/*' '/home/vova/OpenCV/*' '${CMAKE_SOURCE_DIR}/tests/*' '${CMAKE_SOURCE_DIR}/src/*.cu' --output-file coverage.info
    COMMAND genhtml coverage.info --output-directory coverage_report --ignore-errors empty || true
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Generating code coverage report"
    DEPENDS unit_tests
)

message(STATUS "TIFF_INCLUDE_DIRS: ${TIFF_INCLUDE_DIRS}")
message(STATUS "TIFF_LIBRARIES: ${TIFF_LIBRARIES}")
message(STATUS "CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

