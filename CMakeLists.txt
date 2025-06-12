cmake_minimum_required(VERSION 3.20)

project(SatImageFilters LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# -fsanitize=address Санитайзер мешает cuda
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MDd /W2 /EHsc")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/W2,/EHsc -Wno-deprecated-gpu-targets")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g  -Wall -Wextra -pedantic")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -Wall -Wno-deprecated-gpu-targets")
endif()
set(SOURCE_DIR "${CMAKE_SOURCE_DIR}/src")
set(INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")
set(TESTS_DIR "${CMAKE_SOURCE_DIR}/tests")
set(SOURCE_GUI_DIR "${SOURCE_DIR}/gui")
set(INCLUDE_GUI_DIR "${INCLUDE_DIR}/gui")
set(MAINS_DIR "${SOURCE_DIR}/mains")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CUDA_VERSION "AUTO" CACHE STRING "CUDA Toolkit version (e.g., 10.2, 11.4, or AUTO)")
if(CUDA_VERSION STREQUAL "10.2")
    set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2")
    set(CMAKE_CUDA_ARCHITECTURES 30 50 52 60 61 70 72 75)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
elseif(CUDA_VERSION STREQUAL "11.4")
    set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4")
    set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 72 75 80 86)
else()
    # Автоматический выбор
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "11.0")
        set(CMAKE_CUDA_ARCHITECTURES 30 50 52 60 61 70 72 75)
    else()
        set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 72 75 80 86)
    endif()
endif()

set(CUDA_ARCH "AUTO" CACHE STRING "CUDA architecture (e.g., 30, 50, 75, or AUTO for default)")
if(NOT CUDA_ARCH STREQUAL "AUTO")
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})
endif()

add_definitions(-DPROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

enable_language(CUDA)
find_package(TIFF REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(GTest REQUIRED)
find_package(Qt6 COMPONENTS Core Gui Widgets)

include(GoogleTest)

add_test(NAME unit_tests COMMAND unit_tests)

if(Qt6_FOUND)
    message(STATUS "Qt6 found, GUI target will be built")
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTORCC ON)
    set(CMAKE_AUTOUIC ON)

    file(GLOB GUI_SOURCES "${SOURCE_DIR}/*.cpp" "${SOURCE_DIR}/*.cu" "${SOURCE_GUI_DIR}/*.cpp" "${SOURCE_GUI_DIR}/*.ui" "${MAINS_DIR}/gui.cpp")
    file(GLOB GUI_INCLUDE "${INCLUDE_GUI_DIR}/*.h")

    qt6_wrap_cpp(GUI_MOC_SOURCES ${GUI_INCLUDE})
    add_executable(gui ${GUI_SOURCES} ${GUI_MOC_SOURCES})

    target_link_libraries(gui PRIVATE TIFF::TIFF ${CUDA_LIBRARIES} Qt6::Core Qt6::Gui Qt6::Widgets ${OpenCV_LIBRARIES})
    target_include_directories(gui PRIVATE ${INCLUDE_GUI_DIR})
    if(CMAKE_AUTOUIC)
        target_include_directories(gui PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    endif()
    set_target_properties(gui PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_definitions(gui PRIVATE USE_QT)
else()
    message(WARNING "Qt6 not found, GUI target will not be built")
endif()

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

file(GLOB SOURCES "${SOURCE_DIR}/*.cpp" "${SOURCE_DIR}/*.cu" "${MAINS_DIR}/main.cpp")
file(GLOB SPEEDTEST_SOURCES "${SOURCE_DIR}/*.cpp" "${SOURCE_DIR}/*.cu" "${MAINS_DIR}/speedtest.cpp")
file(GLOB TEST_SOURCES "${TESTS_DIR}/*.cpp" "${SOURCE_DIR}/*.cpp" "${SOURCE_DIR}/*.cu")

add_executable(${PROJECT_NAME} ${SOURCES})
add_executable(speedtest ${SPEEDTEST_SOURCES})
add_executable(unit_tests ${TEST_SOURCES})

add_test(NAME unit_test COMMAND unit_tests)

if(NOT MSVC)
    if(CMAKE_BUILD_TYPE MATCHES Debug)
        target_compile_options(unit_tests PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:--coverage;-fprofile-arcs;-ftest-coverage>
            $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info;-Xcompiler;--coverage;-Xcompiler;-fprofile-arcs;-Xcompiler;-ftest-coverage>
        )
        target_link_options(unit_tests PRIVATE --coverage)
    endif()
endif()

target_link_libraries(${PROJECT_NAME} TIFF::TIFF ${CUDA_LIBRARIES} cudart)
target_link_libraries(speedtest TIFF::TIFF ${CUDA_LIBRARIES} cudart)
target_link_libraries(unit_tests PRIVATE TIFF::TIFF ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} ${OpenCV_LIBRARIES})

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(speedtest PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(unit_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if(DEFINED ENV{GITHUB_ACTIONS})
    set(EXCLUDE_CUDA "*/src/*.cu")
else()
    set(EXCLUDE_CUDA "")
endif()

add_custom_target(coverage
    COMMAND lcov --zerocounters --directory .
    COMMAND ${CMAKE_COMMAND} --build . --target unit_tests
    COMMAND ctest -C Debug -R ^unit_tests$
    COMMAND geninfo ${CMAKE_BINARY_DIR} --base-directory ${CMAKE_SOURCE_DIR} --output-filename coverage.info --ignore-errors source,gcov,mismatch,version --rc geninfo_unexecuted_blocks=1
    COMMAND lcov --ignore-errors unused,source,empty,mismatch,version --remove coverage.info '/usr/*' '*/tests/*' '*/OpenCV/*' '/tmp/*' ${EXCLUDE_CUDA} --output-file coverage.info
    COMMAND genhtml coverage.info --output-directory coverage_report --ignore-errors empty || true
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Generating code coverage report"
    DEPENDS unit_tests
)

message(STATUS "TIFF_INCLUDE_DIRS: ${TIFF_INCLUDE_DIRS}")
message(STATUS "TIFF_LIBRARIES: ${TIFF_LIBRARIES}")
message(STATUS "CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
message(STATUS "NVCC Version: ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "GTEST_INCLUDE_DIR: ${GTEST_INCLUDE_DIR}")
