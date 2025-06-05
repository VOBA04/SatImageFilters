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
set(SOURCE_GUI_DIR "${SOURCE_DIR}/gui")
set(INCLUDE_GUI_DIR "${INCLUDE_DIR}/gui")
set(MAINS_DIR "${SOURCE_DIR}/mains")
set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 72 75 80 86)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_AUTOMOC ON)  
set(CMAKE_AUTORCC ON)  
set(CMAKE_AUTOUIC ON)

add_definitions(-DPROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

find_package(TIFF REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(GTest REQUIRED)
find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets)
enable_language(CUDA)

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
if (NOT Qt6_FOUND)
    message(FATAL_ERROR "Qt5 not found")
endif()

include_directories(${SOURCE_DIR})
include_directories(${INCLUDE_DIR})
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${GTEST_INCLUDE_DIRS})

file(GLOB SOURCES "${SOURCE_DIR}/*.cpp" "${SOURCE_DIR}/*.cu" "${MAINS_DIR}/main.cpp")
file(GLOB SPEEDTEST_SOURCES "${SOURCE_DIR}/*.cpp" "${SOURCE_DIR}/*.cu" "${MAINS_DIR}/speedtest.cpp")
file(GLOB TEST_SOURCES "${TESTS_DIR}/*.cpp" "${SOURCE_DIR}/*.cpp" "${SOURCE_DIR}/*.cu")
file(GLOB GUI_SOURCES "${SOURCE_DIR}/*.cpp" "${SOURCE_DIR}/*.cu" "${SOURCE_GUI_DIR}/*.cpp" "${SOURCE_GUI_DIR}/*.ui" "${MAINS_DIR}/gui.cpp")
file(GLOB GUI_INCLUDE "${INCLUDE_GUI_DIR}/*.h")

add_executable(${PROJECT_NAME} ${SOURCES})
add_executable(speedtest ${SPEEDTEST_SOURCES})
add_executable(unit_tests ${TEST_SOURCES})
qt6_wrap_cpp(GUI_MOC_SOURCES ${GUI_INCLUDE})
add_executable(gui ${GUI_SOURCES} ${GUI_MOC_SOURCES})

if(CMAKE_BUILD_TYPE MATCHES Debug)
    target_compile_options(unit_tests PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:--coverage;-fprofile-arcs;-ftest-coverage>
        $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info;-Xcompiler;--coverage;-Xcompiler;-fprofile-arcs;-Xcompiler;-ftest-coverage>
    )
    target_link_options(unit_tests PRIVATE --coverage)
endif()

target_link_libraries(${PROJECT_NAME} TIFF::TIFF ${CUDA_LIBRARIES})
target_link_libraries(speedtest TIFF::TIFF ${CUDA_LIBRARIES})
target_link_libraries(unit_tests PRIVATE TIFF::TIFF gtest gtest_main pthread ${OpenCV_LIBRARIES})
target_link_libraries(gui TIFF::TIFF ${CUDA_LIBRARIES} Qt6::Core Qt6::Gui Qt6::Widgets ${OpenCV_LIBS})

target_include_directories(gui PRIVATE ${INCLUDE_GUI_DIR})
if(CMAKE_AUTOUIC)
    target_include_directories(gui PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(speedtest PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(unit_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(gui PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_definitions(gui PRIVATE USE_QT)

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
