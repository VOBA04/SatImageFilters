cmake_minimum_required(VERSION 3.20)

project(SatImageFilters LANGUAGES CXX CUDA)

set(EXTERNAL_BUILD_DIR "${CMAKE_SOURCE_DIR}/external_build")
file(MAKE_DIRECTORY ${EXTERNAL_BUILD_DIR})

if(WIN32)
    set(LIBTIFF_LIB "${EXTERNAL_BUILD_DIR}/libtiff_install/lib/tiff.lib")
    set(OPENCV_CORE_LIB "${EXTERNAL_BUILD_DIR}/opencv_install/lib/opencv_core*.lib")
else()
    set(LIBTIFF_LIB "${EXTERNAL_BUILD_DIR}/libtiff_install/lib/libtiff.so" "${EXTERNAL_BUILD_DIR}/libtiff_install/lib/libtiff.a")
    set(OPENCV_CORE_LIB "${EXTERNAL_BUILD_DIR}/opencv_install/lib/libopencv_core.so" "${EXTERNAL_BUILD_DIR}/opencv_install/lib/libopencv_core.a")
endif()

set(LIBTIFF_FOUND FALSE)
foreach(LIB_PATH ${LIBTIFF_LIB})
    if(EXISTS ${LIB_PATH})
        set(LIBTIFF_FOUND TRUE)
        message(STATUS "Found libtiff: ${LIB_PATH}")
        break()
    endif()
endforeach()
if(LIBTIFF_FOUND)
    message(STATUS "libtiff уже установлен, пропускаем сборку")
else()
    message(STATUS "libtiff не найден, выполняем сборку")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -S ${CMAKE_SOURCE_DIR}/external/libtiff -B ${EXTERNAL_BUILD_DIR}/libtiff_build -DCMAKE_INSTALL_PREFIX=${EXTERNAL_BUILD_DIR}/libtiff_install
        RESULT_VARIABLE LIBTIFF_CONFIGURE_RESULT
    )
    if(NOT LIBTIFF_CONFIGURE_RESULT EQUAL 0)
        message(FATAL_ERROR "Не удалось настроить libtiff")
    endif()
    execute_process(
        COMMAND ${CMAKE_COMMAND} --build ${EXTERNAL_BUILD_DIR}/libtiff_build --target install
        RESULT_VARIABLE LIBTIFF_BUILD_RESULT
    )
    if(NOT LIBTIFF_BUILD_RESULT EQUAL 0)
        message(FATAL_ERROR "Не удалось собрать или установить libtiff")
    endif()
endif()

set(OPENCV_FOUND FALSE)
foreach(LIB_PATH ${OPENCV_CORE_LIB})
    if(EXISTS ${LIB_PATH})
        set(OPENCV_FOUND TRUE)
        message(STATUS "Found OpenCV: ${LIB_PATH}")
        break()
    endif()
endforeach()
if(OPENCV_FOUND)
    message(STATUS "OpenCV уже установлен, пропускаем сборку")
else()
    message(STATUS "OpenCV не найден, выполняем сборку")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -S ${CMAKE_SOURCE_DIR}/external/opencv -B ${EXTERNAL_BUILD_DIR}/opencv_build -DCMAKE_INSTALL_PREFIX=${EXTERNAL_BUILD_DIR}/opencv_install
                -DWITH_CUDA=OFF
                -DBUILD_opencv_core=ON
                -DBUILD_opencv_imgproc=ON
                -DBUILD_opencv_imgcodecs=ON
                -DBUILD_opencv_highgui=ON
                -DBUILD_opencv_videoio=ON
                -DBUILD_LIST=core,imgproc,imgcodecs,highgui,videoio
                -DBUILD_EXAMPLES=OFF
                -DBUILD_TESTS=OFF
                -DBUILD_PERF_TESTS=OFF
                -DBUILD_DOCS=OFF
                -DBUILD_JAVA=OFF
                -DBUILD_PYTHON=OFF
        RESULT_VARIABLE OPENCV_CONFIGURE_RESULT
    )
    if(NOT OPENCV_CONFIGURE_RESULT EQUAL 0)
        message(FATAL_ERROR "Не удалось настроить OpenCV")
    endif()
    execute_process(
        COMMAND ${CMAKE_COMMAND} --build ${EXTERNAL_BUILD_DIR}/opencv_build --target install
        RESULT_VARIABLE OPENCV_BUILD_RESULT
    )
    if(NOT OPENCV_BUILD_RESULT EQUAL 0)
        message(FATAL_ERROR "Не удалось собрать или установить OpenCV")
    endif()
endif()

list(APPEND CMAKE_PREFIX_PATH ${EXTERNAL_BUILD_DIR}/libtiff_install ${EXTERNAL_BUILD_DIR}/opencv_install)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MDd /W2 /EHsc")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/W2,/EHsc -Wno-deprecated-gpu-targets")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -Wextra -pedantic")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -Wall -Wno-deprecated-gpu-targets")
endif()

set(SOURCE_DIR "${CMAKE_SOURCE_DIR}/src")
set(INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")
set(TESTS_DIR "${CMAKE_SOURCE_DIR}/tests")
set(SOURCE_GUI_DIR "${SOURCE_DIR}/gui")
set(INCLUDE_GUI_DIR "${INCLUDE_DIR}/gui")
set(MAINS_DIR "${SOURCE_DIR}/mains")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CUDA_VERSION "12.8" CACHE STRING "CUDA Toolkit version (e.g., 10.2, 11.4, 12.8, or AUTO)")
if(CUDA_VERSION STREQUAL "10.2")
    set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2")
    set(CMAKE_CUDA_ARCHITECTURES 30 50 52 60 61 70 72 75)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
elseif(CUDA_VERSION STREQUAL "11.4")
    set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4")
    set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 72 75 80 86)
elseif(CUDA_VERSION STREQUAL "12.8")
    if(WIN32)
        set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8")
    else()
        set(CUDAToolkit_ROOT "/usr/local/cuda-12.8")
    endif()
    set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 72 75 80 86 89 90)
else()
    # Автоматический выбор
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "11.0")
        set(CMAKE_CUDA_ARCHITECTURES 30 50 52 60 61 70 72 75)
    else()
        set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 72 75 80 86 89 90)
    endif()
endif()

set(CUDA_ARCH "AUTO" CACHE STRING "CUDA architecture (e.g., 30, 50, 75, or AUTO for default)")
if(NOT CUDA_ARCH STREQUAL "AUTO")
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})
endif()

add_definitions(-DPROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
find_package(TIFF REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui imgcodecs)
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

    target_link_libraries(gui PRIVATE TIFF::TIFF ${CUDA_LIBRARIES} Qt6::Core Qt6::Gui Qt6::Widgets ${OpenCV_LIBS})
    target_include_directories(gui PRIVATE ${INCLUDE_GUI_DIR})
    if(CMAKE_AUTOUIC)
        target_include_directories(gui PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    endif()
    set_target_properties(gui PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_definitions(gui PRIVATE USE_QT)
else()
    message(WARNING "Qt6 not found, GUI target will not be built")
endif()

if(NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "CUDA library not found")
endif()

add_subdirectory(external/googletest)

include_directories(${SOURCE_DIR})
include_directories(${INCLUDE_DIR})
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${TIFF_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB SOURCES "${SOURCE_DIR}/*.cpp" "${SOURCE_DIR}/*.cu" "${MAINS_DIR}/main.cpp")
file(GLOB SPEEDTEST_SOURCES "${SOURCE_DIR}/*.cpp" "${SOURCE_DIR}/*.cu" "${MAINS_DIR}/speedtest.cpp")
file(GLOB TEST_SOURCES "${TESTS_DIR}/*.cpp" "${SOURCE_DIR}/*.cpp" "${SOURCE_DIR}/*.cu")

add_executable(${PROJECT_NAME} ${SOURCES})
add_executable(speedtest ${SPEEDTEST_SOURCES})
add_executable(unit_tests ${TEST_SOURCES})

if(NOT MSVC)
    if(CMAKE_BUILD_TYPE MATCHES Debug)
        target_compile_options(unit_tests PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:--coverage;-fprofile-arcs;-ftest-coverage>
            $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info;-Xcompiler;--coverage;-Xcompiler;-fprofile-arcs;-Xcompiler;-ftest-coverage>
        )
        target_link_options(unit_tests PRIVATE --coverage)
    endif()
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} TIFF::TIFF ${CUDA_LIBRARIES} cudart)
target_link_libraries(speedtest PRIVATE ${OpenCV_LIBS} TIFF::TIFF ${CUDA_LIBRARIES} cudart)
target_link_libraries(unit_tests PRIVATE ${OpenCV_LIBS} TIFF::TIFF gtest gtest_main)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(speedtest PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(unit_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if(DEFINED ENV{GITHUB_ACTIONS})
    set(EXCLUDE_CUDA "*/src/*.cu")
else()
    set(EXCLUDE_CUDA "")
endif()

if(WIN32)
    message(WARNING "Code coverage is not supported on Windows yet")
else()
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
endif()

message(STATUS "CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
message(STATUS "NVCC Version: ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")