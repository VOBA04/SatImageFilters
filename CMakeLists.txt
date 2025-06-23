cmake_minimum_required(VERSION 3.20)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(SatImageFilters LANGUAGES CXX CUDA)

set(EXTERNAL_BUILD_DIR "${CMAKE_SOURCE_DIR}/external_build")
file(MAKE_DIRECTORY ${EXTERNAL_BUILD_DIR})

list(APPEND CMAKE_PREFIX_PATH 
    ${EXTERNAL_BUILD_DIR}/gtest_install
    ${EXTERNAL_BUILD_DIR}/libtiff_install 
    ${EXTERNAL_BUILD_DIR}/opencv_install
)

find_package(GTest QUIET)
if(GTest_FOUND)
    message(STATUS "Найден Google Test: ${GTEST_LIBRARIES}")
else()
    message(STATUS "Google Test не найден, выполняем сборку")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -S ${CMAKE_SOURCE_DIR}/external/googletest -B ${EXTERNAL_BUILD_DIR}/gtest_build
                -DCMAKE_INSTALL_PREFIX=${EXTERNAL_BUILD_DIR}/gtest_install
                -DBUILD_GMOCK=OFF
                -DINSTALL_GTEST=ON
        RESULT_VARIABLE GTEST_CONFIGURE_RESULT
    )
    if(NOT GTEST_CONFIGURE_RESULT EQUAL 0)
        message(FATAL_ERROR "Не удалось настроить Google Test")
    endif()
    execute_process(
        COMMAND ${CMAKE_COMMAND} --build ${EXTERNAL_BUILD_DIR}/gtest_build --target install
        RESULT_VARIABLE GTEST_BUILD_RESULT
    )
    if(NOT GTEST_BUILD_RESULT EQUAL 0)
        message(FATAL_ERROR "Не удалось собрать или установить Google Test")
    endif()
    find_package(GTest REQUIRED)
endif()

find_package(TIFF QUIET)
if(TIFF_FOUND)
    message(STATUS "Найден libtiff: ${TIFF_LIBRARIES}")
else()
    message(STATUS "libtiff не найден, выполняем сборку")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -S ${CMAKE_SOURCE_DIR}/external/libtiff -B ${EXTERNAL_BUILD_DIR}/libtiff_build
                -DCMAKE_INSTALL_PREFIX=${EXTERNAL_BUILD_DIR}/libtiff_install
                -DBUILD_SHARED_LIBS=ON
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
    find_package(TIFF REQUIRED)
endif()

find_package(OpenCV QUIET COMPONENTS core imgproc highgui imgcodecs)
if(OpenCV_FOUND)
    message(STATUS "Найден OpenCV: ${OpenCV_LIBS}")
else()
    message(STATUS "OpenCV не найден, выполняем сборку")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -S ${CMAKE_SOURCE_DIR}/external/opencv -B ${EXTERNAL_BUILD_DIR}/opencv_build
                -DCMAKE_INSTALL_PREFIX=${EXTERNAL_BUILD_DIR}/opencv_install
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
    find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui imgcodecs)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MDd /W2 /EHsc /Zc:__cplusplus")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/W2,/EHsc,/Zc:__cplusplus -Wno-deprecated-gpu-targets")
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

set(CUDA_VERSION "AUTO" CACHE STRING "CUDA Toolkit version (e.g., 10.2, 11.4, 12.8, or AUTO for automatic detection)")
if(CUDA_VERSION STREQUAL "AUTO")
    find_package(CUDAToolkit REQUIRED)
else()
    find_package(CUDAToolkit ${CUDA_VERSION} EXACT REQUIRED)
endif()

if(NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "CUDA Toolkit не найден")
endif()

if(CUDAToolkit_FOUND)
    if(CUDAToolkit_VERSION VERSION_LESS "11.0")
        set(CMAKE_CUDA_ARCHITECTURES 30 50 52 60 61 70 72 75)
        set(CMAKE_CUDA_STANDARD 14)
        message(STATUS "CUDA Toolkit < 11.0: Using architectures 3.0, 5.0, 5.2, 6.0, 6.1, 7.0, 7.2, 7.5")
    elseif(CUDAToolkit_VERSION VERSION_LESS "11.8")
        set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 72 75 80)
        set(CMAKE_CUDA_STANDARD 17)
        message(STATUS "CUDA Toolkit >= 11.0 and < 11.8: Using architectures 5.0, 5.2, 6.0, 6.1, 7.0, 7.2, 7.5, 8.0")
    else()
        set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 72 75 80 86 89 90)
        set(CMAKE_CUDA_STANDARD 17)
        message(STATUS "CUDA Toolkit >= 11.8: Using architectures 5.0, 5.2, 6.0, 6.1, 7.0, 7.2, 7.5, 8.0, 8.6, 8.9, 9.0")
    endif()
endif()

set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)



set(CUDA_ARCH "AUTO" CACHE STRING "CUDA architecture (e.g., 30, 50, 75, or AUTO for default)")
if(NOT CUDA_ARCH STREQUAL "AUTO")
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})
endif()

add_definitions(-DPROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

enable_language(CUDA)
find_package(Qt6 COMPONENTS Core Gui Widgets)

if(Qt6_FOUND)
    message(STATUS "Qt6 found, GUI target will be built")
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTORCC ON)
    set(CMAKE_AUTOUIC ON)

    file(GLOB GUI_SOURCES "${SOURCE_DIR}/*.cpp" "${SOURCE_DIR}/*.cu" "${SOURCE_GUI_DIR}/*.cpp" "${SOURCE_GUI_DIR}/*.ui" "${MAINS_DIR}/gui.cpp")
    file(GLOB GUI_INCLUDE "${INCLUDE_GUI_DIR}/*.h")

    qt6_wrap_cpp(GUI_MOC_SOURCES ${GUI_INCLUDE})
    add_executable(gui ${GUI_SOURCES} ${GUI_MOC_SOURCES})

    target_link_libraries(gui PRIVATE TIFF::TIFF CUDA::cudart CUDA::cuda_driver Qt6::Core Qt6::Gui Qt6::Widgets ${OpenCV_LIBS})
    target_include_directories(gui PRIVATE ${INCLUDE_GUI_DIR} ${CUDAToolkit_INCLUDE_DIRS})
    if(CMAKE_AUTOUIC)
        target_include_directories(gui PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    endif()
    set_target_properties(gui PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_definitions(gui PRIVATE USE_QT)
else()
    message(WARNING "Qt6 not found, GUI target will not be built")
endif()

include_directories(${SOURCE_DIR})
include_directories(${INCLUDE_DIR})
include_directories(${CUDAToolkit_INCLUDE_DIRS})
include_directories(${TIFF_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB SOURCES "${SOURCE_DIR}/*.cpp" "${SOURCE_DIR}/*.cu")
add_library(common STATIC ${SOURCES})
target_include_directories(common PUBLIC
    ${SOURCE_DIR}
    ${INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TIFF_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(common PUBLIC
    TIFF::TIFF 
    CUDA::cudart 
    CUDA::cuda_driver
)
set_target_properties(common 
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

set(TEST_SOURCES "${TESTS_DIR}/test_kernel.cpp")
if(CMAKE_CUDA_STANDARD GREATER_EQUAL 17)
    list(APPEND TEST_SOURCES "${TESTS_DIR}/test_tiff_image.cpp")
else()
    list(APPEND TEST_SOURCES "${TESTS_DIR}/test_tiff_image_old.cpp")
endif()

add_executable(${PROJECT_NAME} "${MAINS_DIR}/main.cpp")
add_executable(speedtest "${MAINS_DIR}/speedtest.cpp")
add_executable(unit_tests ${TEST_SOURCES})
add_executable(benchmark_gpu "${MAINS_DIR}/benchmark_gpu.cpp")

target_link_libraries(${PROJECT_NAME} PRIVATE common)
target_link_libraries(speedtest PRIVATE common)
target_link_libraries(unit_tests PRIVATE ${OpenCV_LIBS} GTest::gtest GTest::gtest_main common)
target_link_libraries(benchmark_gpu PRIVATE common)

foreach(target ${PROJECT_NAME} speedtest unit_tests benchmark_gpu)
    if(WIN32)
         if(EXISTS "${EXTERNAL_BUILD_DIR}/libtiff_install/bin")
            file(GLOB TIFF_DLLS "${EXTERNAL_BUILD_DIR}/libtiff_install/bin/*.dll")
            foreach(dll ${TIFF_DLLS})
                add_custom_command(TARGET ${target} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${dll}"
                        $<TARGET_FILE_DIR:${target}>
                    COMMENT "Копирование TIFF DLL (${dll}) для ${target}"
                )
            endforeach()
        endif()
    endif()
endforeach()
if(WIN32 AND EXISTS "${EXTERNAL_BUILD_DIR}/opencv_install/x64/vc16/bin")
    file(GLOB OPENCV_DLLS "${EXTERNAL_BUILD_DIR}/opencv_install/x64/vc16/bin/*.dll")
    foreach(dll ${OPENCV_DLLS})
        add_custom_command(TARGET unit_tests POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${dll}"
                $<TARGET_FILE_DIR:unit_tests>
            COMMENT "Копирование OpenCV DLL (${dll}) для unit_tests"
        )
    endforeach()
endif()

if(NOT MSVC)
    if(CMAKE_BUILD_TYPE MATCHES Debug)
        target_compile_options(unit_tests PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:--coverage;-fprofile-arcs;-ftest-coverage>
            $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info;-Xcompiler;--coverage;-Xcompiler;-fprofile-arcs;-Xcompiler;-ftest-coverage>
        )
        target_link_options(unit_tests PRIVATE --coverage)
    endif()
endif()

add_test(NAME unit_tests COMMAND unit_tests)

if(DEFINED ENV{GITHUB_ACTIONS})
    set(EXCLUDE_CUDA "*/src/*.cu")
else()
    set(EXCLUDE_CUDA "")
endif()

if(NOT WIN32)
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

message(STATUS "CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "CUDA Toolkit версия: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Toolkit include: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "CUDA Toolkit библиотеки: ${CUDAToolkit_LIBRARY_DIR}")
message(STATUS "NVCC Version: ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
