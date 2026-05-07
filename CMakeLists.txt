cmake_minimum_required(VERSION 3.20)
project(nn-training)

include(CTest)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

find_package(OpenMP COMPONENTS CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_PREFIX_PATH "/usr/local/libtorch")

option(ENABLE_CPPCHECK "Run cppcheck as part of C++ builds" OFF)
option(ENABLE_CLANG_TIDY "Run clang-tidy as part of C++ builds" OFF)
option(ENABLE_COVERAGE "Build with coverage instrumentation and add a coverage target" OFF)

function(network_enable_warnings target)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wshadow
            -Wconversion
            -Wsign-conversion
            -Wold-style-cast
            -Wnon-virtual-dtor
        )
    endif()
endfunction()

find_program(CPPCHECK_EXECUTABLE NAMES cppcheck)
set(CPPCHECK_ARGS
    --enable=warning,style,performance,portability
    --inline-suppr
    --language=c++
    --std=c++17
    --suppress=missingIncludeSystem
)

if(ENABLE_CPPCHECK)
    if(NOT CPPCHECK_EXECUTABLE)
        message(FATAL_ERROR "ENABLE_CPPCHECK is ON, but cppcheck was not found")
    endif()

    set(CMAKE_CXX_CPPCHECK
        ${CPPCHECK_EXECUTABLE}
        ${CPPCHECK_ARGS}
    )
endif()

find_program(CLANG_TIDY_EXECUTABLE NAMES clang-tidy-20 clang-tidy)
set(CLANG_TIDY_ARGS
    --extra-arg=-std=c++17
)

if(ENABLE_CLANG_TIDY)
    if(NOT CLANG_TIDY_EXECUTABLE)
        message(FATAL_ERROR "ENABLE_CLANG_TIDY is ON, but clang-tidy was not found")
    endif()

    set(CMAKE_CXX_CLANG_TIDY
        ${CLANG_TIDY_EXECUTABLE}
        ${CLANG_TIDY_ARGS}
    )
endif()

if(ENABLE_COVERAGE)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        message(FATAL_ERROR "ENABLE_COVERAGE requires GCC or Clang")
    endif()

    find_program(GCOVR_EXECUTABLE NAMES gcovr)
    if(NOT GCOVR_EXECUTABLE)
        message(FATAL_ERROR "ENABLE_COVERAGE is ON, but gcovr was not found")
    endif()

    set(GCOVR_GCOV_EXECUTABLE gcov)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        find_program(LLVM_COV_EXECUTABLE NAMES llvm-cov-20 llvm-cov)
        if(NOT LLVM_COV_EXECUTABLE)
            message(FATAL_ERROR "ENABLE_COVERAGE with Clang requires llvm-cov")
        endif()
        set(GCOVR_GCOV_EXECUTABLE "${LLVM_COV_EXECUTABLE} gcov")
    endif()

    add_compile_options(--coverage -O0 -g)
    add_link_options(--coverage)
endif()

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

add_subdirectory(libs/nn-runtime)

add_executable(${PROJECT_NAME}
    main.cpp
    src/hopfield.cpp
    src/hopfield.h
)

network_enable_warnings(${PROJECT_NAME})

if(CPPCHECK_EXECUTABLE)
    add_custom_target(${PROJECT_NAME}-cppcheck
        COMMAND ${CPPCHECK_EXECUTABLE}
            ${CPPCHECK_ARGS}
            ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/hopfield.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/hopfield.h
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Running cppcheck"
        VERBATIM
    )
endif()

target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE nn-runtime ${TORCH_LIBRARIES} ${OpenCV_LIBS})

if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

if(BUILD_TESTING)
    set(TEST_EXECUTABLE "${PROJECT_NAME}-tests")

    add_executable(${TEST_EXECUTABLE}
        tests/SmokeTests.cpp
    )

    network_enable_warnings(${TEST_EXECUTABLE})

    add_test(NAME smoke COMMAND ${TEST_EXECUTABLE})

    if(ENABLE_COVERAGE)
        add_custom_target(coverage
            COMMAND ${CMAKE_CTEST_COMMAND}
                --test-dir ${CMAKE_BINARY_DIR}
                --output-on-failure
            COMMAND ${GCOVR_EXECUTABLE}
                --root ${CMAKE_SOURCE_DIR}
                --gcov-executable ${GCOVR_GCOV_EXECUTABLE}
                --exclude ${CMAKE_SOURCE_DIR}/build.*
                --exclude ${CMAKE_SOURCE_DIR}/CMakeFiles
                --html-details ${CMAKE_BINARY_DIR}/coverage.html
                --print-summary
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            DEPENDS ${TEST_EXECUTABLE}
            COMMENT "Running tests and generating coverage report"
            VERBATIM
        )
    endif()
endif()
