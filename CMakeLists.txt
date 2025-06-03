cmake_minimum_required(VERSION 3.10)
project(hopfield)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_PREFIX_PATH "/usr/local/libtorch")

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(hopfield hopfield.cpp)

target_compile_options(hopfield PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wconversion
    -Wsign-conversion
    -Wold-style-cast
    -Wnon-virtual-dtor
)

target_include_directories(hopfield PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(hopfield PRIVATE ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET hopfield PROPERTY CXX_STANDARD 17)