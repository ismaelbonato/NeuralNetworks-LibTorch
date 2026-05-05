cmake_minimum_required(VERSION 3.20)
project(nn-training)

find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_PREFIX_PATH "/usr/local/libtorch")

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME} hopfield.cpp)

target_compile_options(${PROJECT_NAME} PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wconversion
    -Wsign-conversion
    -Wold-style-cast
    -Wnon-virtual-dtor
)

target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES} ${OpenCV_LIBS} OpenMP::OpenMP_CXX)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)