cmake_minimum_required(VERSION 3.16)
project(ann_mpi LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math")

find_package(MPI REQUIRED)

# 同步 SGD
add_executable(ann_mpi_sync ann_mpi_sync.cpp)
target_include_directories(ann_mpi_sync PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(ann_mpi_sync PRIVATE MPI::MPI_CXX)

# 异步 SGD
add_executable(ann_mpi_async ann_mpi_async.cpp)
target_include_directories(ann_mpi_async PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(ann_mpi_async PRIVATE MPI::MPI_CXX)
