cmake_minimum_required(VERSION 3.16)
project(ann_hybrid LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math -mfma")

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(hybrid_all hybrid_all.cu)
target_include_directories(hybrid_all PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(hybrid_all PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)

# MPI+OpenMP 双层
add_executable(hybrid_mpi_omp hybrid_all.cu)
target_compile_definitions(hybrid_mpi_omp PRIVATE DISABLE_CUDA)
target_include_directories(hybrid_mpi_omp PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(hybrid_mpi_omp PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)
