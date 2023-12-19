cmake_minimum_required(VERSION 3.0.0)
project(maceLammpsPlugin VERSION 0.1.0)

# use c++ 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set(Torch_DIR ${Torch_DIR}/share/cmake/Torch)

if(LAMMPS_BINARY_ROOT AND LAMMPS_SOURCE_DIR)
  message(STATUS "LAMMPS_BINARY_ROOT is set to ${LAMMPS_BINARY_ROOT}")
  set(LAMMPS_LIBRARY_DIR ${LAMMPS_BINARY_ROOT}/lib)
  set(LAMMPS_INCLUDE_DIRS ${LAMMPS_SOURCE_DIR})
else()
  if (LAMMPS_BINARY_ROOT)
    message(FATAL_ERROR "Lammps source root is not set, please set LAMMPS_SOURCE_DIR to the root of lammps source")
  endif()
  if (LAMMPS_SOURCE_DIR)
    message(FATAL_ERROR "Lammps binary root is not set, please set LAMMPS_BINARY_ROOT to the root of lammps binary")
  endif()
  message(FATAL_ERROR "Lammps source root and binary root are not set, please set LAMMPS_SOURCE_DIR and LAMMPS_BINARY_ROOT")
endif()

find_package(Torch REQUIRED)
message(STATUS "Torch_DIR is set to ${Torch_DIR}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
if(TORCH_FOUND)
  if(CUDA_FOUND)
    message(STATUS "CUDA is found, use CUDA features of Torch")
  else()
    message(STATUS "CUDA is not found, use CPU features of Torch")
  endif()
  message(STATUS "Torch include: ${TORCH_INCLUDE_DIRS}")
endif(TORCH_FOUND)

message(STATUS "Torch version: ${Torch_VERSION}")

add_subdirectory(src)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
