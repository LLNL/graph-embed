
# Copyright 2019 Lawrence Livermore National Security. 
# Produced at the Lawrence Livermore National Laboratory.
# LLNL-CODE-781679. All Rights reserved. See file LICENSE for details.
#
# This file is part of graph-embed. For more information and source code
# availability, see github.com/LLNL/graph-embed
#
# SPDX-License-Identifier: LGPL-2.1


#!/bin/sh
cmake_minimum_required(VERSION 3.5)
project(partitioner VERSION 1.0 LANGUAGES CXX)
enable_testing()


include(FindOpenMP REQUIRED)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
  
include_directories(OpenMP)

find_package(linalgcpp REQUIRED
	     OpenMP::OpenMP_CXX REQUIRED
)

add_library(partitioner
	    src/partitioner.cpp
	    src/matrixutils.cpp
	    src/embed.cpp
	    src/export.cpp
)

set_target_properties(partitioner PROPERTIES
		      CXX_STANDARD 11
		      CXX_STANDARD_REQUIRED YES
		      CXX_EXTENSIONS NO
)

MESSAGE(STATUS ${CXX_COMPILER_ID})

target_compile_options(partitioner PUBLIC -Xpreprocessor -fopenmp -lomp)

target_link_libraries(partitioner
    PUBLIC
        linalgcpp::linalgcpp
    PRIVATE
)

target_include_directories(partitioner
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
	/usr/include
)


# Installation
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/libpartitioner)

install(TARGETS partitioner
    EXPORT partitioner-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Export targets to a script
install(EXPORT partitioner-targets
    FILE partitionerTargets.cmake
    NAMESPACE partitioner::
    DESTINATION ${INSTALL_CONFIGDIR}
)

# Create Config files
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/partitionerConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_LIST_DIR}/cmake/partitionerConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/partitionerConfig.cmake
    INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

configure_file(
  "${PROJECT_SOURCE_DIR}/config/partitioner_config.h.in"
  "${PROJECT_BINARY_DIR}/partitioner_config.h"
  )

# Install Config files
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/partitionerConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/partitionerConfigVersion.cmake
    DESTINATION ${INSTALL_CONFIGDIR}
)

# Export from the build tree
export(EXPORT partitioner-targets
       FILE ${CMAKE_CURRENT_BINARY_DIR}/partitionerTargets.cmake
       NAMESPACE partitioner::
       )

export(PACKAGE partitioner)

# Sub Directories
add_subdirectory(examples)
