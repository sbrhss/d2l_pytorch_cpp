cmake_minimum_required(VERSION 3.00 FATAL_ERROR)

project(d2l-pytorch VERSION 1.0.0 LANGUAGES CXX)

if(NOT Torch_FOUND)
	#list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake")
    find_package(Torch REQUIRED PATHS "/content/pytorch/libtorch")
endif()

set(EXECUTABLE_NAME densenet)

add_executable(${EXECUTABLE_NAME})


target_sources(${EXECUTABLE_NAME} PRIVATE src/main.cpp
                                          src/tools.cpp
                                          src/tools.tpp
										                      include/tools.h         
										)
target_include_directories(${EXECUTABLE_NAME} PRIVATE include)
target_link_libraries(${EXECUTABLE_NAME} ${TORCH_LIBRARIES})


if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast")



set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

set_target_properties(${EXECUTABLE_NAME} PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

