cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(d2l-pytorch VERSION 1.0.0 LANGUAGES CXX)



if(NOT Torch_FOUND)
    find_package(Torch REQUIRED PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../../../../pytorch")
endif()
find_package(OpenCV REQUIRED )
set(EXECUTABLE_NAME naivebayes)

add_executable(${EXECUTABLE_NAME})
target_sources(${EXECUTABLE_NAME} PRIVATE NaiveBayes.cpp)

target_link_libraries(${EXECUTABLE_NAME} ${TORCH_LIBRARIES} )
target_link_libraries(${EXECUTABLE_NAME} ${OpenCV_LIBS} )
set_target_properties(${EXECUTABLE_NAME} PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

add_custom_command(TARGET ${EXECUTABLE_NAME}
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E make_directory output)

