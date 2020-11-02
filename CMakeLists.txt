cmake_minimum_required(VERSION 2.8)

project(arm2d)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

include_directories(include)

add_executable(${PROJECT_NAME} src/polygon.cpp src/arm2d.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE include)

