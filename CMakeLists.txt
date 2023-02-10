cmake_minimum_required(VERSION 3.24)
project(LiveTracking)

set(CMAKE_CXX_STANDARD 20)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(LiveTracking main.cpp)

target_link_libraries(LiveTracking ${CONAN_LIBS})
