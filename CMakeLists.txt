cmake_minimum_required(VERSION 3.24)
project(LiveTracking)

set(CMAKE_CXX_STANDARD 20)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

FIND_PACKAGE(OpenCV REQUIRED COMPONENTS core imgproc highgui)
IF(OpenCV_VERSION VERSION_LESS "3.1.0")
    MESSAGE(FATAL_ERROR "Cannot find OpenCV 3.1.0 or newer.")
ENDIF()

add_executable(LiveTracking main.cpp)

target_link_libraries(LiveTracking PRIVATE caer ${CONAN_LIBS} ${OpenCV_LIBS})
