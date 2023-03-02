cmake_minimum_required(VERSION 3.24)
project(LiveTracking)

set(CMAKE_CXX_STANDARD 20)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -Dmalloc_check_=2")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

find_package(X11 REQUIRED)
include_directories(${X11_INCLUDE_DIR})
link_directories(${X11_LIBRARIES})

FIND_PACKAGE(OpenCV REQUIRED COMPONENTS core imgproc highgui)
IF(OpenCV_VERSION VERSION_LESS "3.1.0")
    MESSAGE(FATAL_ERROR "Cannot find OpenCV 3.1.0 or newer.")
ENDIF()

find_package(PkgConfig)
pkg_check_modules (KESSLER REQUIRED kessler)

add_executable(LiveTracking main.cpp)
target_link_libraries(LiveTracking PRIVATE caer ${CONAN_LIBS} ${OpenCV_LIBS} ${X11_LIBRARIES} ${KESSLER_LIBRARIES})
target_include_directories(LiveTracking PUBLIC ${KESSLER_INCLUDE_DIRS})
message(STATUS ${KESSLER_INCLUDE_DIRS})
target_compile_options(LiveTracking PUBLIC ${KESSLER_CFLAGS_OTHER})
