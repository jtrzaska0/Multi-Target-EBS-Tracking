cmake_minimum_required(VERSION 3.24)
project(LiveTracking)

set(CMAKE_CXX_STANDARD 20)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -Dmalloc_check_=2")

find_package(X11 REQUIRED)
include_directories(${X11_INCLUDE_DIR})
link_directories(${X11_LIBRARIES})

find_package(fmt 9.1.0 REQUIRED)
include_directories(${fmt_INCLUDE_DIR})
link_directories(${fmt_LIBRARIES})

find_package(Boost 1.81.0 REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARIES})

find_package(Eigen3 3.4.0 REQUIRED)
include_directories(${Eigen3_INCLUDE_DIR})
link_directories(${Eigen3_LIBRARIES})

find_package(nlohmann_json 3.11.2 REQUIRED)
include_directories(${nlohmann_json_INCLUDE_DIR})
link_directories(${nlohmann_json_LIBRARIES})

FIND_PACKAGE(OpenCV REQUIRED COMPONENTS core imgproc highgui)
IF (OpenCV_VERSION VERSION_LESS "3.1.0")
    MESSAGE(FATAL_ERROR "Cannot find OpenCV 3.1.0 or newer.")
ENDIF ()

find_package(PkgConfig)
pkg_check_modules(KSSP REQUIRED kssp)

add_executable(LiveTracking main.cpp)
target_link_libraries(LiveTracking PUBLIC caer ${nlohmann_json_LIBRARIES} ${Eigen3_LIBRARIES} ${Boost_LIBRARIES} ${fmt_LIBRARIES} ${OpenCV_LIBS} ${X11_LIBRARIES} ${KSSP_LIBRARIES})
target_include_directories(LiveTracking PUBLIC ${KSSP_INCLUDE_DIRS})
message(STATUS ${KSSP_INCLUDE_DIRS})
target_compile_options(LiveTracking PUBLIC ${KSSP_CFLAGS_OTHER})
