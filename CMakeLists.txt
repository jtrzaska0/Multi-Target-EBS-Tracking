cmake_minimum_required(VERSION 3.24)
project(LiveTracking)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -DLITTLE_ENDIAN")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ptu-sdk)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ptu-sdk/cerial)

find_library(M_LIB m REQUIRED)
find_library(CPI_LIB cpi REQUIRED PATHS ${CMAKE_CURRENT_SOURCE_DIR}/ptu-sdk)

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

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)
if(OpenCV_VERSION VERSION_LESS "3.1.0")
    message(FATAL_ERROR "Cannot find OpenCV 3.1.0 or newer.")
endif()

find_package(PkgConfig)
pkg_check_modules(KSSP REQUIRED kssp)

add_executable(LiveTracking main.cpp ptu-sdk/examples/estrap.o)
target_link_libraries(LiveTracking PUBLIC
        caer
        ${CPI_LIB}
        ${M_LIB}
        ${nlohmann_json_LIBRARIES}
        ${Eigen3_LIBRARIES}
        ${Boost_LIBRARIES}
        ${fmt_LIBRARIES}
        ${OpenCV_LIBS}
        ${X11_LIBRARIES}
        ${KSSP_LIBRARIES})
message(STATUS ${CPI_LIB})
target_compile_options(LiveTracking PUBLIC ${KSSP_CFLAGS_OTHER})
