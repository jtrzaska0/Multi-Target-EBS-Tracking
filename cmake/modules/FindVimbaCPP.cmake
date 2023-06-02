# Look for VimbaCPP library
file(GLOB VimbaCPP_LIBRARY "/opt/Vimba_6_0/VimbaCPP/DynamicLib/x86_64bit/*.so")

# Check if VimbaCPP library is found
if(NOT VimbaCPP_LIBRARY)
    message(FATAL_ERROR "Could not find VimbaCPP library")
else()
    message(STATUS "Found VimbaCPP library at ${VimbaCPP_LIBRARY}")
endif()

# Set VimbaCPP include directories
set(VimbaCPP_INCLUDE_DIRS /opt/Vimba_6_0/)

# Export VimbaCPP library and include directories
set(VimbaCPP_LIBRARIES ${VimbaCPP_LIBRARY})
set(VimbaCPP_INCLUDE_DIR ${VimbaCPP_INCLUDE_DIRS})
mark_as_advanced(VimbaCPP_INCLUDE_DIR VimbaCPP_LIBRARIES)