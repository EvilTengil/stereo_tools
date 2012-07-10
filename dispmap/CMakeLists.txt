project(testprog CXX C)
cmake_minimum_required(VERSION 2.8)
# Set default built type
set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING
    "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel.")

# The CDT Error Parser cannot handle error messages that span more than one line, which is the default gcc behavior. 
# This forces gcc to generate single line error messages with no line wrapping.  
IF(CMAKE_COMPILER_IS_GNUCC)
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fmessage-length=0")
ENDIF(CMAKE_COMPILER_IS_GNUCC)
IF(CMAKE_COMPILER_IS_GNUCXX)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fmessage-length=0")
ENDIF(CMAKE_COMPILER_IS_GNUCXX)

# This enables Eclipse to discover the include dirs automatically
SET(CMAKE_VERBOSE_MAKEFILE ON)

# Settings
if(CMAKE_COMPILER_IS_GNUCXX)
    # Add support for C++11/0x
    add_definitions(-std=gnu++0x)
    
    add_definitions(-g -pthread -O3 -Wall) # -march=corei7 -mtune=corei7 -msse -msse2 -msse3 -msse4 -msse4.2)
endif()

# Libraries
set(LIBS
	opencv_core
	opencv_imgproc
	opencv_highgui
	opencv_ml
	opencv_video
	opencv_features2d
	opencv_calib3d
	opencv_objdetect
	opencv_contrib
	opencv_legacy
	opencv_flann
	GLU
	GL
	GLEW
	glut
	lapack
)

set(SRC
	dispmap.cpp
)

add_executable(testprog ${SRC})
target_link_libraries(testprog ${LIBS})


