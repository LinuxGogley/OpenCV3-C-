cmake_minimum_required (VERSION 2.6)
project (Chapter2)

# OpenCV Package required
FIND_PACKAGE( OpenCV 3.0.0 REQUIRED)

#Add opencv header files to project
include_directories( ${OpenCV_INCLUDE_DIR} )
link_directories(${OpenCV_LIB_DIR})

add_subdirectory(utils)

# Add optional log with a precompiler definition
option(WITH_LOG "Build with outuput logs and images in tmp" OFF)
if(WITH_LOG)
    add_definitions(-DLOG)
endif(WITH_LOG)

# generate our new executable
add_executable( ${PROJECT_NAME} main.cpp)
# link the project with its dependencies
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} Utils)
