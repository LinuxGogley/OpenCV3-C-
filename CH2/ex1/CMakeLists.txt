cmake_minimum_required (VERSION 2.6)
cmake_policy(SET CMP0012 NEW)
PROJECT(Chapter2)
# Requires OpenCV
FIND_PACKAGE( OpenCV 3.0.0 REQUIRED )
# Show a message with the opencv version detected
MESSAGE("OpenCV version : ${OpenCV_VERSION}")
include_directories(${OpenCV_LIB_DIR})
# Create a variable called SRC
SET(SRC main.cpp )
# Create our executable
ADD_EXECUTABLE( ${PROJECT_NAME} ${SRC} )
# Link our library
TARGET_LINK_LIBRARIES( ${PROJECT_NAME} ${OpenCV_LIBS} )
