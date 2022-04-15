project(epipolar)
cmake_minimum_required(VERSION 2.8)

#-------------------------------------------------------------------
# Compiler Options
#-------------------------------------------------------------------

# 17 necessary to use std::file_system
add_definitions("-Wall -std=c++17 -O3")

add_definitions(-D_USE_MATH_DEFINES)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")


#-------------------------------------------------------------------
# Dependencies
#-------------------------------------------------------------------

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)

include_directories(
        ./include/
        ${OpenCV_INCLUDE_DIR}
        ${EIGEN3_INCLUDE_DIR}
        ${PROJECT_SOURCE_DIR}
        ${YAML_CPP_INCLUDE_DIR}
)

## Declare a C++ library
add_library (Frame src/Frame.cpp)
target_link_libraries(Frame ${OpenCV_LIBS} ${EIGEN3_LIBS} ${YAML_CPP_LIBRARIES})

#-------------------------------------------------------------------
# Executables generation
#-------------------------------------------------------------------

add_executable(main src/main.cpp)
target_link_libraries(main ${OpenCV_LIBS} ${EIGEN3_LIBS} ${YAML_CPP_LIBRARIES} Frame)

add_executable(main_sequence src/main_sequence.cpp)
target_link_libraries(main_sequence ${OpenCV_LIBS} ${EIGEN3_LIBS} ${YAML_CPP_LIBRARIES} Frame)

add_executable(main_detect_grid src/main_detect_grid.cpp)
target_link_libraries(main_detect_grid ${OpenCV_LIBS} ${EIGEN3_LIBS} ${YAML_CPP_LIBRARIES})
