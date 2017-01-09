cmake_minimum_required(VERSION 2.8)
project(cvHelper)
find_package(OpenCV REQUIRED)
link_directories(${MYPROJECT_DIR})

include_directories(${MYPROJECT_DIR})

set(SOURCES main.cpp
            src/bold.cpp
            src/helper.cpp
            src/utils.cpp)


set(HEADER_FILES include/bold.hpp
                 include/helper.h
                 include/utils.h)

set (CMAKE_CXX_FLAGS "-O2 -march=native")

add_executable(cvHelper ${SOURCES} ${HEADER_FILES})
target_link_libraries( cvHelper ${OpenCV_LIBS})
