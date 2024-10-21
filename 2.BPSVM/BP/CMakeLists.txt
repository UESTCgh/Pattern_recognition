cmake_minimum_required(VERSION 3.5)
project(BP)

set(CMAKE_CXX_STANDARD 20)

add_executable(BP main.cpp
        bp.h
        bp.cpp)
