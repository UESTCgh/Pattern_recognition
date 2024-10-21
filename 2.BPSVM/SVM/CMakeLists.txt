cmake_minimum_required(VERSION 3.5)
project(SVM)

set(CMAKE_CXX_STANDARD 20)

add_executable(SVM main.cpp
        bp.h
        bp.cpp)
