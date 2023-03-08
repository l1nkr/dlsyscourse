cmake_minimum_required(VERSION 3.10)

project(simple_ml_ext CXX)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)

set(pybind11_DIR /Users/fl/opt/anaconda3/envs/pyg/lib/python3.8/site-packages/pybind11/share/cmake/pybind11/)
find_package(pybind11 REQUIRED)

pybind11_add_module(simple_ml_ext src/simple_ml_ext.cpp)
#add_executable(simple_ml src/simple_ml_ext.cpp)