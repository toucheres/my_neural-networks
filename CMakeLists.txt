cmake_minimum_required(VERSION 3.10.0)
project(my_neural_networks VERSION 0.1.0 LANGUAGES C CXX)
set(sorce main.cpp FCNN.cpp FCNN.h)
add_executable(my_neural_networks ${sorce})