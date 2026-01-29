cmake_minimum_required(VERSION 3.10)
project(ceres_example)

set(CMAKE_CXX_STANDARD 14)

# Ceres 찾기
find_package(Ceres REQUIRED)

# 실행 파일
add_executable(curve_fitting curve_fitting.cpp)
target_link_libraries(curve_fitting Ceres::ceres)
