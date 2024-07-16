cmake_minimum_required(VERSION 3.10)

# 设置项目名称和版本
project(DeepFilter VERSION 1.0)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 指定头文件目录
include_directories(${CMAKE_SOURCE_DIR}/include)

# 查找源文件
file(GLOB SOURCES "main.cpp")

# 添加可执行文件
add_executable(DeepFilter ${SOURCES})

# 设置运行时库的路径
set_target_properties(DeepFilter PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
