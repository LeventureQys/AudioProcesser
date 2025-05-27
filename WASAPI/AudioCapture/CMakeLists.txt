cmake_minimum_required(VERSION 3.10)

# 项目名称和版本
project(AudioCapture VERSION 1.0)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 添加可执行文件
add_executable(AudioCapture main.cpp ./include/AudioCapture.h ./src/AudioCapture.cpp)

# 直接链接 Windows 系统库（无需 find_library）
target_link_libraries(AudioCapture PRIVATE ole32 ksuser)