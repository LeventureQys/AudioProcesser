cmake_minimum_required(VERSION 3.5.0)
project(Audio_Processing VERSION 0.1.0 LANGUAGES C CXX)
# 收集 AudioProcessing 目录下的所有 .cpp 文件

include_directories(AudioProcessing) 
# 递归查找 AudioProcessing 目录及其子目录下的所有 .cc 文件
file(GLOB_RECURSE SOURCES AudioProcessing/*.cc AudioProcessing/*.c)

# 打印找到的源文件列表（可选，用于调试）
message(STATUS "Found source files:")
foreach(SOURCE ${SOURCES})
    message(STATUS "  ${SOURCE}")
endforeach()

message("FILES: ${SOURCES}") # 打印 FILES 变量
add_executable(Audio_Processing 
    AudioProcessing_example.cpp 
    ${SOURCES}
)

