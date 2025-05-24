cmake_minimum_required(VERSION 3.16)

project(AudioRecorder_Core LANGUAGES CXX)

# Qt 配置
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Qt6 REQUIRED COMPONENTS Core Multimedia)

# 关键修改：将头文件目录设为 PUBLIC 属性
add_library(AudioRecorder_Core SHARED
    AudioRecorder_core_global.h
    audiorecorder_core.cpp
    audiorecorder_core.h
    AudioRecorder_Windows.h
    AudioRecorder_Windows.cpp
    def.h
)

# 关键修改：PUBLIC 包含当前源文件目录
target_include_directories(AudioRecorder_Core
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>  # 构建时包含
        $<INSTALL_INTERFACE:include>                   # 安装时包含（可选）
)

target_link_libraries(AudioRecorder_Core 
    PRIVATE 
        Qt6::Core
        Qt6::Multimedia
    PUBLIC  # 如果需要传递Qt依赖（根据实际情况调整）
        # Qt6::Core 
)

target_compile_definitions(AudioRecorder_Core 
    PRIVATE 
        AUDIORECORDER_WINDOWS_LIBRARY
)