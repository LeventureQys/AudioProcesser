cmake_minimum_required(VERSION 3.5)

project(RealTimeDemo LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
# 添加头文件路径
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# QtCreator supports the following variables for Android, which are identical to qmake Android variables.
# Check http://doc.qt.io/qt-5/deployment-android.html for more information.
# They need to be set before the find_package(Qt5 ...) call.

if(ANDROID)
    set(ANDROID_PACKAGE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/android")
    if (ANDROID_ABI STREQUAL "armeabi-v7a")
        set(ANDROID_EXTRA_LIBS
            ${CMAKE_CURRENT_SOURCE_DIR}/path/to/libcrypto.so
            ${CMAKE_CURRENT_SOURCE_DIR}/path/to/libssl.so)
    endif()
endif()

find_package(Qt5 COMPONENTS Widgets Multimedia REQUIRED)
# 查找库文件
find_library(DEEPFILTERNET_LIB NAMES deepfilter PATHS ${CMAKE_CURRENT_SOURCE_DIR}/lib)
message(STATUS "DeepFilterNet library found at: ${DEEPFILTERNET_LIB}")
 #确保库文件被找到
if (NOT DEEPFILTERNET_LIB)
    message(FATAL_ERROR "DeepFilterNet library not found")
endif()
if(ANDROID)
  add_library(RealTimeDemo SHARED
    main.cpp
    mainwindow.cpp
    mainwindow.h
    mainwindow.ui
  )
else()
  add_executable(RealTimeDemo
    main.cpp
    mainwindow.cpp
    mainwindow.h
    mainwindow.ui
    Audio.h
  Audio.cpp)
endif()
# 链接库
#target_link_libraries(RealTimeDemo PRIVATE Qt5::Widgets Qt5::Multimedia)
target_link_libraries(RealTimeDemo PRIVATE Qt5::Widgets Qt5::Multimedia ${DEEPFILTERNET_LIB})

# 如果使用 MSVC 编译器，添加特定的编译选项
#if (MSVC)
#    target_compile_options(RealTimeDemo PRIVATE /W4)
#else()
#    target_compile_options(RealTimeDemo PRIVATE -Wall -Wextra -pedantic -Werror)
#endif()