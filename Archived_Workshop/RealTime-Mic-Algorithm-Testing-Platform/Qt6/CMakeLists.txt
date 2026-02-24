cmake_minimum_required(VERSION 3.5.0)
project(RealTime-Mic-Algorithm-Testing-Platform VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(Qt6 COMPONENTS Widgets Multimedia REQUIRED)

add_executable(RealTime-Mic-Algorithm-Testing-Platform 
main.cpp
AudioCondition.ui
AudioCondition.h
AudioCondition.cpp
Audio.h
)
target_link_libraries(RealTime-Mic-Algorithm-Testing-Platform PRIVATE Qt6::Widgets Qt6::Multimedia ${DEEPFILTERNET_LIB})