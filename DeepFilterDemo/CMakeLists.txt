cmake_minimum_required(VERSION 3.5.0)
project(DeepFilterDemo VERSION 0.1.0 LANGUAGES C CXX)

# 设置 C++ 标准为 C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  # 禁用编译器特定的扩展
# 添加头文件路径
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# 添加子目录，包含完整的CMakeLists.txt


# 查找库文件
find_library(DEEPFILTERNET_LIB NAMES deepfilter PATHS ${CMAKE_CURRENT_SOURCE_DIR}/lib64)
message(STATUS "DeepFilterNet library found at: ${DEEPFILTERNET_LIB}")
# 确保库文件被找到
if (NOT DEEPFILTERNET_LIB)
    message(FATAL_ERROR "DeepFilterNet library not found")
endif()

# 添加可执行文件
add_executable(DeepFilterDemo main.cpp)

# 链接库
target_link_libraries(DeepFilterDemo ${DEEPFILTERNET_LIB})
# 添加编译选项
target_compile_features(DeepFilterDemo PRIVATE cxx_std_20)

# 如果使用 MSVC 编译器，添加特定的编译选项
if (MSVC)
    target_compile_options(DeepFilterDemo PRIVATE /W4)
else()
    target_compile_options(DeepFilterDemo PRIVATE -Wall -Wextra -pedantic -Werror)
endif()
