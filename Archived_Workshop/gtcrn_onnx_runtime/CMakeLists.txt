cmake_minimum_required(VERSION 3.10)
project(gtcrn_native)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 设置输出目录（确保库和可执行文件在同一目录，方便链接）
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# 定义ONNX Runtime路径（建议使用绝对路径）
set(ONNX_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/onnx/include")
set(ONNX_LIB_PATH "${PROJECT_SOURCE_DIR}/onnx/lib/onnxruntime.lib")

# 收集源文件和头文件
file(GLOB STFT_SRC "${PROJECT_SOURCE_DIR}/STFT/*.cpp")
file(GLOB WAV_READER_SRC "${PROJECT_SOURCE_DIR}/wav_reader/*.cpp")
file(GLOB API_SRC "${PROJECT_SOURCE_DIR}/src/*.cpp")
file(GLOB API_HEAD "${PROJECT_SOURCE_DIR}/api/*.h")

# 定义gtcrn_api共享库（先构建库）
add_library(gtcrn_api SHARED
    ${STFT_SRC}
    ${WAV_READER_SRC}
    ${API_SRC}
    ${API_HEAD}
)

# 为共享库添加导出宏定义（匹配api.h中的GTCRN_API_EXPORTS）
target_compile_definitions(gtcrn_api PRIVATE GTCRN_API_EXPORTS)

# 链接ONNX库
target_include_directories(gtcrn_api PUBLIC
    ${PROJECT_SOURCE_DIR}/api
    ${PROJECT_SOURCE_DIR}/STFT
    ${PROJECT_SOURCE_DIR}/wav_reader
    ${ONNX_INCLUDE_DIR}
)
target_link_libraries(gtcrn_api PRIVATE ${ONNX_LIB_PATH})

# 构建主程序（依赖gtcrn_api）
add_executable(gtcrn_native
    ${PROJECT_SOURCE_DIR}/main.cpp
)
target_link_libraries(gtcrn_native PRIVATE gtcrn_api)

# 构建demo程序（依赖gtcrn_api，确保gtcrn_api先构建）
add_executable(gtcrn_demo
    ${PROJECT_SOURCE_DIR}/demo/main.cpp
    ${WAV_READER_SRC}
)
target_include_directories(gtcrn_demo PRIVATE
    ${PROJECT_SOURCE_DIR}/wav_reader
    ${PROJECT_SOURCE_DIR}/api
)
# 显式指定依赖，确保构建顺序
add_dependencies(gtcrn_demo gtcrn_api)
target_link_libraries(gtcrn_demo PRIVATE gtcrn_api)