cmake_minimum_required(VERSION 3.5)

project(AudioRecorder LANGUAGES CXX)

# 设置Qt6路径（如果需要）
# set(Qt6_DIR "path/to/Qt6")

# 查找Qt6包
find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets Multimedia)

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/$<CONFIG>)

# 添加子目录
add_subdirectory(./AudioRecorder_Core)

# 主项目设置
set(SOURCES
    main.cpp
    mainwindow.cpp
    mainwindow.h
    mainwindow.ui  # 确保UI文件包含在源文件列表中
)

# 添加可执行文件
add_executable(AudioRecorder ${SOURCES})

# 链接Qt6库
target_link_libraries(AudioRecorder PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets    # 必须链接Widgets模块以支持UI
    Qt6::Multimedia
)

# 链接子项目
target_link_libraries(AudioRecorder PRIVATE AudioRecorder_Core)

# 自动处理moc、uic、rcc
set_target_properties(AudioRecorder PROPERTIES
    AUTOMOC ON      # 自动处理Qt的元对象系统
    AUTOUIC ON      # 自动处理UI文件
    AUTORCC ON      # 自动处理资源文件
    WIN32_EXECUTABLE ON
)