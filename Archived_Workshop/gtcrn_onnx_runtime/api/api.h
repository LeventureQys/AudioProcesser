// api.h
#ifndef API_H
#define API_H

#include <stdint.h>
#include <stdlib.h>

// 导出/导入声明（Windows专用）
#ifdef _WIN32
#ifdef GTCRN_API_EXPORTS
#define GTCRN_API __declspec(dllexport)
#else
#define GTCRN_API __declspec(dllimport)
#endif
#else
#define GTCRN_API
#endif

// 处理状态枚举
typedef enum {
    PROCESS_SUCCESS = 0,
    PROCESS_INIT_FAILED,
    PROCESS_INVALID_INPUT,
    PROCESS_MEMORY_ERROR,
    PROCESS_INTERNAL_ERROR
} ProcessStatus;

// 前向声明处理上下文
typedef struct AudioProcessorContext AudioProcessorContext;

// 导出接口函数
GTCRN_API AudioProcessorContext* audio_processor_init(
    const char* model_path,
    int sample_rate,
    int frame_size,
    int hop_size
);

GTCRN_API ProcessStatus audio_processor_process(
    AudioProcessorContext* context,
    const int16_t* input,
    size_t input_size,
    int16_t* output,
    size_t* output_size
);

GTCRN_API void audio_processor_destroy(AudioProcessorContext* context);

#endif // API_H