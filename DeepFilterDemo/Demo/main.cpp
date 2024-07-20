#include <stdio.h>
#include <ctime>
#include "iostream"
#include "include/df/deep_filter.h"
#include <chrono>
#define FRAME_SIZE 480
// 获取当前时间的毫秒时间戳
long long currentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    return millis;
}


int main(int argc, char** argv) {
    int first = 1;
    short x[FRAME_SIZE];
    FILE* f1, * fout;
    DFState* st;
    st = df_create("D:/WorkShop/CurrentWork/DeepFilterNet/models/DeepFilterNet3_onnx.tar.gz", 100.);
    f1 = fopen("D:/temp/audiotest/HighSNRatio.pcm", "rb");
    fout = fopen("D:/temp/audiotest/output/HighSNRatio.pcm", "wb");
 
    while (1) {
        fread(x, sizeof(short), FRAME_SIZE, f1);
        if (feof(f1)) break;
        long long firstTime = currentTimestamp();
        df_process_frame_i16(st, x, x);
        long long lastTime = currentTimestamp();
        if (!first) fwrite(x, sizeof(short), FRAME_SIZE, fout);
        first = 0;
        printf("time cost: %lld ms\n", lastTime - firstTime);
    }


    df_free(st);
    fclose(f1);
    fclose(fout);
    return 0;
}