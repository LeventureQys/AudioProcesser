#include <stdio.h>
#include <windows.h>
#include <iostream>
#include "deep_filter.h"  // 你的头文件路径
#include "functional"
#include "map"
typedef DFState* (*FuncType)(const char*, float);  // 定义一个函数指针类型，根据你的DLL中函数的实际签名修改

int main() {

    

    const char* path = "D:\\WorkShop\\CurrentWork\\FIRFilter_Venture\\DeepFilter\\DeepFilter_CPP\\source\\DeepFilterNet3_ll_onnx.tar.gz";

    // 加载DLL
    HMODULE hDll = LoadLibrary(TEXT("D:\\WorkShop\\CurrentWork\\FIRFilter_Venture\\DeepFilter\\DeepFilter_CPP\\bin\\deep_filter_ladspa.dll"));
    if (hDll == NULL) {
        DWORD dwError = GetLastError();
        std::cerr << "无法加载DLL! 错误代码: " << dwError << std::endl;
        return 1;
    }

    // 获取函数指针，替换为实际的函数名
    FuncType func = (FuncType)GetProcAddress(hDll, "df_create");
    if (!func) {
        std::cerr << "无法找到函数!" << std::endl;
        FreeLibrary(hDll);
        return 1;
    }

    // 调用函数
    DFState* st = func(path, 10.0f);

    // 释放DLL
    FreeLibrary(hDll);

    return 0;
}
