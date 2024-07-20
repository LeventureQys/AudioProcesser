# 编译说明

使用cmake进行编译,DeepFilter中所使用的库目前都已经打包编译好了,可以直接使用。

# 使用说明

具体调用在main.cpp内，需要使用到DeepFilterNet3_onnx.tar.gz，具体我存放在当前目录下的/modules目录下，或者在DeepFilterNet/modules/下，请使用DeepFilterNet3参数，否则会报错导致程序崩溃


# 运行结果
降噪效果正常，9s音频的延迟大概在1200ms左右，也就是约等于130ms/s

