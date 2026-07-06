# ONNX - 它到底快在哪，又慢在哪

---

## 一、ONNX 是什么

ONNX（Open Neural Network Exchange）本质上是一种模型的中间表示格式（IR），类似于编译器里的 LLVM IR。

用 PyTorch 训练完一个模型之后，模型的计算逻辑是用 Python 描述的，跑推理的时候要经过 Python 解释器、PyTorch 的调度器、再到底层的 CUDA kernel。这条链路很长，开销不小。ONNX 做的事情是：把模型的计算图从 PyTorch 的世界里"导出"成一个独立的、与框架无关的静态计算图，然后交给专门的推理引擎（比如 ONNX Runtime）去执行。

打个比方：PyTorch 训练出来的模型像一份 Python 脚本，每次执行都要解释器逐行翻译；ONNX 导出后的模型像一份编译好的二进制文件，直接跑就行。

一个典型的 ONNX 文件（`.onnx`）里面存的是：

- 计算图的拓扑结构（哪些算子、怎么连接）
- 每个算子的类型和参数（Conv、MatMul、Relu 等）
- 模型的权重（以 protobuf 格式序列化）
- 输入输出的 shape 和数据类型

这里要注意一点：ONNX 本身只是一个格式规范，它不负责执行。真正跑推理的是 ONNX Runtime（简称 ORT）或者其他兼容 ONNX 的推理引擎（TensorRT、OpenVINO 等）。说"用 ONNX 加速"，实际上是"用 ONNX Runtime 加速"。

---

## 二、ONNX Runtime 为什么能快

理解了 ONNX 是什么之后，关键问题来了：为什么换个引擎跑就能快？

### 2.1 去掉了 Python 开销

PyTorch 推理时，即使模型本身的计算是在 GPU 上跑的，Python 层面仍然有大量开销：

- 每个算子调用都要经过 Python 的函数调度
- 动态图机制意味着每次 forward 都要重新构建计算图
- GIL（全局解释器锁）在多线程场景下是个瓶颈

ORT 是纯 C++ 实现的推理引擎，模型加载完之后，整个推理过程不经过 Python。对于那些计算量不大但算子数量多的模型（比如很多小卷积串联的网络），光是去掉 Python 开销就能带来可观的提速。

### 2.2 图优化（Graph Optimization）

这是 ORT 最核心的加速手段。静态计算图的好处是：引擎可以在推理之前，对整张图做全局优化。常见的优化包括：

**算子融合（Operator Fusion）**：把多个相邻算子合并成一个。比如 Conv + BatchNorm + ReLU 三个算子，在训练时是分开执行的（因为 BatchNorm 在训练和推理时行为不同），但在推理时可以融合成一个 kernel。这样做的好处是减少了 kernel launch 的次数，也减少了中间结果在显存里的读写。

**常量折叠（Constant Folding）**：如果计算图里有些节点的输入全是常量（比如某个 reshape 的 shape 参数），那就在加载时直接算好，推理时跳过。

**冗余节点消除**：去掉那些对输出没有影响的计算节点，比如连续的两次 transpose 如果互为逆操作，就直接删掉。

**内存规划（Memory Planning）**：分析整张图的数据流，提前规划好每个中间张量的内存分配和复用策略，避免运行时频繁的 malloc/free。

### 2.3 硬件特化的执行后端

ORT 支持多种 Execution Provider（EP）：

| EP | 适用硬件 | 特点 |
|---|---|---|
| CPU EP | 通用 CPU | 使用 MLAS 数学库，针对 x86/ARM 做了 SIMD 优化 |
| CUDA EP | NVIDIA GPU | 调用 cuDNN/cuBLAS |
| TensorRT EP | NVIDIA GPU | 进一步将子图编译为 TensorRT engine |
| DirectML EP | Windows GPU | 通过 DirectX 12 调用 GPU，支持 AMD/Intel/NVIDIA |
| OpenVINO EP | Intel CPU/GPU/VPU | Intel 硬件上的深度优化 |

选对 EP 很重要。同一个模型，用 CPU EP 和用 CUDA EP 的性能差距可以是数量级的。

---

## 三、什么时候 ONNX 能加速

说了这么多原理，落到实际场景里，以下情况用 ONNX 通常能获得明显加速：

### 3.1 模型结构固定、输入 shape 固定

这是 ONNX 最舒服的场景。输入 shape 固定意味着 ORT 可以在加载时就把所有优化做到位，包括内存预分配、kernel 选择、算子融合等。典型例子：

- 固定长度的音频帧处理（比如每次处理 1024 个采样点）
- 固定分辨率的图像分类
- 固定序列长度的 NLP 推理

### 3.2 算子数量多、单个算子计算量小

前面说了，Python 调度开销是按算子数量线性增长的。如果一个模型有几百个小算子（比如 MobileNet 这类轻量网络），PyTorch 推理时大量时间花在调度上而不是计算上。换成 ORT 之后，调度开销几乎为零，提速非常明显。

实测数据参考：一个包含 200+ 算子的轻量语音增强模型，PyTorch 推理耗时 12ms，转 ONNX 后 ORT CPU 推理耗时 3ms，快了 4 倍。

### 3.3 需要部署到非 Python 环境

如果最终要把模型部署到 C++ 服务、移动端、嵌入式设备上，ONNX 几乎是必经之路。ORT 提供 C/C++、C#、Java、JavaScript 等语言的 API，不依赖 Python 环境。

### 3.4 CPU 推理场景

在纯 CPU 推理的场景下，ORT 的 MLAS 库针对矩阵运算做了大量 SIMD 优化（AVX2、AVX-512、NEON 等），通常比 PyTorch 的 CPU 后端（基于 MKL 或 OpenBLAS）更快，尤其是在 batch size 较小的时候。

---

## 四、什么时候 ONNX 反而更慢

这才是很多人踩坑的地方。

### 4.1 动态 shape 场景

如果模型的输入 shape 每次都不一样（比如变长序列、不同分辨率的图像），ORT 的很多优化就失效了：

- 无法预分配内存，每次推理都要重新规划
- 算子融合的某些模式依赖固定 shape，动态 shape 下无法触发
- 某些 EP（比如 TensorRT）需要为每个新 shape 重新编译 engine，第一次遇到新 shape 时会有巨大的延迟

PyTorch 的动态图机制天然支持动态 shape，反而没有这个问题。

**实际建议**：如果必须处理变长输入，可以用 padding 把输入对齐到几个固定的 shape 档位（比如 256、512、1024），然后为每个档位各导出一个 ONNX 模型，或者使用 ORT 的动态 shape 支持但接受一定的性能损失。

### 4.2 模型包含不支持的算子

ONNX 的算子集（opset）是有限的。如果模型里用了某些 PyTorch 独有的算子或者自定义算子，导出时可能会：

- 导出失败
- 被拆解成一堆等价但低效的基础算子组合

后者尤其隐蔽。表面上导出成功了，但实际推理速度反而变慢了，因为原本一个高效的自定义 kernel 被替换成了十几个基础算子的串联。

遇到这种情况，要么给 ORT 注册自定义算子（Custom Operator），要么考虑换用其他推理引擎。

### 4.3 模型本身计算量很大，瓶颈在 GPU kernel

如果模型的瓶颈是大矩阵乘法或大卷积（比如 ResNet-152、大型 Transformer），那 Python 调度开销相对于 GPU 计算时间来说微不足道。这种情况下，PyTorch 和 ORT 调用的底层 kernel 是一样的（都是 cuDNN/cuBLAS），性能差距很小，甚至可能因为 ORT 的 kernel 选择策略不如 PyTorch 的 autotuner 而略慢。

简单说：模型越大、单次计算越重，ONNX 的加速比越小。

### 4.4 导出过程引入了额外开销

PyTorch 导出 ONNX 时使用 `torch.onnx.export`，底层是通过 tracing（追踪）或 scripting（脚本化）来捕获计算图。这个过程有时候会引入一些不必要的操作：

- 某些 Python 控制流被展开成冗长的计算图
- 某些 in-place 操作被替换成 copy + 操作
- 数据类型转换节点被插入

这些都可能导致导出后的模型比原始 PyTorch 模型更慢。导出之后，建议用 Netron 之类的工具可视化一下计算图，检查有没有异常。

---

## 五、内存占用模式

这是另一个经常被忽略的话题。

### 5.1 模型加载阶段

ONNX 模型文件本身就是权重 + 计算图的序列化。加载时，ORT 会：

1. 反序列化 protobuf，解析计算图结构
2. 将权重数据加载到内存（CPU）或显存（GPU）
3. 执行图优化（这一步会产生临时的内存开销）
4. 为中间张量预分配内存池

加载完成后的内存占用大致等于：**模型权重大小 + 中间张量内存池 + 引擎自身开销**。

对比 PyTorch：PyTorch 加载模型时只加载权重，中间张量是在 forward 时动态分配的。所以 ORT 的初始内存占用通常比 PyTorch 高一些，但推理时的内存波动更小。

### 5.2 推理阶段

ORT 的内存管理策略和 PyTorch 有本质区别：

**PyTorch（动态分配）**：
- 每次 forward 时，按需分配中间张量的内存
- forward 结束后，中间张量被 Python GC 回收（或者由 CUDA caching allocator 缓存）
- 内存占用呈锯齿状波动：forward 时上升，结束后下降

**ORT（预分配 + 复用）**：
- 加载时分析整张计算图，计算出中间张量的最大内存需求
- 预分配一个内存池，推理时所有中间张量从池中分配
- 生命周期不重叠的张量共享同一块内存
- 内存占用基本恒定，没有波动


### 5.3 实际影响

这种内存模式的差异在以下场景中很重要：

**长时间运行的服务**：ORT 的恒定内存占用更友好，不会因为 GC 延迟导致内存峰值。PyTorch 的锯齿状模式在高并发下可能导致 OOM，因为多个请求的内存峰值可能叠加。

**嵌入式/资源受限环境**：ORT 的内存占用可预测，方便做资源规划。你可以在部署前就精确知道模型需要多少内存。

**GPU 显存**：ORT 的 CUDA EP 同样使用预分配策略。如果你在一张卡上跑多个模型，ORT 的显存占用更可控。但要注意，ORT 默认会预分配较大的显存池，可以通过 `arena_extend_strategy` 参数调整。

### 5.4 量化对内存的影响

如果在 ONNX 上做量化（INT8/FP16），内存占用会进一步降低：

| 精度 | 权重大小（相对） | 中间张量大小（相对） |
|---|---|---|
| FP32 | 1x | 1x |
| FP16 | 0.5x | 0.5x |
| INT8 | 0.25x | 取决于实现，通常 0.25x~0.5x |

但量化不是免费的午餐，精度损失需要评估。对于音频处理这类对精度敏感的场景，建议先做 FP16 量化（精度损失通常可忽略），INT8 则需要仔细验证。

---

## 六、实践建议

最后总结几条实操经验：

1. **先 profile，再决定要不要转 ONNX**。用 PyTorch Profiler 看一下推理的时间分布，如果 90% 的时间花在 GPU kernel 上，转 ONNX 意义不大。如果大量时间花在 CPU 端的调度和数据搬运上，转 ONNX 大概率有收益。

2. **导出后一定要做数值验证**。用相同的输入，对比 PyTorch 和 ORT 的输出，确保误差在可接受范围内（通常 FP32 下 atol=1e-5 是合理的阈值）。

3. **固定 shape 能固定就固定**。哪怕要为不同 shape 导出多个模型，也比用动态 shape 跑得快。

4. **注意 opset 版本**。不同 opset 版本支持的算子不同，优化效果也不同。一般建议用最新的稳定版本（目前是 opset 18-20）。

5. **ORT 的 Session Options 值得调**。`graph_optimization_level`、`intra_op_num_threads`、`execution_mode` 这几个参数对性能影响很大，不要用默认值就完事了。

6. **如果目标是 NVIDIA GPU，考虑 TensorRT EP**。它会把 ONNX 子图进一步编译为 TensorRT engine，通常比纯 CUDA EP 再快 20-50%，但首次加载时间会显著增加。

---

## 参考

- [ONNX 官方规范](https://onnx.ai/onnx/intro/)
- [ONNX Runtime 性能调优文档](https://onnxruntime.ai/docs/performance/tune-performance.html)
- [ONNX Runtime Graph Optimizations](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html)
