import torch
from transformers import HubertModel, Wav2Vec2FeatureExtractor
import onnx
import onnxruntime as ort
import numpy as np
# 差异统计:
# 绝对差异均值: 0.000005
# 绝对差异最大值: 0.000071
# 绝对差异方差: 0.000000

# PyTorch输出统计:
# 均值: -0.001013
# 方差: 0.145676

# ONNX输出统计:
# 均值: -0.001013
# 方差: 0.145677
# 模型名称
MODEL_NAME = "facebook/hubert-base-ls960"
ONNX_OUTPUT_PATH = "hubert_raw.onnx"  # 输出原始特征的ONNX模型

# 加载模型和特征提取器
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
model = HubertModel.from_pretrained(MODEL_NAME)
model.eval()  # 设置为评估模式

# 创建示例输入 (16kHz音频，随机数据)
sample_rate = 16000
duration_seconds = 1  # 1秒音频，对应约49个时间步
input_audio = np.random.randn(int(sample_rate * duration_seconds)).astype(np.float32)

# 预处理音频 (转为PyTorch张量)
inputs = feature_extractor(
    input_audio, 
    sampling_rate=sample_rate, 
    return_tensors="pt",
    padding=True
)

# 导出为ONNX (只保留原始last_hidden_state)
with torch.no_grad():
    # 定义导出函数：仅返回原始的last_hidden_state
    def export_raw_model():
        outputs = model(** inputs, output_hidden_states=True)
        return outputs.last_hidden_state  # 直接返回(1,49,768)的特征
    
    # 导出ONNX模型
    torch.onnx.export(
        model,  # 原始Hubert模型
        args=tuple(inputs.values()),  # 输入参数
        f=ONNX_OUTPUT_PATH,
        input_names=["input_values"],  # 输入名称
        output_names=["last_hidden_state"],  # 输出名称（明确为原始特征）
        dynamic_axes={
            "input_values": {0: "batch_size", 1: "sequence_length"},  # 支持动态批次和序列长度
            "last_hidden_state": {0: "batch_size", 1: "time_steps"}  # 输出的动态维度
        },
        opset_version=14,
        do_constant_folding=True
    )

# 执行导出
export_raw_model()

# 验证并比较PyTorch与ONNX的输出（均为(1,49,768)）
def verify_raw_model():
    # 加载ONNX模型并检查有效性
    onnx_model = onnx.load(ONNX_OUTPUT_PATH)
    onnx.checker.check_model(onnx_model)
    
    # 准备输入数据（转为numpy）
    input_data = inputs["input_values"].numpy()
    
    # 获取PyTorch模型的原始输出 (1,49,768)
    with torch.no_grad():
        outputs_pytorch = model(**inputs, output_hidden_states=True)
        output_pytorch = outputs_pytorch.last_hidden_state.numpy()  # 直接取原始特征
    
    # 获取ONNX模型的输出 (1,49,768)
    ort_session = ort.InferenceSession(ONNX_OUTPUT_PATH)
    input_name = ort_session.get_inputs()[0].name
    output_onnx = ort_session.run(None, {input_name: input_data})[0]  # 原始特征
    
    # 打印输出形状（确认均为(1,49,768)）
    print(f"PyTorch输出形状: {output_pytorch.shape}")
    print(f"ONNX输出形状: {output_onnx.shape}")
    
    # 计算差异统计
    absolute_diff = np.abs(output_pytorch - output_onnx)
    
    print("\n差异统计:")
    print(f"绝对差异均值: {np.mean(absolute_diff):.6f}")
    print(f"绝对差异最大值: {np.max(absolute_diff):.6f}")
    print(f"绝对差异方差: {np.var(absolute_diff):.6f}")
    
    # 输出各自的均值和方差
    print("\nPyTorch输出统计:")
    print(f"均值: {np.mean(output_pytorch):.6f}")
    print(f"方差: {np.var(output_pytorch):.6f}")
    
    print("\nONNX输出统计:")
    print(f"均值: {np.mean(output_onnx):.6f}")
    print(f"方差: {np.var(output_onnx):.6f}")

# 执行验证
verify_raw_model()
print(f"\n模型已成功导出至 {ONNX_OUTPUT_PATH}")
