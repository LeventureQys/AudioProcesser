import torch
from transformers import HubertModel, Wav2Vec2FeatureExtractor
import onnx
import onnxruntime as ort
import numpy as np


### 输出(1, 49, 768)

# 第一个维度 1：批次大小（Batch Size）,表示当前输入的音频样本数量。
# 第二个维度 49：时间步（Time Steps）,这个数值与输入音频的长度直接相关：16k 采样率下，HuBERT 大约每 20ms 生成一个时间步，因此 1 秒音频约对应 50 个时间步（你的结果 49 是正常的计算偏差）。
# 第三个维度 768：特征维度（Feature Dimension）,这个值由你使用的模型版本决定：facebook/hubert-base-ls960 是基础版，输出维度为 768；如果使用 large 版（如 facebook/hubert-large-ls960），这个维度会是 1024。

# 需要注意的是 这里使用的是v1版本

# 模型名称 - 可以根据需要选择不同大小的模型
MODEL_NAME = "facebook/hubert-base-ls960"
ONNX_OUTPUT_PATH = "hubert_rvc.onnx"

# 加载模型和特征提取器
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
model = HubertModel.from_pretrained(MODEL_NAME)

# 设置为评估模式
model.eval()

# 创建一个示例输入 (16kHz音频，这里生成随机数据作为示例)
# 实际使用时应替换为真实的16kHz音频数据
sample_rate = 16000
duration_seconds = 1  # 1秒的音频
input_audio = np.random.randn(int(sample_rate * duration_seconds)).astype(np.float32)

# 预处理音频
inputs = feature_extractor(
    input_audio, 
    sampling_rate=sample_rate, 
    return_tensors="pt",
    padding=True
)

# 导出为ONNX
with torch.no_grad():
    # 定义导出函数
    def export_model():
        # 我们只需要最后一层的特征
        outputs = model(**inputs, output_hidden_states=True)
        # 取最后一层的隐藏状态
        last_hidden_state = outputs.last_hidden_state
        
        # 对时间维度求平均，得到固定大小的特征向量
        # 这里我们将其降维到12维，适合RVC后续处理
        feature_vector = torch.mean(last_hidden_state, dim=1)
        # 如果需要精确的12维，可以添加一个线性层进行降维
        projection = torch.nn.Linear(feature_vector.shape[-1], 12).to(feature_vector.device)
        return projection(feature_vector)
    
    # 导出模型
    torch.onnx.export(
        model,
        args=tuple(inputs.values()),
        f=ONNX_OUTPUT_PATH,
        input_names=["input_values"],
        output_names=["feature_vector"],
        dynamic_axes={
            "input_values": {0: "batch_size", 1: "sequence_length"}
        },
        opset_version=14,
        do_constant_folding=True
    )

# 执行导出
export_model()

# 验证导出的ONNX模型
def verify_onnx_model():
    # 加载ONNX模型
    onnx_model = onnx.load(ONNX_OUTPUT_PATH)
    onnx.checker.check_model(onnx_model)
    
    # 使用ONNX Runtime运行
    ort_session = ort.InferenceSession(ONNX_OUTPUT_PATH)
    
    # 准备输入数据
    input_name = ort_session.get_inputs()[0].name
    input_data = inputs["input_values"].numpy()
    
    # 运行推理
    outputs = ort_session.run(None, {input_name: input_data})
    
    # 检查输出形状是否为12维
    feature_vector = outputs[0]
    print(f"输出特征形状: {feature_vector.shape}")

# 验证模型
verify_onnx_model()
print(f"模型已成功导出至 {ONNX_OUTPUT_PATH}")

