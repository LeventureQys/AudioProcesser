
# NoiseSuppressor算法 

WebRTC 的 Noise Suppressor (NS) 算法核心确实使用了​​维纳滤波（Wiener Filtering）​​的思想，但并不是直接的传统维纳滤波实现，而是采用了更先进的​​改进型维纳滤波​​结合​​MMSE-STSA（Minimum Mean-Square Error Short-Time Spectral Amplitude）​​准则的混合方法。

G(k,l) = ξ(k,l) / (1 + ξ(k,l))

## 一、算法整体架构：

<img width="644" height="1852" alt="eb452dab6cd23" src="https://github.com/user-attachments/assets/345f426f-2d5f-47fc-af51-5dc16075de9c" />

## 二、详细处理流程

### 1.预处理阶段

<img width="360" height="1052" alt="0d250eeb3f5ce" src="https://github.com/user-attachments/assets/8182dd0f-4adf-470e-91d2-f8709972586e" />

- 分帧处理： 将音频信号分割为20-30ms的帧
- 加窗： 通常使用汉宁窗减少频谱泄露
- FFT变换：将时域信号转换为频域表示

### 2. 噪声估计模块

<img width="772" height="1239" alt="e87f6c2473b75" src="https://github.com/user-attachments/assets/93efbaf4-43f7-48ce-9d56-66177180dfb4" />

- 使用最小统计量方法跟踪噪声基底
- 在非语音段更新噪声估计
- 采用递归平滑技术保持估计稳定性

### 3.增益计算模块

<img width="424" height="1468" alt="b61e0f458fbe2" src="https://github.com/user-attachments/assets/7d2718d7-e8cd-4cb7-910e-e0896e22ead4" />

- 基于MMSE-STSA(最小均方误差短时谱幅度)准则
- 采用决策导向方法更新先验SNR
- 增益限制防止音乐噪声

### 4. 后处理阶段

<img width="360" height="1052" alt="480faf5ddff48" src="https://github.com/user-attachments/assets/fed4167b-562c-478b-93d9-bd246624310f" />


