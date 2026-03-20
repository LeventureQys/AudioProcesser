"""
MNIST数据集查看器
展示如何读取和可视化下载的MNIST数据
"""

import struct
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def read_idx_images(filename):
    """
    读取MNIST图像文件（IDX格式）

    文件格式：
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    """
    with open(filename, 'rb') as f:
        # 读取文件头
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))

        # 验证magic number
        if magic != 2051:
            raise ValueError(f'Invalid magic number {magic}, expected 2051')

        print(f"图像文件信息:")
        print(f"  - 图像数量: {num_images}")
        print(f"  - 图像大小: {rows} x {cols}")

        # 读取所有图像数据
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows, cols)

        return images


def read_idx_labels(filename):
    """
    读取MNIST标签文件（IDX格式）

    文件格式：
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number
    0004     32 bit integer  60000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    """
    with open(filename, 'rb') as f:
        # 读取文件头
        magic, num_labels = struct.unpack('>II', f.read(8))

        # 验证magic number
        if magic != 2049:
            raise ValueError(f'Invalid magic number {magic}, expected 2049')

        print(f"\n标签文件信息:")
        print(f"  - 标签数量: {num_labels}")

        # 读取所有标签
        labels = np.frombuffer(f.read(), dtype=np.uint8)

        return labels


def visualize_samples(images, labels, num_samples=20):
    """可视化一些样本"""
    # 创建图形
    fig, axes = plt.subplots(4, 5, figsize=(12, 10))
    fig.suptitle('MNIST手写数字样本', fontsize=16, fontweight='bold')

    # 显示样本
    for i, ax in enumerate(axes.flat):
        if i < num_samples:
            # 显示图像
            ax.imshow(images[i], cmap='gray')
            ax.set_title(f'标签: {labels[i]}', fontsize=12, fontweight='bold')
            ax.axis('off')

    plt.tight_layout()
    plt.savefig('mnist_samples.png', dpi=150, bbox_inches='tight')
    print(f"\n样本图像已保存到: mnist_samples.png")
    plt.show()


def show_statistics(images, labels):
    """显示数据集统计信息"""
    print("\n" + "="*60)
    print("MNIST数据集统计信息")
    print("="*60)

    # 基本信息
    print(f"\n图像数据:")
    print(f"  - 形状: {images.shape}")
    print(f"  - 数据类型: {images.dtype}")
    print(f"  - 像素值范围: [{images.min()}, {images.max()}]")
    print(f"  - 平均像素值: {images.mean():.2f}")

    print(f"\n标签数据:")
    print(f"  - 形状: {labels.shape}")
    print(f"  - 数据类型: {labels.dtype}")
    print(f"  - 唯一标签: {np.unique(labels)}")

    # 每个数字的数量
    print(f"\n每个数字的样本数量:")
    for digit in range(10):
        count = np.sum(labels == digit)
        print(f"  数字 {digit}: {count:5d} 张 ({count/len(labels)*100:.2f}%)")

    print("="*60)


def visualize_digit_distribution(labels):
    """可视化数字分布"""
    # 统计每个数字的数量
    digit_counts = [np.sum(labels == i) for i in range(10)]

    # 绘制柱状图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(10), digit_counts, color='steelblue', alpha=0.8, edgecolor='black')

    # 添加数值标签
    for bar, count in zip(bars, digit_counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.xlabel('数字', fontsize=12, fontweight='bold')
    plt.ylabel('样本数量', fontsize=12, fontweight='bold')
    plt.title('MNIST数据集中各数字的分布', fontsize=14, fontweight='bold')
    plt.xticks(range(10))
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('mnist_distribution.png', dpi=150, bbox_inches='tight')
    print(f"分布图已保存到: mnist_distribution.png")
    plt.show()


def show_single_digit_examples(images, labels, digit=5, num_examples=10):
    """显示某个特定数字的多个示例"""
    # 找到所有该数字的索引
    indices = np.where(labels == digit)[0]

    # 随机选择一些示例
    selected_indices = np.random.choice(indices, min(num_examples, len(indices)), replace=False)

    # 可视化
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle(f'数字 "{digit}" 的不同写法', fontsize=14, fontweight='bold')

    for i, (ax, idx) in enumerate(zip(axes.flat, selected_indices)):
        ax.imshow(images[idx], cmap='gray')
        ax.set_title(f'样本 {i+1}', fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'mnist_digit_{digit}_examples.png', dpi=150, bbox_inches='tight')
    print(f"数字{digit}的示例已保存到: mnist_digit_{digit}_examples.png")
    plt.show()


def main():
    """主函数"""
    # 数据目录
    data_dir = Path('./data/mnist')

    # 检查文件是否存在
    train_images_file = data_dir / 'train-images-idx3-ubyte'
    train_labels_file = data_dir / 'train-labels-idx1-ubyte'

    if not train_images_file.exists() or not train_labels_file.exists():
        print("错误: MNIST数据文件不存在！")
        print(f"请先运行下载器: python download.py --dataset mnist")
        return

    print("="*60)
    print("MNIST数据集查看器")
    print("="*60)

    # 读取训练集
    print("\n正在读取训练集...")
    train_images = read_idx_images(train_images_file)
    train_labels = read_idx_labels(train_labels_file)

    # 显示统计信息
    show_statistics(train_images, train_labels)

    # 可视化样本
    print("\n正在生成可视化图像...")
    visualize_samples(train_images, train_labels, num_samples=20)

    # 可视化数字分布
    visualize_digit_distribution(train_labels)

    # 显示某个数字的多个示例
    show_single_digit_examples(train_images, train_labels, digit=5, num_examples=10)

    print("\n✓ 完成！已生成所有可视化图像。")

    # 使用PyTorch加载（如果已安装）
    try:
        import torch
        from torchvision import datasets, transforms

        print("\n" + "="*60)
        print("使用PyTorch加载MNIST（推荐方式）")
        print("="*60)

        # 定义数据变换
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # 加载数据集
        train_dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=False,  # 已经下载过了
            transform=transform
        )

        test_dataset = datasets.MNIST(
            root='./data',
            train=False,
            download=False,
            transform=transform
        )

        print(f"\nPyTorch加载成功:")
        print(f"  - 训练集大小: {len(train_dataset)}")
        print(f"  - 测试集大小: {len(test_dataset)}")

        # 获取一个样本
        image, label = train_dataset[0]
        print(f"\n单个样本:")
        print(f"  - 图像形状: {image.shape}")
        print(f"  - 标签: {label}")

        print("\n使用示例:")
        print("```python")
        print("from torch.utils.data import DataLoader")
        print("train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)")
        print("for images, labels in train_loader:")
        print("    # images: (batch_size, 1, 28, 28)")
        print("    # labels: (batch_size,)")
        print("    pass")
        print("```")

    except ImportError:
        print("\n提示: 安装PyTorch后可以更方便地加载数据:")
        print("  pip install torch torchvision")


if __name__ == '__main__':
    main()
