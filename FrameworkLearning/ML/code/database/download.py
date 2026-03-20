"""
数据集下载器
支持断点续传、进度条显示、多线程下载、自动代理检测
"""

import os
import requests
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import gzip
import shutil
from typing import List, Tuple, Dict
import hashlib


class DatasetDownloader:
    """数据集下载器，支持断点续传、多线程下载和自动代理检测"""

    # MNIST数据集的URL - 使用国内镜像源

    # 阿里云镜像源（推荐，速度快）
    MNIST_URLS_ALIYUN = {
        'train_images': 'https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz',
        'train_labels': 'https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz',
        'test_images': 'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz',
        'test_labels': 'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz'
    }

    # OpenI启智社区镜像源
    MNIST_URLS_OPENI = {
        'train_images': 'https://git.openi.org.cn/datasets/mnist/resolve/master/train-images-idx3-ubyte.gz',
        'train_labels': 'https://git.openi.org.cn/datasets/mnist/resolve/master/train-labels-idx1-ubyte.gz',
        'test_images': 'https://git.openi.org.cn/datasets/mnist/resolve/master/t10k-images-idx3-ubyte.gz',
        'test_labels': 'https://git.openi.org.cn/datasets/mnist/resolve/master/t10k-labels-idx1-ubyte.gz'
    }

    # GitHub镜像（通过jsdelivr CDN加速）
    MNIST_URLS_JSDELIVR = {
        'train_images': 'https://cdn.jsdelivr.net/gh/pytorch/pytorch@master/test/cpp/api/mnist/train-images-idx3-ubyte.gz',
        'train_labels': 'https://cdn.jsdelivr.net/gh/pytorch/pytorch@master/test/cpp/api/mnist/train-labels-idx1-ubyte.gz',
        'test_images': 'https://cdn.jsdelivr.net/gh/pytorch/pytorch@master/test/cpp/api/mnist/t10k-images-idx3-ubyte.gz',
        'test_labels': 'https://cdn.jsdelivr.net/gh/pytorch/pytorch@master/test/cpp/api/mnist/t10k-labels-idx1-ubyte.gz'
    }

    # 原始源（Yann LeCun官网）
    MNIST_URLS_ORIGINAL = {
        'train_images': 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'train_labels': 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'test_images': 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'test_labels': 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
    }

    # 数据集配置（默认使用PyTorch的AWS S3镜像）
    DATASETS = {
        'mnist': {
            'urls': MNIST_URLS_ALIYUN,
            'description': 'MNIST手写数字数据集',
            'mirrors': {
                'aliyun': MNIST_URLS_ALIYUN,
                'openi': MNIST_URLS_OPENI,
                'jsdelivr': MNIST_URLS_JSDELIVR,
                'original': MNIST_URLS_ORIGINAL
            }
        }
    }

    def __init__(self, download_dir: str = './data', num_threads: int = 4):
        """
        初始化下载器

        Args:
            download_dir: 下载目录
            num_threads: 下载线程数
        """
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.num_threads = num_threads
        self.proxies = self._detect_proxy()

    def _detect_proxy(self) -> Dict[str, str]:
        """
        检测系统代理设置

        Returns:
            Dict[str, str]: 代理配置字典 {'http': proxy_url, 'https': proxy_url}
        """
        proxies = {}

        # 检查环境变量中的代理设置
        # 支持大小写变体
        for proto in ['http', 'https']:
            # 检查小写变体
            proxy = os.environ.get(f'{proto}_proxy')
            if not proxy:
                # 检查大写变体
                proxy = os.environ.get(f'{proto.upper()}_PROXY')

            if proxy:
                proxies[proto] = proxy

        # 如果检测到代理，输出信息
        if proxies:
            print(f"检测到系统代理配置:")
            for proto, proxy in proxies.items():
                print(f"  {proto}: {proxy}")

        return proxies

    def download_file(self, url: str, filename: str, resume: bool = True) -> bool:
        """
        下载单个文件，支持断点续传

        Args:
            url: 文件URL
            filename: 保存的文件名
            resume: 是否启用断点续传

        Returns:
            bool: 下载是否成功
        """
        filepath = self.download_dir / filename

        # 检查文件是否已存在
        if filepath.exists() and not resume:
            print(f"文件已存在: {filename}")
            return True

        # 获取已下载的文件大小
        downloaded_size = 0
        if filepath.exists() and resume:
            downloaded_size = filepath.stat().st_size

        # 设置HTTP请求头，支持断点续传
        headers = {}
        if resume and downloaded_size > 0:
            headers['Range'] = f'bytes={downloaded_size}-'

        try:
            # 发送请求 (使用检测到的代理)
            response = requests.get(url, headers=headers, stream=True, timeout=30, proxies=self.proxies)

            # 检查是否支持断点续传
            if resume and downloaded_size > 0:
                if response.status_code == 206:  # Partial Content
                    mode = 'ab'  # 追加模式
                    print(f"断点续传: {filename} (已下载 {downloaded_size / 1024 / 1024:.2f} MB)")
                elif response.status_code == 200:
                    mode = 'wb'  # 覆盖模式
                    downloaded_size = 0
                    print(f"服务器不支持断点续传，重新下载: {filename}")
                else:
                    print(f"下载失败: {filename}, 状态码: {response.status_code}")
                    return False
            else:
                mode = 'wb'
                response.raise_for_status()

            # 获取文件总大小
            total_size = int(response.headers.get('content-length', 0))
            if mode == 'ab':
                total_size += downloaded_size

            # 使用tqdm显示进度条
            progress_bar = tqdm(
                total=total_size,
                initial=downloaded_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                desc=filename
            )

            # 下载文件
            with open(filepath, mode) as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        progress_bar.update(len(chunk))

            progress_bar.close()

            # 如果是gzip文件，自动解压
            if filename.endswith('.gz'):
                self._extract_gz(filepath)

            print(f"✓ 下载完成: {filename}")
            return True

        except requests.exceptions.RequestException as e:
            print(f"✗ 下载失败: {filename}")
            print(f"  错误信息: {str(e)}")
            return False
        except Exception as e:
            print(f"✗ 下载失败: {filename}")
            print(f"  错误信息: {str(e)}")
            return False

    def _extract_gz(self, gz_filepath: Path) -> None:
        """
        解压gzip文件

        Args:
            gz_filepath: gzip文件路径
        """
        output_filepath = gz_filepath.with_suffix('')

        # 如果解压后的文件已存在，跳过
        if output_filepath.exists():
            print(f"  解压文件已存在: {output_filepath.name}")
            return

        print(f"  正在解压: {gz_filepath.name}")
        try:
            with gzip.open(gz_filepath, 'rb') as f_in:
                with open(output_filepath, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"  ✓ 解压完成: {output_filepath.name}")
        except Exception as e:
            print(f"  ✗ 解压失败: {gz_filepath.name}")
            print(f"    错误信息: {str(e)}")

    def download_dataset(self, dataset_name: str, resume: bool = True,
                        use_multithreading: bool = True, mirror: str = 'tsinghua') -> bool:
        """
        下载指定数据集

        Args:
            dataset_name: 数据集名称 (如 'mnist')
            resume: 是否启用断点续传
            use_multithreading: 是否使用多线程下载
            mirror: 镜像源选择 ('tsinghua', 'modelscope', 'original')

        Returns:
            bool: 下载是否成功
        """
        if dataset_name not in self.DATASETS:
            print(f"错误: 不支持的数据集 '{dataset_name}'")
            print(f"支持的数据集: {list(self.DATASETS.keys())}")
            return False

        dataset_info = self.DATASETS[dataset_name]

        # 选择镜像源
        if 'mirrors' in dataset_info and mirror in dataset_info['mirrors']:
            urls = dataset_info['mirrors'][mirror]
            mirror_name = {
                'aliyun': 'AWS S3镜像(PyTorch官方)',
                'openi': 'OpenI启智社区',
                'jsdelivr': 'jsDelivr CDN',
                'original': '原始源(Yann LeCun)'
            }.get(mirror, mirror)
        else:
            urls = dataset_info['urls']
            mirror_name = '默认源'

        print(f"\n开始下载: {dataset_info['description']}")
        print(f"下载目录: {self.download_dir.absolute()}")
        print(f"镜像源: {mirror_name}")
        print(f"线程数: {self.num_threads if use_multithreading else 1}")
        print(f"断点续传: {'启用' if resume else '禁用'}")
        print(f"代理: {self.proxies if self.proxies else '不使用'}\n")

        # 创建数据集子目录
        dataset_dir = self.download_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        original_dir = self.download_dir
        self.download_dir = dataset_dir

        # 准备下载任务 - 从URL中提取真实文件名
        download_tasks = []
        for filename, url in urls.items():
            # 从URL中提取真实的文件名（如 train-images-idx3-ubyte.gz）
            real_filename = url.split('/')[-1]
            download_tasks.append((url, real_filename))

        success_count = 0
        if use_multithreading and len(download_tasks) > 1:
            # 多线程下载
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = {
                    executor.submit(self.download_file, url, filename, resume): filename
                    for url, filename in download_tasks
                }

                for future in as_completed(futures):
                    filename = futures[future]
                    try:
                        if future.result():
                            success_count += 1
                    except Exception as e:
                        print(f"下载任务失败: {filename}")
                        print(f"错误: {str(e)}")
        else:
            # 单线程下载
            for url, filename in download_tasks:
                if self.download_file(url, filename, resume):
                    success_count += 1

        # 恢复原始目录
        self.download_dir = original_dir

        # 输出下载结果
        total_count = len(download_tasks)
        print(f"\n{'='*60}")
        print(f"下载完成: {success_count}/{total_count} 个文件成功")

        if success_count == total_count:
            print(f"✓ {dataset_info['description']} 下载成功！")
            print(f"数据保存在: {dataset_dir.absolute()}")
            return True
        else:
            print(f"✗ 部分文件下载失败，请检查网络连接后重试")
            return False

    def list_datasets(self) -> None:
        """列出所有支持的数据集"""
        print("\n支持的数据集:")
        print("=" * 60)
        for name, info in self.DATASETS.items():
            print(f"  {name:15s} - {info['description']}")
            print(f"                    包含 {len(info['urls'])} 个文件")
        print("=" * 60)


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description='数据集下载器 - 支持断点续传和多线程下载',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 下载MNIST数据集 (默认使用AWS S3镜像)
  python download.py --dataset mnist

  # 使用OpenI启智社区镜像
  python download.py --dataset mnist --mirror openi

  # 使用jsDelivr CDN镜像
  python download.py --dataset mnist --mirror jsdelivr

  # 使用原始源（可能较慢）
  python download.py --dataset mnist --mirror original

  # 使用单线程下载
  python download.py --dataset mnist --no-multithread

  # 禁用断点续传
  python download.py --dataset mnist --no-resume

  # 指定下载目录和线程数
  python download.py --dataset mnist --dir ./datasets --threads 8

  # 列出所有支持的数据集
  python download.py --list

镜像源说明:
  aliyun     : AWS S3镜像 (PyTorch官方，推荐)
  openi      : OpenI启智社区镜像 (国内访问快)
  jsdelivr   : jsDelivr CDN镜像 (全球CDN加速)
  original   : 原始源 (Yann LeCun官网)
        """
    )

    parser.add_argument(
        '--dataset',
        type=str,
        help='要下载的数据集名称 (如: mnist)'
    )

    parser.add_argument(
        '--dir',
        type=str,
        default='./data',
        help='下载目录 (默认: ./data)'
    )

    parser.add_argument(
        '--threads',
        type=int,
        default=4,
        help='下载线程数 (默认: 4)'
    )

    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='禁用断点续传'
    )

    parser.add_argument(
        '--no-multithread',
        action='store_true',
        help='禁用多线程下载'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='列出所有支持的数据集'
    )

    parser.add_argument(
        '--mirror',
        type=str,
        default='aliyun',
        choices=['aliyun', 'openi', 'jsdelivr', 'original'],
        help='镜像源选择: aliyun(AWS S3,推荐), openi(启智社区), jsdelivr(CDN), original(原始源)'
    )

    args = parser.parse_args()

    # 创建下载器
    downloader = DatasetDownloader(
        download_dir=args.dir,
        num_threads=args.threads
    )

    # 列出数据集
    if args.list:
        downloader.list_datasets()
        return

    # 检查是否指定了数据集
    if not args.dataset:
        parser.print_help()
        print("\n错误: 请使用 --dataset 指定要下载的数据集，或使用 --list 查看支持的数据集")
        return

    # 下载数据集
    success = downloader.download_dataset(
        dataset_name=args.dataset,
        resume=not args.no_resume,
        use_multithreading=not args.no_multithread,
        mirror=args.mirror
    )

    if not success:
        exit(1)


if __name__ == '__main__':
    main()
