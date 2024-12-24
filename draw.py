import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 特征文件路径
embedding_path = "/mnt/nas/dhb/dhb3/CVPR24_FRCSyn_ADMIS/dataset/CASIA/IR50_embeddings/纯IR50_embed-24epoch.npy"
output_dir = "/home/dhb/dhb3/TFace/tasks/IR50/output"
os.makedirs(output_dir, exist_ok=True)

# 检查文件是否存在
if not os.path.exists(embedding_path):
    raise FileNotFoundError(f"Embedding file not found at {embedding_path}")

# 加载特征数据
embeddings = np.load(embedding_path)

# 打印特征数据的基本信息
print(f"Loaded embeddings with shape: {embeddings.shape}")

# 计算实际数据的相关性矩阵
correlation_matrix = np.corrcoef(embeddings, rowvar=False)

# 随机高斯数据（具有相同形状的标准正态分布）
random_gaussian = np.random.randn(*embeddings.shape)

# 计算随机高斯数据的相关性矩阵
random_correlation_matrix = np.corrcoef(random_gaussian, rowvar=False)

# 绘制相关性矩阵函数
def plot_correlation_matrix(corr_matrix, title, output_file):
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', cbar=True)
    plt.title(title)
    plt.xlabel("Dimensions")
    plt.ylabel("Dimensions")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

# 计算实际数据的直方图
def plot_histograms(dim_values, random_values, dimension, output_file):
    plt.figure(figsize=(10, 6))
    plt.hist(dim_values, bins=50, density=True, alpha=0.6, color='blue', label='Actual Data')
    plt.hist(random_values, bins=50, density=True, alpha=0.6, color='red', label='Random Gaussian')
    plt.title(f'Dimension {dimension} Distribution (Actual vs Random Gaussian)')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

# 计算前10个维度的直方图（仅实际数据）
def plot_top_10_histograms_only():
    plt.figure(figsize=(12, 8))
    for i in range(10):
        plt.subplot(5, 2, i+1)
        plt.hist(embeddings[:, i], bins=50, density=True, alpha=0.6, color='blue', label=f'Dimension {i} Actual')
        plt.title(f'Dimension {i}')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_10_dimensions_histograms_only.png'))
    plt.show()

# 绘制前10个维度的直方图（每个维度一张图，包括高斯数据对比）
def plot_top_10_histograms_with_gaussian():
    for i in range(10):
        plot_histograms(embeddings[:, i], random_gaussian[:, i], i, os.path.join(output_dir, f'dimension_{i}_histogram_actual_vs_random.png'))

# 为每个维度单独生成一张直方图图像（仅实际数据）
def plot_separate_top_10_histograms():
    for i in range(10):
        dimension_values = embeddings[:, i]

        # 计算实际分布的直方图数据
        density_original, bins_original = np.histogram(dimension_values, bins=50, density=True)
        bin_centers_original = 0.5 * (bins_original[1:] + bins_original[:-1])

        # 绘制实际分布的折线图
        plt.figure(figsize=(10, 6))
        plt.plot(bin_centers_original, density_original, label=f'Dimension {i} Actual', color='blue', linewidth=2)
        plt.title(f'Dimension {i} Distribution (Actual Data Only)')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()

        # 保存每个维度的图像到文件
        output_file = os.path.join(output_dir, f'dimension_{i}_histogram_actual.png')
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

# 获取用户输入并选择性生成图像
print("Select the type of plots you want to generate (comma-separated):")
print("1. Actual Data Correlation Matrix")
print("2. Random Gaussian Correlation Matrix")
print("3. Top 10 Dimensions Actual Data Histograms")
print("4. Top 10 Dimensions Actual vs Random Gaussian Histograms")
print("5. Separate Histogram of Top 10 Dimensions (Each in Separate Image)")

# user_input = input("Enter your choices (e.g., 1,3,5): ")
user_input = "1,2,3,4,5"  
user_choices = [int(choice.strip()) for choice in user_input.split(',')]

# 根据用户选择生成对应的图像
if 1 in user_choices:
    output_path = os.path.join(output_dir, f'actual_data_correlation_matrix.png')
    plot_correlation_matrix(correlation_matrix, "Correlation Matrix - Actual Data", output_path)

if 2 in user_choices:
    output_path = os.path.join(output_dir, f'random_gaussian_correlation_matrix.png')
    plot_correlation_matrix(random_correlation_matrix, "Correlation Matrix - Random Gaussian Data", output_path)

if 3 in user_choices:
    plot_top_10_histograms_only()

if 4 in user_choices:
    plot_top_10_histograms_with_gaussian()

if 5 in user_choices:
    plot_separate_top_10_histograms()

print("Selected plots have been generated and saved.")
