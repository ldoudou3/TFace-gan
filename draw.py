import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import torch.nn.functional as F

# 配置输入特征文件路径和输出目录
# embedding_path = "/mnt/nas/dhb/dhb3/CVPR24_FRCSyn_ADMIS/dataset/CASIA/generator_embeddings/generator_embed-24epoch-tiny判别器.npy"
# output_dir = "/home/dhb/dhb3/TFace/tasks/gan/History/history3-tiny-大小合适-防不住"
embedding_path = "/mnt/nas/dhb/dhb3/CVPR24_FRCSyn_ADMIS/dataset/CASIA/IR50_embeddings/纯IR50_embed-24epoch_490623*512.npy"
output_dir = "/home/dhb/dhb3/TFace/tasks/IR50/output-sample"
os.makedirs(output_dir, exist_ok=True)

# 检查特征文件是否存在
if not os.path.exists(embedding_path):
    raise FileNotFoundError(f"Embedding file not found at {embedding_path}")

# 加载特征向量
embeddings = np.load(embedding_path)
print(f"Loaded embeddings with shape: {embeddings.shape}")

# 生成随机高斯噪声，形状与特征向量相同
random_gaussian = np.random.randn(490623, 512)

# 将随机高斯噪声转换为 PyTorch tensor 并进行 L2 归一化
random_gaussian_tensor = torch.tensor(random_gaussian, dtype=torch.float32)
random_gaussian_normalized = F.normalize(random_gaussian_tensor, p=2, dim=1).numpy()


def sample_data(data, num_samples=50000, random_state=42):
    np.random.seed(random_state)
    indices = np.random.choice(data.shape[0], size=num_samples, replace=False)
    return data[indices]

# 采样特征向量和随机高斯噪声
num_samples = 3000  # 设置采样数量
sampled_embeddings = sample_data(embeddings, num_samples=num_samples)
sampled_gaussian = sample_data(random_gaussian_normalized, num_samples=num_samples)


# 随机采样函数

# 函数：绘制单个维度的分布图
def plot_dimension_distribution(dimension_data, title, xlabel, ylabel, output_file, color='blue', linestyle='-'):
    plt.figure(figsize=(10, 6))
    density, bins = np.histogram(dimension_data, bins=50, density=True)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    plt.plot(bin_centers, density, label=title, color=color, linestyle=linestyle, linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

# 函数：绘制特征与高斯噪声对比分布图
def plot_feature_vs_gaussian(feature_data, gaussian_data, dimension, output_file):
    # 计算特征向量的直方图数据
    feature_density, feature_bins = np.histogram(feature_data, bins=50, density=True)
    feature_bin_centers = 0.5 * (feature_bins[1:] + feature_bins[:-1])
    
    # 计算高斯噪声的直方图数据
    gaussian_density, gaussian_bins = np.histogram(gaussian_data, bins=50, density=True)
    gaussian_bin_centers = 0.5 * (gaussian_bins[1:] + gaussian_bins[:-1])
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(feature_bin_centers, feature_density, label="Feature Data", color='blue', linewidth=2)
    plt.plot(gaussian_bin_centers, gaussian_density, label="Random Gaussian", color='red', linestyle='--', linewidth=2)
    plt.title(f'Dimension {dimension} Distribution (Feature vs Gaussian)')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

# 绘制01开头的特征向量前10个维度分布
def plot_feature_only_top_10():
    for i in range(10):
        dimension_data = sampled_embeddings[:, i]
        output_file = os.path.join(output_dir, f"01_feature_dimension_{i}.png")
        plot_dimension_distribution(dimension_data, f"Feature Dimension {i}", "Value", "Density", output_file, color='blue')

# 绘制02开头的高斯噪声前10个维度分布
def plot_gaussian_only_top_10():
    for i in range(10):
        gaussian_data = sampled_gaussian[:, i]
        output_file = os.path.join(output_dir, f"02_gaussian_dimension_{i}.png")
        plot_dimension_distribution(gaussian_data, f"Gaussian Dimension {i}", "Value", "Density", output_file, color='red')

# 绘制03开头的特征与高斯噪声对比的前10个维度分布
def plot_feature_vs_gaussian_top_10():
    for i in range(10):
        feature_data = sampled_embeddings[:, i]
        gaussian_data = sampled_gaussian[:, i]
        output_file = os.path.join(output_dir, f"03_feature_vs_gaussian_dimension_{i}.png")
        plot_feature_vs_gaussian(feature_data, gaussian_data, i, output_file)

# 绘制03开头的特征与高斯噪声对比的前10个维度分布,同时有量化数值
def plot_feature_vs_gaussian_with_statistics_10():
    for i in range(10):
        feature_data = sampled_embeddings[:, i]
        gaussian_data = sampled_gaussian[:, i]
        output_file = os.path.join(output_dir, f"04_feature_vs_gaussian_dimension_{i}.png")
        plot_feature_vs_gaussian_with_statistics(feature_data, gaussian_data, i, output_file)

# 修改后的绘图函数，显示 Kolmogorov-Smirnov 检验、偏度、峰度以及解释性文字
# 

# 修改后的绘图函数，显示 Kolmogorov-Smirnov 检验、偏度、峰度以及解释性文字（同时包括高斯向量量化值）
def plot_feature_vs_gaussian_with_statistics(feature_data, gaussian_data, dimension, output_file):
    from scipy.stats import kurtosis, skew, ks_2samp

    # 计算实际数据的直方图数据
    feature_density, feature_bins = np.histogram(feature_data, bins=50, density=True)
    feature_bin_centers = 0.5 * (feature_bins[1:] + feature_bins[:-1])
    
    # 计算高斯噪声的直方图数据
    gaussian_density, gaussian_bins = np.histogram(gaussian_data, bins=50, density=True)
    gaussian_bin_centers = 0.5 * (gaussian_bins[1:] + gaussian_bins[:-1])
    
    # 计算 KS 检验
    ks_stat_feature, p_value_feature = ks_2samp(feature_data, gaussian_data)
    ks_stat_gaussian, p_value_gaussian = ks_2samp(gaussian_data, gaussian_data)  # 理论上，高斯数据对比自身的KS应很小

    # 计算偏度和峰度（Fisher=False 返回原始峰度）
    feature_skewness = skew(feature_data)
    feature_kurtosis = kurtosis(feature_data, fisher=False)

    gaussian_skewness = skew(gaussian_data)
    gaussian_kurtosis = kurtosis(gaussian_data, fisher=False)
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(feature_bin_centers, feature_density, label="Feature Data", color='blue', linewidth=2)
    plt.plot(gaussian_bin_centers, gaussian_density, label="Random Gaussian", color='red', linestyle='--', linewidth=2)
    plt.title(f'Dimension {dimension} Distribution (Feature vs Gaussian)')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()

    # 在图上打印统计值和解释说明
    text_x = 0.05  # x 坐标
    text_y = 0.85  # y 坐标，从顶部开始打印

    # 特征数据统计值
    plt.gcf().text(text_x, text_y, f"Feature KS Statistic: {ks_stat_feature:.4f}", fontsize=10, transform=plt.gca().transAxes)
    plt.gcf().text(text_x, text_y - 0.05, f"Feature p-value: {p_value_feature:.4f} (p ≥ 0.05 indicates close to normal)", fontsize=10, transform=plt.gca().transAxes)
    plt.gcf().text(text_x, text_y - 0.10, f"Feature Skewness: {feature_skewness:.4f} (|Skew| < 0.5 for Gaussian)", fontsize=10, transform=plt.gca().transAxes)
    plt.gcf().text(text_x, text_y - 0.15, f"Feature Kurtosis: {feature_kurtosis:.4f} (Kurt ≈ 3 for Gaussian)", fontsize=10, transform=plt.gca().transAxes)

    # 高斯数据统计值
    text_y = text_y - 0.25  # 在下方打印高斯数据统计值
    plt.gcf().text(text_x, text_y, f"Gaussian KS Statistic: {ks_stat_gaussian:.4f}", fontsize=10, transform=plt.gca().transAxes)
    plt.gcf().text(text_x, text_y - 0.05, f"Gaussian p-value: {p_value_gaussian:.4f} (p ≥ 0.05 indicates close to normal)", fontsize=10, transform=plt.gca().transAxes)
    plt.gcf().text(text_x, text_y - 0.10, f"Gaussian Skewness: {gaussian_skewness:.4f} (|Skew| < 0.5 for Gaussian)", fontsize=10, transform=plt.gca().transAxes)
    plt.gcf().text(text_x, text_y - 0.15, f"Gaussian Kurtosis: {gaussian_kurtosis:.4f} (Kurt ≈ 3 for Gaussian)", fontsize=10, transform=plt.gca().transAxes)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_top_10_histograms_only():
    plt.figure(figsize=(12, 8))
    for i in range(10):
        plt.subplot(5, 2, i+1)
        plt.hist(sampled_embeddings[:, i], bins=50, density=True, alpha=0.6, color='blue', label=f'Dimension {i} Actual')
        plt.title(f'Dimension {i}')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05_top_10_dimensions_histograms_only.png'))
    plt.show()

def plot_correlation_matrix():
    # 计算实际数据的相关性矩阵
    correlation_matrix = np.corrcoef(embeddings, rowvar=False)
    # 计算随机高斯数据的相关性矩阵
    random_correlation_matrix = np.corrcoef(random_gaussian, rowvar=False)

    # 找到两个矩阵的全局最小值和最大值
    global_min = min(correlation_matrix.min(), random_correlation_matrix.min())
    global_max = max(correlation_matrix.max(), random_correlation_matrix.max())

    # 绘制实际数据的相关性矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', cbar=True, vmin=global_min, vmax=global_max)
    plt.title("Correlation Matrix - Actual Data")
    plt.xlabel("Dimensions")
    plt.ylabel("Dimensions")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Correlation Matrix of Embeddings.png'))
    plt.show()

    # 绘制随机高斯数据的相关性矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(random_correlation_matrix, annot=False, cmap='coolwarm', cbar=True, vmin=global_min, vmax=global_max)
    plt.title("Correlation Matrix - Random Gaussian Data")
    plt.xlabel("Dimensions")
    plt.ylabel("Dimensions")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Correlation Matrix of Gaussian.png'))
    plt.show()


# 生成图像
print("Generating plots...")
# plot_feature_only_top_10()
# plot_gaussian_only_top_10()
# plot_feature_vs_gaussian_top_10()
# plot_feature_vs_gaussian_with_statistics_10()
# plot_top_10_histograms_only()
plot_correlation_matrix()
print(f"Plots have been saved to {output_dir}.")
