import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score




# 加载数据
iris = load_iris()
X = iris.data
y = iris.target

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ==================== 手动实现均值漂移算法 ====================
def euclidean_distance(x1, x2):
    """计算两点间的欧氏距离"""
    return np.sqrt(np.sum((x1 - x2) ** 2))


def gaussian_kernel(distance, bandwidth):
    """高斯核函数，用于计算权重"""
    return (1 / (bandwidth * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (distance / bandwidth) ** 2)


def mean_shift(X, bandwidth=0.8, min_shift=1e-3, max_iter=300):
    """
    手动实现均值漂移聚类算法

    参数:
    X: 输入数据，形状为(n_samples, n_features)
    bandwidth: 核函数带宽
    min_shift: 停止迭代的最小移动距离
    max_iter: 最大迭代次数

    返回:
    labels: 聚类标签
    centers: 聚类中心
    """
    n_samples, n_features = X.shape
    # 初始化所有点为候选中心
    centers = np.copy(X)
    shifted = np.ones(n_samples, dtype=bool)
    iteration = 0

    while np.any(shifted) and iteration < max_iter:
        iteration += 1
        for i in range(n_samples):
            if not shifted[i]:
                continue

            # 当前点
            point = centers[i]
            # 计算所有点到当前点的距离
            distances = np.array([euclidean_distance(point, x) for x in X])
            # 应用核函数计算权重
            weights = gaussian_kernel(distances, bandwidth)
            # 计算加权均值
            numerator = np.sum(X * weights[:, np.newaxis], axis=0)
            denominator = np.sum(weights)
            new_point = numerator / denominator

            # 计算移动距离
            shift_distance = euclidean_distance(point, new_point)
            centers[i] = new_point

            # 判断是否继续移动
            if shift_distance < min_shift:
                shifted[i] = False

    # 合并相近的聚类中心
    merged_centers = []
    labels = np.full(n_samples, -1, dtype=int)
    current_label = 0

    for i in range(n_samples):
        center = centers[i]
        # 检查是否已存在相近的中心
        if len(merged_centers) == 0:
            merged_centers.append(center)
            labels[i] = current_label
            current_label += 1
        else:
            distances = np.array([euclidean_distance(center, c) for c in merged_centers])
            min_distance = np.min(distances)
            if min_distance < bandwidth / 2:  # 阈值可调整
                labels[i] = np.argmin(distances)
            else:
                merged_centers.append(center)
                labels[i] = current_label
                current_label += 1

    return labels, np.array(merged_centers)


# 应用均值漂移算法
ms_labels, ms_centers = mean_shift(X_scaled, bandwidth=0.8)


# ==================== 手动实现DBSCAN算法 ====================
def dbscan(X, eps=0.6, min_samples=5):
    """
    手动实现DBSCAN聚类算法

    参数:
    X: 输入数据，形状为(n_samples, n_features)
    eps: 邻域半径
    min_samples: 形成核心点所需的最小样本数

    返回:
    labels: 聚类标签，-1表示噪声点
    """
    n_samples = X.shape[0]
    labels = np.full(n_samples, -1, dtype=int)  # 初始化为-1（噪声）
    core_points = np.zeros(n_samples, dtype=bool)
    cluster_id = 0

    # 计算邻域
    neighbors = []
    for i in range(n_samples):
        # 计算点i到所有点的距离
        distances = np.array([euclidean_distance(X[i], x) for x in X])
        # 找出eps邻域内的点
        i_neighbors = np.where(distances <= eps)[0]
        neighbors.append(i_neighbors)

        # 判断是否为核心点
        if len(i_neighbors) >= min_samples:
            core_points[i] = True

    # 扩展聚类
    for i in range(n_samples):
        if labels[i] != -1 or not core_points[i]:
            continue  # 已分类或非核心点

        # 创建新聚类
        labels[i] = cluster_id

        # 扩展当前聚类
        seeds = list(neighbors[i])
        seeds.remove(i)  # 移除自身

        j = 0
        while j < len(seeds):
            current_point = seeds[j]

            # 如果是噪声点，归类到当前聚类
            if labels[current_point] == -1:
                labels[current_point] = cluster_id

            # 如果是核心点，添加其邻域点到种子集
            if core_points[current_point]:
                for neighbor in neighbors[current_point]:
                    if labels[neighbor] == -1 and neighbor not in seeds:
                        seeds.append(neighbor)
            j += 1

        # 创建下一个聚类
        cluster_id += 1

    return labels


# 应用DBSCAN算法
dbscan_labels = dbscan(X_scaled, eps=0.6, min_samples=5)

# ==================== 可视化与评估 ====================
# PCA降维用于可视化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


# 绘制聚类结果
def plot_clusters(title, labels):
    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)

    for label in unique_labels:
        if label == -1:  # 噪声点
            plt.scatter(X_pca[labels == label, 0], X_pca[labels == label, 1],
                        c='black', marker='x', s=50, label='Noise')
        else:
            plt.scatter(X_pca[labels == label, 0], X_pca[labels == label, 1],
                        label=f'Cluster {label}', alpha=0.7)

    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.show()


# 计算评估指标
def print_metrics(y_true, y_pred, algorithm_name):
    ari = adjusted_rand_score(y_true, y_pred)
    # 计算轮廓系数时需要排除噪声点
    valid_indices = y_pred != -1
    if len(np.unique(y_pred[valid_indices])) > 1:
        sil = silhouette_score(X_scaled[valid_indices], y_pred[valid_indices])
    else:
        sil = "N/A (single cluster)"

    print(f"{algorithm_name}聚类结果:")
    print(f"调整兰德指数(ARI): {ari:.4f}")
    print(f"轮廓系数: {sil}\n")


# 输出结果
print_metrics(y, ms_labels, "均值漂移")
plot_clusters("均值漂移聚类结果", ms_labels)

print_metrics(y, dbscan_labels, "DBSCAN")
plot_clusters("DBSCAN聚类结果", dbscan_labels)