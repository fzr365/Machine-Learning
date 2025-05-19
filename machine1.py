# 导入必要库
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter
import numpy as np

# 加载数据集
wine = load_wine()
X, y = wine.data, wine.target

# 划分训练集和测试集（70%训练，30%测试）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 特征标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 手动实现KNN类
class KNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X_train, y_train):
        #存储训练数据
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        #预测测试集
        predictions = []
        for x in X_test:
            # 计算欧氏距离
            distances = [np.sqrt(np.sum((x - x_train) ** 2)) for x_train in self.X_train]

            # 获取距离最近的k个样本的索引
            k_indices = np.argsort(distances)[:self.k]

            # 统计k个邻居的类别标签
            k_labels = self.y_train[k_indices]
            most_common = Counter(k_labels).most_common(1)
            predictions.append(most_common[0][0])
        return np.array(predictions)


# 初始化KNN（k=5）
knn = KNN(k=5)
knn.fit(X_train, y_train)

# 预测并计算准确率
y_pred = knn.predict(X_test)
accuracy = np.sum(y_pred == y_test) / len(y_test)
print(f"准确率: {accuracy:.2f}\n")

# 初始化存储指标的列表
precisions = []
recalls = []
f1s = []
class_names = wine.target_names
class_counts = np.bincount(y_test)

# 计算每个类别的指标
for i in range(len(class_names)):
    true_class = y_test == i
    pred_class = y_pred == i
    tp = np.sum(true_class & pred_class)
    fp = np.sum(pred_class) - tp
    fn = np.sum(true_class) - tp

    # 计算指标
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    # 存储结果
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)

    # 打印类别指标
    print(f"{class_names[i]} 类别:")
    print(f"  Precision: {precision:.2f}")
    print(f"  Recall:    {recall:.2f}")
    print(f"  F1-Score:  {f1:.2f}")
    print(f"  Support:   {class_counts[i]}\n")

# 计算加权平均
weights = class_counts
weighted_precision = sum(p * w for p, w in zip(precisions, weights)) / sum(weights)
weighted_recall = sum(r * w for r, w in zip(recalls, weights)) / sum(weights)
weighted_f1 = sum(f * w for f, w in zip(f1s, weights)) / sum(weights)

# 打印最终报告
print("=" * 50)
print("加权平均指标（基于样本数）:")
print(f"Precision: {weighted_precision:.2f}")
print(f"Recall:    {weighted_recall:.2f}")
print(f"F1-Score:  {weighted_f1:.2f}")
print(f"Support:   {len(y_test)}")