import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# 决策树类
class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        # 决策树最大深度，用于预剪枝
        self.max_depth = max_depth
        # 节点分裂所需最小样本数，用于预剪枝
        self.min_samples_split = min_samples_split
        self.root = None

    class Node:
        def __init__(self, is_leaf=False):
            self.is_leaf = is_leaf
            self.feature_idx = None
            self.threshold = None
            self.left = None
            self.right = None
            self.label = None

    # 计算基尼指数，衡量数据集不纯度
    def gini_index(self, y):
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return 1 - np.sum(p ** 2)

    # 寻找最佳分裂点
    def best_split(self, X, y):
        best_gini = float('inf')
        best_idx, best_thresh = None, None
        n_features = X.shape[1]
        for idx in range(n_features):
            feature = X[:, idx]
            sorted_idx = np.argsort(feature)
            sorted_X, sorted_y = feature[sorted_idx], y[sorted_idx]
            for i in range(1, len(sorted_X)):
                thresh = (sorted_X[i - 1] + sorted_X[i]) / 2
                left_y = sorted_y[:i]
                right_y = sorted_y[i:]
                gini = (len(left_y) * self.gini_index(left_y) + len(right_y) * self.gini_index(right_y)) / len(y)
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thresh = thresh
        return best_idx, best_thresh, best_gini

    # 递归构建决策树
    def build_tree(self, X, y, depth=0):
        if len(np.unique(y)) == 1 or len(X) < self.min_samples_split or (self.max_depth and depth >= self.max_depth):
            node = self.Node(is_leaf=True)
            node.label = np.bincount(y).argmax()
            return node
        idx, thresh, _ = self.best_split(X, y)
        if idx is None:
            node = self.Node(is_leaf=True)
            node.label = np.bincount(y).argmax()
            return node
        left_mask = X[:, idx] <= thresh
        node = self.Node()
        node.feature_idx = idx
        node.threshold = thresh
        node.left = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self.build_tree(X[~left_mask], y[~left_mask], depth + 1)
        return node

    # 训练决策树
    def fit(self, X, y):
        self.root = self.build_tree(X, y)

    # 对单个样本进行预测
    def predict_sample(self, x, node):
        if node.is_leaf:
            return node.label
        if x[node.feature_idx] <= node.threshold:
            return self.predict_sample(x, node.left)
        else:
            return self.predict_sample(x, node.right)

    # 对多个样本进行预测
    def predict(self, X):
        return np.array([self.predict_sample(x, self.root) for x in X])


# 随机森林类
class RandomForest:
    def __init__(self, n_estimators=50, max_depth=None, min_samples_split=2, max_features=None):
        # 随机森林中决策树的数量
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        # 每次分裂时随机选择的特征数量
        self.max_features = max_features
        self.trees = []

    # 自助采样，有放回抽取样本
    def bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[idx], y[idx]

    # 训练随机森林
    def fit(self, X, y):
        n_features = X.shape[1]
        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))
        for _ in range(self.n_estimators):
            X_samp, y_samp = self.bootstrap_sample(X, y)
            feat_idx = np.random.choice(n_features, size=self.max_features, replace=False)
            X_samp_sub = X_samp[:, feat_idx]
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_samp_sub, y_samp)
            tree.feat_idx_used = feat_idx
            self.trees.append(tree)

    # 对多个样本进行预测
    def predict(self, X):
        predictions = []
        for tree in self.trees:
            feat_idx = tree.feat_idx_used
            X_sub = X[:, feat_idx]
            pred = tree.predict(X_sub)
            predictions.append(pred)
        predictions = np.array(predictions).T
        return np.array([np.bincount(pred).argmax() for pred in predictions])


if __name__ == "__main__":
    # 加载 MNIST 数据集
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # 数据预处理，展平并归一化
    x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
    x_test = x_test.reshape(-1, 784).astype(np.float32) / 255.0

    # 数据标准化
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # 创建并训练随机森林模型
    rf = RandomForest(n_estimators=50, max_depth=10, min_samples_split=10)
    rf.fit(x_train, y_train)

    # 预测并计算准确率
    y_pred = rf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")