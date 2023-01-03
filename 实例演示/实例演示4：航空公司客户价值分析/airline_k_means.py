import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans  # 导入kmeans算法

airline_scale = np.load('airline_scale.npz')['arr_0']
k = 5  # 确定聚类中心数
# 构建模型
kmeans_model = KMeans(n_clusters=k)
fit_kmeans = kmeans_model.fit(airline_scale)  # 模型训练

# kmeans_model.cluster_centers_  # 查看聚类中心
# kmeans_model.labels_ # 查看样本的类别标签
print('样本的类别标签：', kmeans_model.labels_)
print('样本的类别标签类型：', type(kmeans_model.labels_))
# 统计不同类别样本的数目

# 易错：只用pandas中Series类型才有value_counts()属性
r1 = pd.Series(kmeans_model.labels_).value_counts()
print('最终每个类别的数目为：\n', r1)

# 做可视化：
plt.figure(figsize=(6, 6))
L = 5
angles = np.linspace(0, 2 * np.pi, L, endpoint=False)
labels = ['L', 'F', 'R', 'C', 'M']
data = kmeans_model.cluster_centers_
# 闭合曲线：
print(data)
angles = np.concatenate((angles, [angles[0]]))
# 为了形成闭合，把二维数组每一行下标为0的数拼接到列,
# 二维数组与二维数组拼接，
# 所以reshape成二维的，且变成五列一行
data = np.concatenate((data, data[:, 0].reshape(5, 1)), axis=1).T
# print('angles',angles)
print('data', data)
# 绘图：
plt.polar(angles, data)
# plt.xticks(angles, labels)
plt.show()
