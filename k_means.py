# k_means.py k_means算法相关函数

import matplotlib.pyplot as plt
import numpy as np
from numpy import nonzero
from sklearn.decomposition import PCA
from tools import *


# 计算sse函数
def get_sse(data, label, center):
    k = center.shape[0]
    sse = 0
    for i in range(k):
        data_i = data[i == label]
        dst = np.linalg.norm(data_i - center[i, :], axis=1)
        sse += np.sum(dst, axis=0)
    sse = round(sse, 2)
    return sse


# 从数据集中随机选择k个点作为初始簇心
def k_means_init_center(data, k):
    n = data.shape[0]
    if k > n:
        center = data[np.random.choice(data.shape[0], n, replace=False)]
    else:
        center = data[np.random.choice(data.shape[0], k, replace=False)]
    return center


# 计算数据与各个簇心的距离，并且返回每个样本离得最近簇序号（从0开始）
def k_means_get_label(data, center):
    dst = np.linalg.norm(data[:, np.newaxis] - center, axis=2)
    label = np.argmin(dst, axis=1)
    return label


# 根据当前的簇划分，计算新的簇心
def k_means_get_center(data, label, k):
    center = np.array([data[label == i].mean(axis=0) for i in range(k)])
    return center


# 算法迭代一次，返回迭代一次后的簇心和各点簇标签，用于可视化
def k_means_one_iter(data, label, k):
    new_center = k_means_get_center(data, label, k)
    new_label = k_means_get_label(data, new_center)
    return new_center, new_label


# k-means算法，迭代直到收敛，T为最大迭代次数，E为收敛误差
def k_means(data, k, T, E):
    n = data.shape[0]
    if n == 1:
        center = [data[0]]
        label = [0]
        t = 0
        return np.array(center), label, t
    center = k_means_init_center(data, k)
    t = 0
    while t < T:
        label = k_means_get_label(data, center)
        new_center = k_means_get_center(data, label, k)
        if np.linalg.norm(new_center - center) < E:
            center = new_center
            label = k_means_get_label(data, center)
            return center, label, t
        center = new_center
        t += 1
    label = k_means_get_label(data, center)
    return center, label, t


# 会尝试不同初始随机点的k-means算法，选取SSE最小的结果返回
def k_means_with_try(data, k, T, E, try_num):
    n = data.shape[0]
    if n == 1:
        center = [data[0]]
        label = [0]
        t = 0
        return np.array(center), label, t
    best_center, best_label, t = k_means(data, k, T, E)
    min_sse = float("inf")
    for i in range(try_num):
        tmp_center, tmp_label, t = k_means(data, k, T, E)
        tmp_sse = get_sse(data, tmp_label, tmp_center)
        if tmp_sse < min_sse:
            min_sse = tmp_sse
            best_center = tmp_center
            best_label = tmp_label
    label = best_label
    center = best_center
    return center, label, t


# 绘图函数，给出当前的簇标签和簇心，绘制相应图像
def draw(data, k, label, p_time, title):
    plt.clf()
    plt.title(title)

    data_bkup = data
    # 将属性降成2维以绘制图像
    data_bkup = PCA(n_components=2).fit_transform(data_bkup)
    # 最多15个颜色
    color = np.array(
        ["#FF0000", "#0000FF", "#00FF00", "#FFFF00", "#00FFFF", "#FF00FF", "#800000", "#008000", "#000080", "#808000",
         "#800080", "#008080", "#444444", "#FFD700", "#008080"])
    # 循换打印k个簇，每个簇使用不同的颜色
    # print("k:", k, "\n")
    for i in range(k):
        # tmp_data = data_bkup[nonzero(label == i)]
        tmp_data = data_bkup[label == i]
        # print("tmp_data", tmp_data, "\n")
        # print("tmp_data.shape", tmp_data.shape, "\n")
        plt.scatter(tmp_data[:, 0], tmp_data[:, 1], c=color[i], s=7, marker='o')
    # 因为PCA导致簇心偏移，因此需要计算新的二维簇心, 仅用于画图
        center = k_means_get_center(data_bkup, label, k)
    # 打印簇心
    # print("center", center, "\n")
    plt.scatter(center[:, 0], center[:, 1], marker='x', color='m', s=30)
    plt.pause(p_time)


# 分步绘图函数，给出当前的簇标签和簇心，分步绘制相应图像
def k_means_step_draw(data, k, T, E, p_time):
    center = k_means_init_center(data, k)
    plt.clf()
    # 绘制原图
    data_bkup = data
    data_bkup = PCA(n_components=2).fit_transform(data_bkup)
    t = 0
    title = "k-means step: " + str(t)
    plt.title(title)
    plt.scatter(data_bkup[:, 0], data_bkup[:, 1], marker='o', c='black', s=7)
    plt.pause(p_time)
    print("第" + str(t) + "次迭代已完成\n")
    label = k_means_get_label(data, center)
    sse = get_sse(data, label, center)
    t += 1
    title = "k-means step: " + str(t) + " SSE: " + str(sse)
    draw(data, k, label, p_time, title)
    print("第" + str(t) + "次迭代已完成, SSE:" + str(sse) + "\n")
    while (t < T):
        t += 1
        out = 0
        new_center, label = k_means_one_iter(data, label, k)
        sse = get_sse(data, label, center)
        if np.linalg.norm(new_center - center) < E:
            out = 1
            title = "k_means step: " + str(t) + "  SSE: " + str(sse) + " Converged"
        else:
            title = "k_means step: " + str(t) + "  SSE: " + str(sse)
        if t == T:
            title = "k_means step: " + str(t) + "  SSE: " + str(sse) + " Step num limited"
            print("超出最大步数限制！" + " 聚类结果SSE：" + str(sse) + "\n")
        plt.title(title)
        draw(data, k, label, p_time, title)
        print("第" + str(t) + "次迭代已完成, SSE:" + str(sse) + "\n")
        center = new_center
        if out:
            print("数据已收敛" + "，聚类结果SSE：" + str(sse) + "\n")
            break
