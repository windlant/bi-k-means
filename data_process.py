# data_process.py 数据处理函数，主要负责读取数据集
import numpy as np


# 使用的是UCI上的鸢尾花数据集，4维，150例，绘图时使用PCA算法降维成2维
# 读取数据集，返回属性数据矩阵（去掉最后一列label）
def get_data_feature(datadir):
    f = open(datadir)
    datastr = f.read()
    f.close()
    dataraw = datastr.strip('\n').split('\n')
    row = len(dataraw)
    col = len(dataraw[0].strip('\n').split(','))
    datafeature = np.zeros((row, col-1), dtype=float)
    count = 0
    for line in dataraw:
        tmp = line.strip('\n').split(',')
        datafeature[count :] = list(map(float, tmp[0:col-1]))
        count += 1
    return datafeature
