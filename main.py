# main.py 主函数，负责用户交互逻辑
# 聚类算法作业 2331915 龚乙骁

from tools import *
import sys
from bi_k_means import *
from data_process import *
import warnings
warnings.filterwarnings("ignore")

# 当前使用的是鸢尾花数据集有n=150个样本，每个样本有fn=4个属性，已去除最后一列的标签
# 如果需要使用新的数据集请保证数据格式与iris.data一致
# 可以将数据集更换为wine.data，路径为".log("new_label:\n"+str(new_label))/dataset/wine.data"
default_data_dir = "./dataset/iris.data"
# default_data_dir = "./dataset/wq.data"  酒类质量数据集，6000+样本
# default_data_dir = "./dataset/wine.data" 酒类数据集，178样本
T = 200  # 最大迭代次数
E = 1e-5  # 迭代误差
default_p_time = 1  # 默认绘图间隔
try_num = 5  # 二分k-means算法中的k-means时尝试的次数
note = "聚类算法作业 2331915 龚乙骁\n" \
    + "当前使用的是鸢尾花数据集iris.data，该数据集有n=150个样本，每个样本有fn=4个属性，已去除最后一列的标签\n" \
    + "如果需要使用新的数据集请保证数据格式与iris.data一致\n" \
    + "可以将数据集更换为./dataset/wine.data（178样本） 或者./dataset/wq.data(6000+样本)\n" \
    + "k-means算法将随机在样本中选取k个点作为初始簇心\n" \
    + "高维数据将通过PCA降维到2维进行可视化\n"


def main():
    plt.ion()
    print(note)
    sys.stdout.flush()
    data_dir = default_data_dir
    while True:
        # 数据集选择
        input_str = "当前数据集为：" + data_dir
        print(input_str)
        input_str = "是否更换数据集？输入1是，输入0否：\n"
        b = bool_get(input_str)
        if b:
            input_str = "请输入数据集路径：\n"
            data_dir = dir_get(input_str)

        data = get_data_feature(data_dir)
        n = data.shape[0]

        if n <= 1:
            print("数据集样本数过小，请更换数据集！\n")
            continue
        # 簇心数k输入
        input_str = "请输入簇心数k(1<k<16)：\n"
        k = int_get(2, 15, input_str)
        if k > n:
            k = n
            print("检测到k大于样本数n， 已将k设置为n\n")
        # 算法选择
        input_str = "算法选择，输入1是k-means，输入0是二分k-means：\n"
        bb = bool_get(input_str)
        if bb:
            input_str = "是否观看算法过程？输入1是，输入0否：\n"
            b = bool_get(input_str)
            if b:
                input_str = "请输入迭代间隔（单位：秒）t(0.5 <= t <= 2): \n"
                p_time = float_get(0.5, 2, input_str)
                k_means_step_draw(data, k, T, E, p_time)
            else:
                center, label, t = k_means(data, k, T, E)
                # print("label.shape", label.shape, "\n")
                sse = get_sse(data, label, center)
                draw(data, k, label, default_p_time, "k_means" + " SSE: " + str(sse))
                print("图形绘制完毕，迭代次数为：" + str(t) + " 聚类结果SSE：" + str(sse) + "\n")
        else:
            input_str = "是否观看算法过程？输入1是，输入0否：\n"
            b = bool_get(input_str)
            if b:
                input_str = "请输入迭代间隔（单位：秒）t(1 <= t <= 3): \n"
                p_time = float_get(1, 3, input_str)
                bi_k_means_step_draw(data, k, T, E, try_num, p_time)
            else:
                center, label, final_data = bi_k_means(data, k, T, E, try_num)
                # print("label.shape", label.shape, "\n")
                sse = get_sse(final_data, label, center)
                draw(final_data, k, label, default_p_time, "bi_k_means" + " SSE: " + str(sse))
                print("图形绘制完毕" + "，聚类结果SSE：" + str(sse) + "\n")

        pause()
        plt.close()
        input_str = "是否退出程序？输入1退出，输入0继续：\n"
        b = bool_get(input_str)
        if b == 1:
            exit(1)
    plt.ioff()


if __name__ == "__main__":
    main()
