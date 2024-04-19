# bi_k_means.py 二分k_means算法相关函数
from k_means import *
from tools import *


# 数据重排函数，将每次新二分出来的其中一簇放到数据最后，确保绘图时其余簇颜色不变
def reorder_data(data, center, label, i, tmp_data, tmp_center, tmp_label):
    center_bkup = center
    cur_k = center_bkup.shape[0]
    data_remain = data[label != i]
    label_remain = label[label != i]
    new_data = np.concatenate((data_remain, tmp_data), axis=0)
    bool1 = tmp_label == 0
    bool2 = tmp_label == 1
    tmp_label[bool1] = i
    tmp_label[bool2] = cur_k
    new_label = np.concatenate((label_remain, tmp_label), axis=0)
    center_bkup[i] = tmp_center[0]
    new_center = np.concatenate((center_bkup, tmp_center[[1]]), axis=0)
    return new_data, new_center, new_label


# 二分k_means算法函数，每次迭代都对所有簇进行二分，选取整体SSE最小的划分方式
def bi_k_means(data, k, T, E, try_num):
    center, label, t = k_means_with_try(data, 2, T, E, try_num)

    cur_k = 2
    best_data = data
    data_bkup = data
    while cur_k < k:
        min_sse = float("inf")
        # print("center: ", center, "\n")
        for i in range(cur_k):
            tmp_data = data_bkup[label == i]
            tmp_center, tmp_label, t = k_means_with_try(tmp_data, 2, T, E, try_num)
            if tmp_center.shape[0] <= 1:
                continue
            else:
                new_data, new_center, new_label = reorder_data(data_bkup, center, label, i, tmp_data, tmp_center, tmp_label)
            sse = get_sse(new_data, new_label, new_center)
            # print("sse"+str(i)+": ", sse, "\n")
            if sse < min_sse:
                min_sse = sse
                # print("min_sse" + str(i) + ": ", min_sse, "\n")
                best_center = new_center
                best_label = new_label
                best_data = new_data
        center = best_center
        label = best_label
        final_data = best_data
        data_bkup = best_data
        # draw(final_data, cur_k+1, label, 1, "test" + str(cur_k))
        # pause()
        cur_k += 1
    return center, label, final_data


# 二分k_means算法函数，每次迭代都对所有簇进行二分，选取整体SSE最小的划分方式
def bi_k_means_step_draw(data, k, T, E, try_num, p_time):
    # 绘制原图
    data_bkup = data
    data_bkup = PCA(n_components=2).fit_transform(data_bkup)
    t = 0
    title = "bi_k_means  current k = 1"
    plt.title(title)
    plt.scatter(data_bkup[:, 0], data_bkup[:, 1], marker='o', c='black', s=7)
    plt.pause(p_time)

    center, label, t = k_means_with_try(data, 2, T, E, try_num)
    cur_k = 2
    sse = get_sse(data, label, center)
    title = "bi_k_means  current k = 2  SSE: " + str(sse)
    draw(data, 2, label, p_time, title)
    print("已完成2个簇的划分\n")
    best_data = data
    data_bkup = data
    while cur_k < k:
        min_sse = float("inf")
        # print("center: ", center, "\n")
        # pause()
        for i in range(cur_k):
            tmp_data = data_bkup[label == i]
            tmp_center, tmp_label, t = k_means_with_try(tmp_data, 2, T, E, try_num)
            # 将新簇重排入整体，计算整体SSE
            # print_shape("tmp_center", tmp_center)
            if tmp_center.shape[0] <= 1:
                continue
            else:
                new_data, new_center, new_label = reorder_data(data_bkup, center, label, i, tmp_data, tmp_center, tmp_label)
            sse = get_sse(new_data, new_label, new_center)
            # print("sse"+str(i)+": ", sse, "\n")
            if sse < min_sse:
                min_sse = sse
                best_center = new_center
                best_label = new_label
                best_data = new_data
        center = best_center
        label = best_label
        final_data = best_data
        data_bkup = best_data
        if cur_k == k - 1:
            title = "bi_k_means  current k = " + str(cur_k+1) + "  SSE: " + str(min_sse) + " converged"
        else:
            title = "bi_k_means  current k = " + str(cur_k + 1) + "  SSE: " + str(min_sse)
        draw(final_data, cur_k+1, label, p_time, title)
        print("已完成" + str(cur_k+1) + "个簇的划分\n")
        cur_k += 1
