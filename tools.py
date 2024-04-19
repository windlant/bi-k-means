# tools.py 工具函数，用于用户输入或者调试


# 按回车键继续函数
def pause():
    input("按回车键继续...\n")


# 整数输入函数，确保用户输入的是范围内的整数
def int_get(low, up, input_str):
    while True:
        k = input(input_str)
        try:
            k = int(k)
        except:
            print("输入错误！请重新输入\n")
            continue
        if k < low or k > up:
            print("输入错误！请重新输入\n")
            continue
        return k


# 布尔数输入函数，确保用户输入的是布尔数
def bool_get(input_str):
    while True:
        k = input(input_str)
        try:
            k = int(k)
        except:
            print("输入错误！请重新输入\n")
            continue
        if k != 0 and k != 1:
            print("输入错误！请重新输入\n")
            continue
        return k


# 浮点输入函数，确保用户输入的是范围内的整数
def float_get(low, up, input_str):
    while True:
        f = input(input_str)
        try:
            f = float(f)
        except:
            print("输入错误！请重新输入\n")
            continue
        if f < low or f > up:
            print("输入错误！请重新输入\n")
            continue
        return f


# 路径输入函数，确保用户输入的是正确的数据集路径
def dir_get(input_str):
    while True:
        d = input(input_str)
        try:
            f = open(d)
        except:
            print("输入路径无效！请重新输入\n")
            continue
        f.close()
        return d


# 临时写入日志函数
def log(log_str):
    f = open("./log.txt", "w")
    f.write(str(log_str))
    f.close()


# 打印数组形状函数
def print_shape(name_str, name):
    print(name_str+".shape：", name.shape, "\n")
