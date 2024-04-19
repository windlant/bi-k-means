import numpy as np
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
aa = np.array([1, 2, 3, 1, 5, 6, 7])
c = np.array(range(3)) != 2
cc = np.array(range(7)) != 5
line1 = a[1]
line2 = a[1, :]
b = [1, 5, 7]
a = np.concatenate((a, [a[2]]), axis=0)
print("a", a)
print("np.argmin(aa)", np.argmin(aa), "\n")
# print("line1.shape", line1.shape)
# print("line2", line2)
# print("line2.shape", line2.shape)
