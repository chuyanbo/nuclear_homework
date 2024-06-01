import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 读取CSV文件
file_path = './data.csv'
data = pd.read_csv(file_path)
# print(data)
name_data = data.iloc[:,0]
lambda_data = data.iloc[:, 1].astype(float)
sigma_data = data.iloc[:, 2].astype(float)

# 定义初始同位素浓度 (假设单位为atoms/cm^3)
N=np.array([2.5e20,0,0,0,3.5e22,0,0,0,0])

# 定义flux,beta
flux = 1e14 # 中子通量
beta_data = lambda_data+flux*sigma_data*1e-24
# print(beta_data)
# 定义时间步长 (假设单位为d)
time_step = 1  # 一天
total_time = 365  # 一年
times = np.arange(0, total_time, time_step)

# 初始化历史数据数组
Nhistory = np.zeros((len(times), len(N)))
label = name_data

# 计算
for i, t in enumerate(times):
    minus = beta_data * N
    add = np.roll(minus, 1)
    add[0] = 0
    add[6] += minus[3]
    add[4] = 0
    minus[0] += 586.691 * flux * N[0] *1e-24
    N = (-minus + add) * time_step * 86400 + N
    Nhistory[i, :] = N

# print(N)
# 结果可视化
plt.figure(figsize=(10, 6))
for i in range(Nhistory.shape[1]):
    plt.plot(times, Nhistory[:, i], label=f"{label[i]}")

# 结果可视化
plt.xlabel('Time (days)')
plt.yscale('log')
plt.ylabel('Concentration (atoms/cm^3)')
plt.legend(label)
plt.title('N Over Time')
plt.show()
