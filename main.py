import numpy as np 
from reader import read_h5_database
from solver import power_solver,flux_solver

def init_N_vec(N0,nuclei_key,n_dim):
    N = np.zeros((n_dim))
    for key in N0.keys():
        index_N = nuclei_key[key]
        number_N = N0[key]
        N[index_N] = number_N
    return N

def plot_result(t,N,n_dim):
    import matplotlib.pyplot as plt
    for i in range(n_dim):
        plt.plot(t, N[i], label=f'N{i}')
    plt.xlabel('T(Time)')
    plt.ylabel('N(Nuclei number)')
    plt.yscale('log')
    plt.legend()
    plt.title('Solution of Nuclei Chain')
    plt.show()

def Burn_up_chain_calculator(reactor_type,solver_type, chain_key ,file_path,
           t_min,t_max,t_interval,
           temperature,N0,flux0,power0,init_flue=1e23):
    """
    parameter:
     - reactor_type (string) 反应堆运行状态 
        - 'flux' 定通量
        - 'power' 定功率
     - solver_type (string) 微分方程求解方式（见下）
     - chain_key (string) 反应链类别
     - file_path (string) 衰变链数据库路径
     - t_min (float64) 模拟计算时间下限
     - t_max (float64) 模拟计算时间上限
     - t_interval (float64) 模拟计算时间间隔
     - temperature (float64) 反应堆温度
     - N0 (dictionary) 初始衰变核素 e.g. N0 = {"U238":1e23,"U235":1e23} 注意：核素一定要在chain中
     定通量
     - flux0 (float64) 初始通量（定通量）
     定功率
     - power0 (float64) 初始通量（定功率）
     - flux0 (float64) 初始燃料（定功率）
    solver:
     - A 粒子数变化率矩阵
     - N 粒子数初始值
     - t 模拟时间点
     
    求解方式：
    flux: 定通量情况
        'analytical':对角化解析求解(精度低)
        'BDF':后向差分有限元
        'FDF':前向差分有限元
        'standard':scipy自带标准求解器(慢)
        'rk4':四阶龙格库塔方法
        'taylor':矩阵泰勒展开方法
        'krylov':Krylov子空间法
        'CRAM':切比雪夫有理近似法
        'LPAM':拉盖尔多项式法
    power: 定功率情况
        'BDF':后向差分有限元
        'FDF':前向差分有限元
        'rk4':四阶龙格库塔方法
        'taylor':矩阵泰勒展开方法
        'krylov':Krylov子空间法
        'CRAM':切比雪夫有理近似法
        'LPAM':拉盖尔多项式法
        
    output输出:
    results: 
     - shape->[n,time_step]
     - 其中 n 为核素种类数， time_step 为模拟的时间步数
     - results[i]为第i种核素随时间数量变化的数据
    nuclei_names
     - 列表，用于查询第i个核素的种类名称 nuclei_names[i]
    """
    t_sequence = np.arange(t_min,t_max,t_interval)
    dA , auA , nuclei_key , nuclei_names  = read_h5_database(file_path,chain_key,temperature)
    n_dim = len(dA)
    N_init = init_N_vec(N0,nuclei_key,n_dim)
    if reactor_type == 'flux':
        A = dA + flux0 * auA * 1e-28
        results = flux_solver(solver_type,A,N_init,t_sequence)
        #  result1 = flux_solver("analytical",A,N_init,t_sequence)
        #  plot_result(t_sequence,result1,n_dim)
        #  result2 = flux_solver("BDF",A,N_init,t_sequence)
        #  plot_result(t_sequence,result2,n_dim)
        #  result3 = flux_solver("standard",A,N_init,t_sequence)
        #  plot_result(t_sequence,result3,n_dim)
        #  result4 = flux_solver("rk4",A,N_init,t_sequence)
        #  plot_result(t_sequence,result4,n_dim)
        #  result5 = flux_solver("taylor",A,N_init,t_sequence)
        #  plot_result(t_sequence,result5,n_dim)
        #  result6 = flux_solver("krylov",A,N_init,t_sequence)
        #  plot_result(t_sequence,result6,n_dim)
        #  result7 = flux_solver("CRAM",A,N_init,t_sequence)
        #  plot_result(t_sequence,result7,n_dim)
        #  result8 = flux_solver("LPAM",A,N_init,t_sequence)
        #  plot_result(t_sequence,result8,n_dim)
        #  result9= flux_solver("FDF",A,N_init,t_sequence)
        #  plot_result(t_sequence,result9,n_dim)
    if reactor_type == 'power':
        def A_func(t):
            return  dA + flux0 * auA * 1e-28 * np.exp(-t*(1e-13))
        results = power_solver(solver_type,A_func,N_init,t_sequence)
        result2 = power_solver("BDF",A_func,N_init,t_sequence)
        plot_result(t_sequence,result2,n_dim)
        result4 = power_solver("rk4",A_func,N_init,t_sequence)
        plot_result(t_sequence,result4,n_dim)
        result5 = power_solver("taylor",A_func,N_init,t_sequence)
        plot_result(t_sequence,result5,n_dim)
        result6 = power_solver("krylov",A_func,N_init,t_sequence)
        plot_result(t_sequence,result6,n_dim)
        result7 = power_solver("CRAM",A_func,N_init,t_sequence)
        plot_result(t_sequence,result7,n_dim)
        result8 = power_solver("LPAM",A_func,N_init,t_sequence)
        plot_result(t_sequence,result8,n_dim)
        result9= power_solver("FDF",A_func,N_init,t_sequence)
        plot_result(t_sequence,result9,n_dim)
    return results , nuclei_names
         
    
    

if __name__ == '__main__':
    
    reactor_type = "power"
    solver_type = "FDF"
    chain_key = "chain_1"
    file_path = "./data/data.h5"
    t_min = 0
    t_max = 1e10
    t_interval =  (t_max-t_min)/1000
    temperature = 300
    N0 = {"U235":1e23}
    flux0 = 1e14
    power0 = 1
    Burn_up_chain_result,nuclei_name = Burn_up_chain_calculator(reactor_type,solver_type, chain_key ,file_path,
           t_min,t_max,t_interval,
           temperature,N0,flux0,power0)