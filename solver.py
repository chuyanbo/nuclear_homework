import numpy as np
from scipy.integrate import solve_ivp
import scipy.linalg
import matplotlib.pyplot as plt
from reader import read_h5_database

def analytical_solver(A, N0, t):
    # 对矩阵 A 进行特征值分解
    eigvals, P = np.linalg.eig(A)
    P_inv = np.linalg.inv(P)
    
    # 构造 D 矩阵 (特征值对角矩阵)
    D = np.diag(eigvals)
    
    # 初始化解矩阵 N
    N = np.zeros((len(N0), len(t)))
    
    # 计算每个时间点的解
    for i, ti in enumerate(t):
        # 计算矩阵指数 e^(Dt)
        exp_Dt = np.diag(np.exp(eigvals * ti))
        # 计算解 N(t) = P * exp(Dt) * P_inv * N0
        N[:, i] = P @ exp_Dt @ P_inv @ N0
    
    return N

def bdf_solver(A, N0, t):
    n = len(N0)
    dt = t[1] - t[0]
    N = np.zeros((n, len(t)))
    N[:, 0] = N0
    
    I = np.eye(n)
    
    # 使用 scipy.linalg.solve 求解线性方程
    for i in range(1, len(t)):
        N[:, i] = scipy.linalg.solve(I - dt * A, N[:, i - 1])
    
    return N

def standerd_solver_ode(A, N0,t_eval):
    """
    - t_span: 元组 (t_start, t_end) 时间区间
    - t_eval: 时间点序列
    """
    t_span = (t_eval[0],t_eval[-1])
    # Define the ODE function
    def ode_function(t, N):
        return A @ N
    
    # Solve the ODE
    solution = solve_ivp(ode_function, t_span, N0, t_eval=t_eval, vectorized=True)
    
    return solution.y

def solver(reactor_type,solver_type, chain_key ,file_path,
           t_min,t_max,t_interval,
           temperature,N0,flux0):
    """
    parameter:
     - reactor_type 反应堆运行状态
     - solver_type 求解方式
     - chain_key 反应链类别
     - file_path 数据库路径
     - t_min 时间下限
     - t_max 时间上限
     - t_interval 时间间隔
     - temperature 温度
     - N0 初始燃料值
     - flux0 初始通量
    solver:
     - A 粒子数变化率矩阵
     - N 粒子数初始值
     - t 模拟时间点
    """
    t = np.arange(t_min,t_max,t_interval)
    dA , auA = read_h5_database(file_path,chain_key,temperature)
    n = len(dA)
    if reactor_type == 'flux':
        N = np.zeros((n))
        N[0] = N0
        A = dA + flux0 * auA
        if solver_type == 'analytical':
            result = analytical_solver(A,N,t)
        elif solver_type == 'BDF':
            result = bdf_solver(A,N,t)
        elif solver_type == 'standard':
            result = standerd_solver_ode(A,N,t)
        return result
    
    if reactor_type == 'power':
        N = np.zeros((n))
        N[0] = N0
        A = dA + flux0 * auA
        if solver_type == 'analytical':
            result = analytical_solver(A,N,t)
        elif solver_type == 'BDF':
            result = bdf_solver(A,N,t)
        elif solver_type == 'standard':
            result = standerd_solver_ode(A,N,t)
        return result
    
    return
