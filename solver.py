import numpy as np
from scipy.integrate import solve_ivp
import scipy.linalg
import matplotlib.pyplot as plt
from reader import read_h5_database

def flux_analytical_solver(A, N0, t):
    """
    矩阵对角化求解 dN/dt = A(t)N
    
    :param A: 时间t对应的矩阵A(t)
    :param N0: 初始向量N(0)
    :param t: 时间点t
    """
    # 对矩阵 A 进行特征值分解
    eigvals, P = np.linalg.eig(A)
    P_inv = np.linalg.inv(P)
    # 构造 D 矩阵 (特征值对角矩阵) D = np.diag(eigvals)
    # 初始化解矩阵 N
    N = np.zeros((len(N0), len(t)))
    
    # 计算每个时间点的解
    for i, ti in enumerate(t):
        # 计算矩阵指数 e^(Dt)
        exp_Dt = np.diag(np.exp(eigvals* ti))
        # 计算解 N(t) = P * exp(Dt) * P_inv * N0
        N[:, i] = P @ exp_Dt @ P_inv @ N0
    
    return N

def flux_fdf_solver(A, N0, t):
    """
    FDF前向差分算法求解 dN/dt = A(t)N
    
    :param A: 时间t对应的矩阵A(t)
    :param N0: 初始向量N(0)
    :param t: 时间点t
    """
    n = len(N0)
    N = np.zeros((n, len(t)))
    N[:, 0] = N0
    I = np.eye(n)
    # 使用 scipy.linalg.solve 求解线性方程
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        N[:, i] = (I + dt * A) @ N[:, i - 1]
    
    return N

def flux_bdf_solver(A, N0, t):
    """
    BDF后向差分算法求解 dN/dt = A(t)N
    
    :param A: 时间t对应的矩阵A(t)
    :param N0: 初始向量N(0)
    :param t: 时间点t
    """
    n = len(N0)
    N = np.zeros((n, len(t)))
    N[:, 0] = N0
    
    I = np.eye(n)
    
    # 使用 scipy.linalg.solve 求解线性方程
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        N[:, i] = scipy.linalg.solve(I - dt * A, N[:, i - 1])
    
    return N

def flux_standerd_solver_ode(A, N0,t_eval):
    """
    标准求解器 dN/dt = A(t)N
    
    :param A: 时间t对应的矩阵A(t)
    :param N0: 初始向量N(0)
    :param t: 时间点t
    """
    t_span = (t_eval[0],t_eval[-1])
    # Define the ODE function
    def ode_function(t, N):
        return A @ N
    
    # Solve the ODE
    solution = solve_ivp(ode_function, t_span, N0, t_eval=t_eval, vectorized=True)
    
    return solution.y

def flux_rk4_solver(A_func,N0,t):
    """
    四阶龙格库塔求解 dN/dt = A(t)N
    
    :param A: 时间t对应的矩阵A(t)
    :param N0: 初始向量N(0)
    :param t: 时间点t
    """
    n = len(N0)
    N = np.zeros((n,len(t)))
    N[:,0] = N0
    
    for i in range(1, len(t)):
        h = t[i] - t[i-1]
        k1 = h * A_func @ N[:,i-1]
        k2 = h * A_func @ (N[:,i-1] + k1/2)
        k3 = h * A_func @ (N[:,i-1] + k2/2)
        k4 = h * A_func @ (N[:,i-1] + k3)
        
        N[:,i] = N[:,i-1] + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    return N

def flux_taylor_expand_truncate_solver(A, N0, t, max_terms=10):
    """
    泰勒展开与截断方法求解 dN/dt = A(t)N
    
    :param A: 时间t对应的矩阵A(t)
    :param N0: 初始向量N(0)
    :param t: 时间点t
    """
    n = len(N0)
    N = np.zeros((n, len(t)))
    N[:, 0] = N0
    
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        term = np.eye(len(A))  # 初始化为单位矩阵
        for n in range(1, max_terms):
            term = term @ (A * t[i]) / n
            N[:,i] += term @ N0
    return N

def flux_krylov_subspace_solver(A, N0, t, m=30):
    from scipy.sparse.linalg import expm_multiply
    """
    Krylov子空间法求解 dN/dt = A(t)N
    
    :param A: 时间t对应的矩阵A(t)
    :param N0: 初始向量N(0)
    :param t: 时间点t
    :param m: Krylov子空间的维数
    """
    n = len(N0)
    N = np.zeros((n, len(t)))
    N[:,0] = N0
    
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        N[:,i] = expm_multiply(A * t[i], N0, start=0, stop=1, num=2)[1]
    return N

def cram16(A):
    """
    使用16阶切比雪夫有理近似法（CRAM）计算矩阵指数 e^(A*dt)
    
    参数:
    - A: n x n 系统矩阵
    - dt: 时间步长
    
    返回:
    - e^(A*dt): 矩阵指数的近似值
    """
    # 定义CRAM 16阶系数
    alpha = np.array([
        2.1248537104952237488e-16,
        -5.0901521865224915650e-07 - 2.4220017652852287970e-05j,
        2.1151742182466030907e-04 + 4.3892969647380673918e-03j,
        1.1339775178483930527e+02 + 1.0194721704215856450e+02j,
        1.5059585270023467528e+01 - 5.7514052776421819979e+00j,
        -6.4500878025539646595e+01 - 2.2459440762652096056e+02j,
        -1.4793007113557999718e+00 + 1.7686588323782937906e+00j,
        -6.2518392463207918892e+01 - 1.1190391094283228480e+01j,
        4.1023136835410021273e-02 - 1.5743466173455468191e-01j
    ])
    theta = np.array([
        0,
        -1.0843917078696988026e+01 + 1.9277446167181652284e+01j,
        -5.2649713434426468895e+00 + 1.6220221473167927305e+01j,
        5.9481522689511774808e+00 + 3.5874573620183222829e+00j,
        3.5091036084149180974e+00 + 8.4361989858843750826e+00j,
        6.4161776990994341923e+00 + 1.1941223933701386874e+00j,
        1.4193758971856659786e+00 + 1.0925363484496722585e+01j,
        4.9931747377179963991e+00 + 5.9968817136039422260e+00j,
        -1.4139284624888862114e+00 + 1.3497725698892745389e+01j
    ])
    identity = np.eye(A.shape[0])
    
    result = alpha[0] * identity
    for i in range(1, len(alpha)):
        result += 2 * np.real(alpha[i] * scipy.linalg.inv(-identity * theta[i] + A))
    
    return result

def flux_cram_solver(A_func, N0, t):
    """
    使用CRAM方法求解微分方程 dN/dt = A(t) * N
    
    参数:
    - A_func: 返回矩阵A的函数, A(t)
    - N0: 初始状态向量
    - t: 时间点数组
    
    返回:
    - N: 在每个时间点的解向量数组
    """
    n = len(N0)
    N = np.zeros((n, len(t)))
    N[:, 0] = N0
    
    for i in range(1, len(t)):
        exp_A = cram16(A_func*(t[i]-t[i-1]))
        N[:, i] = np.dot(exp_A, N[:, i - 1])
    
    return N

def flux_lpam_method(A_func, N0, t, k=18, a=20, tau=20):
    from scipy.special import genlaguerre
    n = len(N0)
    N = np.zeros((n,len(t)))
    N[:,0] = N0
    
    for i in range(1, len(t)):
        At = A_func*t[i]
        Pt = np.zeros((n, n))
        P_j = np.linalg.inv(np.eye(n) - At/tau  )  # Only compute once if A is time-invariant
        poly = np.linalg.matrix_power(P_j,(a+1)) 
        for j in range(k + 1):
            L_j_a = genlaguerre(j, a)(tau)
            Pt += poly * L_j_a
            poly = (-At/tau) @ poly @ P_j
        
        N[:,i] = Pt @ N0
        
    return N

def flux_solver(solver_type,A,N,t):
    """
    solver:
     - A 粒子数变化率矩阵
     - N 粒子数初始值
     - t 模拟时间点
     """
    if solver_type == 'analytical':
            result = flux_analytical_solver(A,N,t)
    elif solver_type == 'BDF':
            result = flux_bdf_solver(A,N,t)
    elif solver_type == 'FDF':
            result = flux_fdf_solver(A,N,t)
    elif solver_type == 'standard':
            result = flux_standerd_solver_ode(A,N,t)
    elif solver_type == 'rk4':
            result = flux_rk4_solver(A,N,t)
    elif solver_type == 'taylor':
            result = flux_taylor_expand_truncate_solver(A,N,t, max_terms=10)
    elif solver_type == 'krylov':
            result = flux_krylov_subspace_solver(A,N,t, m=30)
    elif solver_type == 'CRAM':
            result = flux_cram_solver(A,N,t)
    elif solver_type == 'LPAM':
            result = flux_lpam_method(A,N,t)
    return result

# RK4 Method
def power_rk4_solver(A_func, N0, t):
    n = len(N0)
    N = np.zeros((n, len(t)))
    N[:, 0] = N0
    
    for i in range(1, len(t)):
        h = t[i] - t[i-1]
        k1 = h * A_func(t[i-1]) @ N[:,i-1]
        k2 = h * A_func(t[i-1] + h/2) @ (N[:,i-1] + k1/2)
        k3 = h * A_func(t[i-1] + h/2) @ (N[:,i-1] + k2/2)
        k4 = h * A_func(t[i-1] + h) @ (N[:,i-1] + k3)
        
        N[:,i] = N[:,i-1] + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    return N

def power_bdf_solver(A_func, N0, t):
    """
    BDF后向差分算法求解 dN/dt = A(t)N
    
    :param A: 时间t对应的矩阵A(t)
    :param N0: 初始向量N(0)
    :param t: 时间点t
    """
    n = len(N0)
    N = np.zeros((n, len(t)))
    N[:, 0] = N0
    
    I = np.eye(n)
    
    # 使用 scipy.linalg.solve 求解线性方程
    for i in range(1, len(t)):
        dt = t[i] - t[i-1] 
        A = A_func(t[i])
        N[:, i] = scipy.linalg.solve(I - dt * A, N[:, i - 1])
    
    return N

def power_fdf_solver(A_func, N0, t):
    """
    FDF前向差分算法求解 dN/dt = A(t)N
    
    :param A: 时间t对应的矩阵A(t)
    :param N0: 初始向量N(0)
    :param t: 时间点t
    """
    n = len(N0)
    dt = t[1] - t[0]
    N = np.zeros((n, len(t)))
    N[:, 0] = N0
    
    I = np.eye(n)
    
    # 使用 scipy.linalg.solve 求解线性方程
    for i in range(1, len(t)):
        dt = t[i] - t[i-1] 
        A = A_func(t[i])
        N[:, i] = (I + dt * A) @ N[:, i - 1]
    
    return N

def power_cram_solver(A_func, N0, t):
    """
    使用CRAM方法求解微分方程 dN/dt = A(t) * N
    
    参数:
    - A_func: 返回矩阵A的函数, A(t)
    - N0: 初始状态向量
    - t: 时间点数组
    
    返回:
    - N: 在每个时间点的解向量数组
    """
    n = len(N0)
    N = np.zeros((n, len(t)))
    N[:, 0] = N0
    
    for i in range(1, len(t)):
        exp_A = cram16(A_func(t[i])*(t[i]-t[i-1]))
        N[:, i] = np.dot(exp_A, N[:, i - 1])
    
    return N

def power_lpam_method(A_func, N0, t, k=6, a=20, tau=20):
    from scipy.special import genlaguerre
    n = len(N0)
    N = np.zeros((n,len(t)))
    N[:,0] = N0
    
    for i in range(1, len(t)):
        At = A_func(t[i])*(t[i]-t[i-1])
        Pt = np.zeros((n, n))
        P_j = np.linalg.inv(np.eye(n) - At/tau  )  # Only compute once if A is time-invariant
        poly = np.linalg.matrix_power(P_j,(a+1)) 
        for j in range(k + 1):
            L_j_a = genlaguerre(j, a)(tau)
            Pt += poly * L_j_a
            poly = (-At/tau) @ poly @ P_j
        
        N[:,i] = Pt @ N[:,i-1]
        
    return N

def power_taylor_expand_truncate_solver(A, N0, t, max_terms=4):
    """
    泰勒展开与截断方法求解 dN/dt = A(t)N
    
    :param A: 时间t对应的矩阵A(t)
    :param N0: 初始向量N(0)
    :param t: 时间点t
    """
    n = len(N0)
    N = np.zeros((n, len(t)))
    N[:, 0] = N0
    
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        A_t = A(t[i])*dt
        term = np.eye(len(A_t))  # 初始化为单位矩阵
        N[:,i] += N[:,i-1]
        for n in range(1, max_terms):
            term = term @ (A_t) / n
            N[:,i] += term @ N[:,i-1]
    return N

def power_krylov_subspace_solver(A, N0, t, m=30):
    from scipy.sparse.linalg import expm_multiply
    """
    Krylov子空间法求解 dN/dt = A(t)N
    
    :param A: 时间t对应的矩阵A(t)
    :param N0: 初始向量N(0)
    :param t: 时间点t
    :param m: Krylov子空间的维数
    """
    n = len(N0)
    N = np.zeros((n, len(t)))
    N[:,0] = N0
    
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        N[:,i] = expm_multiply(A(t[i])*dt, N[:,i-1], start=0, stop=1, num=2)[1]
    return N

def power_solver(solver_type,A,N,t):
    """
    solver:
     - A 粒子数变化率矩阵
     - N 粒子数初始值
     - t 模拟时间点
     """
    if solver_type == 'BDF':
            result = power_bdf_solver(A,N,t)
    elif solver_type == 'rk4':
            result = power_rk4_solver(A,N,t)
    elif solver_type == 'FDF':
            result = power_fdf_solver(A,N,t)
    elif solver_type == 'CRAM':
            result = power_cram_solver(A,N,t)
    elif solver_type == 'LPAM':
            result = power_lpam_method(A,N,t)
    elif solver_type == 'taylor':
            result = power_taylor_expand_truncate_solver(A,N,t, max_terms=10)
    elif solver_type == 'krylov':
            result = power_krylov_subspace_solver(A,N,t, m=30)
    return result
