import numpy as np
import matplotlib.pyplot as plt

def analytical_solution(A, N0, t):
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

# 示例使用
A = np.array([[0.5, 0.1], [-0.2, 0.3]])  # 示例 2x2 矩阵
N0 = np.array([1, 0])  # 初始条件
t = np.linspace(0, 10, 100)  # 时间点

N = analytical_solution(A, N0, t)

# 绘制结果
plt.plot(t, N[0], label='N1')
plt.plot(t, N[1], label='N2')
plt.xlabel('时间')
plt.ylabel('N')
plt.legend()
plt.title('微分方程 dN/dt = AN 的解析解')
plt.show()

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

def bdf_solver(A, N0, t):
    """
    使用后向差分法（BDF）求解 dN/dt = AN.
    
    参数:
    - A: n x n 系统矩阵
    - N0: 初始 n 维向量
    - t: 时间点数组
    
    返回:
    - N: 在每个时间点的解向量数组
    """
    n = len(N0)
    dt = t[1] - t[0]
    N = np.zeros((n, len(t)))
    N[:, 0] = N0
    
    I = np.eye(n)
    
    # 使用 scipy.linalg.solve 求解线性方程
    for i in range(1, len(t)):
        N[:, i] = scipy.linalg.solve(I - dt * A, N[:, i - 1])
    
    return N

# 示例使用
A = np.array([[0.5, 0.1], [-0.2, 0.3]])  # 示例 2x2 矩阵
N0 = np.array([1, 0])  # 初始条件
t = np.linspace(0, 10, 100)  # 时间点

N = bdf_solver(A, N0, t)

# 绘制结果
plt.plot(t, N[0], label='N1')
plt.plot(t, N[1], label='N2')
plt.xlabel('时间')
plt.ylabel('N')
plt.legend()
plt.title('使用后向差分法求解 dN/dt = AN')
plt.show()

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def solve_ode(A, N0, t_span, t_eval=None):
    """
    Solve the differential equation dN/dt = AN.
    
    Parameters:
    - A: n x n matrix
    - N0: initial n-dimensional vector
    - t_span: tuple (t_start, t_end) indicating the time span
    - t_eval: array of time points at which to store the solution
    
    Returns:
    - t: array of time points
    - N: array of solution vectors at each time point
    """
    n = len(N0)
    
    # Define the ODE function
    def ode_function(t, N):
        return A @ N
    
    # Solve the ODE
    solution = solve_ivp(ode_function, t_span, N0, t_eval=t_eval, vectorized=True)
    
    return solution.t, solution.y

# Example usage
A = np.array([[0.5, 0.1], [-0.2, 0.3]])  # Example 2x2 matrix
N0 = np.array([1, 0])  # Initial condition
t_span = (0, 10)  # Time span from t=0 to t=10
t_eval = np.linspace(0, 10, 100)  # Time points at which to evaluate the solution

t, N = solve_ode(A, N0, t_span, t_eval)

# Plot the results
plt.plot(t, N[0], label='N1')
plt.plot(t, N[1], label='N2')
plt.xlabel('Time')
plt.ylabel('N')
plt.legend()
plt.title('Solution of dN/dt = AN')
plt.show()

import numpy as np
from scipy.linalg import expm

def taylor_expand_truncate(A, N0, t, max_terms=10):
    """
    泰勒展开与截断方法求解 dN/dt = A(t)N
    
    :param A: 时间t对应的矩阵A(t)
    :param N0: 初始向量N(0)
    :param t: 时间点t
    :param delta_t: 时间步长
    :param max_terms: 泰勒展开的最大项数
    :return: 在时间t+delta_t时刻的N向量
    """
    n = len(N0)
    N = np.zeros((n, len(t)))
    N[:, 0] = N0
    
    for id,time in enumerate(t):
        term = np.eye(len(A))  # 初始化为单位矩阵
        for n in range(1, max_terms):
            term = term @ (A * time) / n
            N[:,id] += term @ N0
    return N

A = np.array([[0.5, 0.1], [-0.2, 0.3]])  # Example 2x2 matrix
N0 = np.array([1, 0])  # Initial condition
t_eval = np.linspace(0, 10, 100)
N = taylor_expand_truncate(A, N0, t_eval,max_terms=10)
# Plot the results
plt.plot(t, N[0], label='N1')
plt.plot(t, N[1], label='N2')
plt.xlabel('Time')
plt.ylabel('N')
plt.legend()
plt.title('Solution Taylor of dN/dt = AN')
plt.show()

from scipy.interpolate import pade

def pade_approx(A, N0, t, order=(4, 4)):
    """
    帕德近似法求解 dN/dt = A(t)N
    
    :param A: 时间t对应的矩阵A(t)
    :param N0: 初始向量N(0)
    :param t: 时间点t
    :param delta_t: 时间步长
    :param order: 帕德近似的阶数（默认使用6阶）
    :return: 在时间t+delta_t时刻的N向量
    """
    n = len(N0)
    N = np.zeros((n, len(t)))
    N[:, 0] = N0
    
    for id,time in enumerate(t):

        num, den = pade(A * time, order[0])
        N[:,id] = np.linalg.solve(den, num @ N0)
    return N

# 示例使用
A = np.array([[0.5, 0.1], [-0.2, 0.3]])  # Example 2x2 matrix
N0 = np.array([1, 0])  # Initial condition
t_eval = np.linspace(0, 10, 100)
# N = pade_approx(A, N0, t_eval, order=(6, 6))
# # Plot the results
# plt.plot(t, N[0], label='N1')
# plt.plot(t, N[1], label='N2')
# plt.xlabel('Time')
# plt.ylabel('N')
# plt.legend()
# plt.title('Solution PAde of dN/dt = AN')
# plt.show()

from scipy.sparse.linalg import expm_multiply

def krylov_subspace(A, N0, t, m=30):
    """
    Krylov子空间法求解 dN/dt = A(t)N
    
    :param A: 时间t对应的矩阵A(t)
    :param N0: 初始向量N(0)
    :param t: 时间点t
    :param delta_t: 时间步长
    :param m: Krylov子空间的维数
    :return: 在时间t+delta_t时刻的N向量
    """
    n = len(N0)
    N = np.zeros((n, len(t)))
    N[:,0] = N0
    
    for id,time in enumerate(t):
        N[:,id] = expm_multiply(A * time, N0, start=0, stop=1, num=2)[1]
    return N

# 示例使用
A = np.array([[0.5, 0.1], [-0.2, 0.3]])  # Example 2x2 matrix
N0 = np.array([1, 0])  # Initial condition
t_eval = np.linspace(0, 10, 100)
N = krylov_subspace(A, N0, t_eval,m=30)
# Plot the results
plt.plot(t, N[0], label='N1')
plt.plot(t, N[1], label='N2')
plt.xlabel('Time')
plt.ylabel('N')
plt.legend()
plt.title('Solution krylov of dN/dt = AN')
plt.show()

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

def cram16(A, dt):
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
    
    A_dt = A * dt
    identity = np.eye(A.shape[0])
    
    result = alpha[0] * identity
    for i in range(1, len(alpha)):
        result += 2 * np.real(alpha[i] * scipy.linalg.inv(-identity * theta[i] + A_dt))
    
    return result

def solve_ode_cram(A_func, N0, t):
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
        dt = t[i] - t[i - 1]
        A_t = A_func(t[i])
        exp_A_dt = cram16(A_t, dt)
        N[:, i] = np.dot(exp_A_dt, N[:, i - 1])
    
    return N

# 示例函数 A(t)
def A_func(t):
    return np.array([
        [0.5 , 0.1],
        [-0.2, 0.3 ]
    ])

# 示例使用
N0 = np.array([1, 0])  # 初始条件
t = np.linspace(0, 10, 100)  # 时间点

N = solve_ode_cram(A_func, N0, t)

# 绘制结果
plt.plot(t, N[0], label='N1')
plt.plot(t, N[1], label='N2')
plt.xlabel('时间')
plt.ylabel('N')
plt.legend()
plt.title('使用CRAM方法求解 dN/dt = A(t)N')
plt.show()

# RK4 Method
def rk4_method(A_func, N0, t):
    n = len(N0)
    N = np.zeros((len(t), n))
    N[0] = N0
    
    for i in range(1, len(t)):
        h = t[i] - t[i-1]
        k1 = h * A_func(t[i-1]) @ N[i-1]
        k2 = h * A_func(t[i-1] + h/2) @ (N[i-1] + k1/2)
        k3 = h * A_func(t[i-1] + h/2) @ (N[i-1] + k2/2)
        k4 = h * A_func(t[i-1] + h) @ (N[i-1] + k3)
        
        N[i] = N[i-1] + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    return N

N = rk4_method(A_func, N0, t)

# 绘制结果
plt.plot(t, N[:,0], label='N1')
plt.plot(t, N[:,1], label='N2')
plt.xlabel('时间')
plt.ylabel('N')
plt.legend()
plt.title('使用CRAM方法求解 dN/dt = A(t)N')
plt.show()

# LPAM Method
def lpam_method(A_func, N0, t, k=18, a=20, tau=20):
    from scipy.special import genlaguerre
    n = len(N0)
    N = np.zeros((len(t), n))
    N[0] = N0
    
    for i in range(1, len(t)):
        At = A_func(t[i])*t[i]
        Pt = np.zeros((n, n))
        P_j = np.linalg.inv(np.eye(n) - At/tau  )  # Only compute once if A is time-invariant
        poly = np.linalg.matrix_power(P_j,(a+1)) 
        for j in range(k + 1):
            L_j_a = genlaguerre(j, a)(tau)
            Pt += poly * L_j_a
            poly = (-At/tau) @ poly @ P_j
        
        N[i] = Pt @ N0
        
    return N


N = lpam_method(A_func, N0, t)

# 绘制结果
plt.plot(t, N[:,0], label='N1')
plt.plot(t, N[:,1], label='N2')
plt.xlabel('时间')
plt.ylabel('N')
plt.legend()
plt.title('使用CRAM方法求解 dN/dt = A(t)N')
plt.show()

def QRAM_approx(A_func, N0, t, m=5):
    """
    求积组有理近似法求解 dN/dt = A(t)N
    
    :param A_func: 返回时间t对应的矩阵A(t)的函数
    :param N0: 初始向量N(0)
    :param t: 当前时间点t
    :param delta_t: 时间步长
    :param N: 逼近阶数
    :return: 在时间t+delta_t时刻的N向量
    """

    # 获取抛物线求积组的参数
    xk = np.array([0.1309, 0.1194, 0.2500])


    n = len(N0)
    N = np.zeros((len(t), n))
    N[0] = N0
    
    for i in range(1, len(t)):
        # 计算当前时间点的A矩阵
        A_t = A_func(t)
        delta_t = t[i]-t[i-1]
        zk = delta_t * (-0.5 + 0.5j * xk)  # 求积点
        ck = delta_t * np.array([-1, -1, -1]) / m  # 求积权重
        
        # QRAM方法计算
        for k in range(m):
            # 有理逼近求和
            N[i] += ck[k] * np.linalg.solve(np.eye(A_t.shape[0]) - zk[k] * A_t, N[i-1])
        
        return N

N = QRAM_approx(A_func, N0, t)

# 绘制结果
plt.plot(t, N[:,0], label='N1')
plt.plot(t, N[:,1], label='N2')
plt.xlabel('时间')
plt.ylabel('N')
plt.legend()
plt.title('使用QRAM方法求解 dN/dt = A(t)N')
plt.show()