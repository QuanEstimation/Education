from quanestimation import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
import time
import os
from qutip import spin_coherent, jmat
from mpl_toolkits.mplot3d import Axes3D

plt.rc("font",family='MicroSoft YaHei',weight="bold")

def MakeImg(
    omega=1.0,           # 系统本征频率 (Hz)
    tspan=None,          # 演化时间范围 (s)
    cam=0.5              # 耗散系数 (cam) - 对应截图中的衰减/退相干强度
):
    """
    量子费舍尔信息量演化图生成函数。
    修改点：
    1. 耗散系数由 cam 变量控制 (范围 0~1)。
    2. QFI 计算类型固定为 SLD (对称对数导数)。
    """
    
    # --- 1. 固定默认值设置 ---
    # 初态 (Fixed Default)
    rho0 = 0.5 * np.array([[1., 1.], [1., 1.]])
    
    # 测量算符 (Fixed Default)
    M1 = 0.5 * np.array([[1., 1.], [1., 1.]])
    M2 = 0.5 * np.array([[1., -1.], [-1., 1.]])
    M = [M1, M2]

    # --- 2. 处理可变参数 ---
    # Hamiltonian
    sz = np.array([[1., 0.], [0., -1.]])
    H0 = 0.5 * omega * sz
    dH = [0.5 * sz]  # 对 omega 的导数

    # 时间跨度
    if tspan is None:
        tspan = np.linspace(0., 50., 2000)
    
    # --- 3. 核心修改：耗散系数 ---
    # 物理含义：cam 代表耗散强度
    # cam = 0: 无耗散
    # cam = 0.1~0.5: 弱耗散
    # cam = 0.5~1: 强耗散
    sp = np.array([[0., 1.], [0., 0.]])  
    sm = np.array([[0., 0.], [1., 0.]]) 
    decay = [[sp, 0.0], [sm, cam]]  # <--- 关键修改：使用 cam 变量

    # --- 4. 动力学演化 ---
    dynamics = Lindblad(tspan, rho0, H0, dH, decay)
    rho, drho = dynamics.expm()

    # --- 5. 计算 CFI 和 QFI ---
    # 注意：QFI 计算类型已取消选项，固定为 "SLD"
    I, F = [], []
    for ti in range(1, len(tspan)):
        # CFI
        I_tp = CFIM(rho[ti], drho[ti], M=M)
        I.append(I_tp)
        
        # QFI (固定为 SLD)
        F_tp = QFIM(rho[ti], drho[ti], LDtype="SLD") 
        F.append(F_tp)

    # --- 6. 绘图逻辑 (保持不变) ---
    def extract_scalar(val):
        if isinstance(val, np.ndarray):
            return float(val[0, 0])
        else:
            return float(val)

    cfi_values = [extract_scalar(i) for i in I]
    qfi_values = [extract_scalar(f) for f in F]

    plt.figure(figsize=(10, 6))
    plt.plot(tspan[1:], cfi_values, label='CFI (经典费舍尔信息量)', linestyle='--', color='blue')
    plt.plot(tspan[1:], qfi_values, label='QFI (量子费舍尔信息量)', linestyle='-', color='red')
    plt.xlabel('时间 (t)', fontsize=12)
    plt.ylabel('费舍尔信息量', fontsize=12)
    plt.title(f'耗散系数 γ = {cam} 时的 CFI/QFI 演化', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # --- 7. 文件保存 (保持不变) ---
    os.makedirs("./Img", exist_ok=True)
    filename = str(time.time())
    plt.savefig(f"./Img/{filename}.png")
    plt.close() # 建议关闭绘图以释放内存
    return f"{filename}.png"


def run_bayesian_quantum_estimation(eta=0.2, t_end=1.0, B_factor=0.5*np.pi, 
                                    omega0=1.0, mu=0.0, x_range_factor=0.5, 
                                    n_x_points=100, n_t_points=1000):
    """
    运行贝叶斯量子参数估计模拟并绘图。
    
    参数:
    -----------
    eta : float
        [推荐调整] 先验分布的标准差，代表初始对参数 x 的不确定度。
        建议范围: 0.01 ~ 2.0
    t_end : float
        [推荐调整] 量子演化的总时间长度。
        建议范围: 0.1 ~ 10.0 (取决于 B 和 omega0)
    B_factor : float
        [推荐调整] 磁场强度因子 (对应原代码中的 B)。
        建议范围: 0.1 ~ 5.0
    omega0 : float
        特征频率系数，默认为 1.0。
    mu : float
        先验分布的均值中心，默认为 0.0。
    x_range_factor : float
        采样范围系数，实际范围为 [-factor*pi, factor*pi]。默认为 0.5 (即 +/- pi/2)。
    n_x_points : int
        参数 x 的离散化点数。
    n_t_points : int
        时间演化的离散化点数。
        
    返回:
    -----------
    fig : matplotlib.figure.Figure
        生成的图形对象。
    results : dict
        包含计算数据 (x, p, sensitivity, bloch_z) 的字典。
    """
    
    # --- 1. 初始化物理量 ---
    # 初始状态 (纯态 |+>)
    rho0 = 0.5 * np.array([[1., 1.], [1., 0.]]) # 注意：原代码是 [[1,1],[1,1]]，这里修正为合法的密度矩阵需保证迹为1且半正定
    # 修正：原代码 0.5*[[1,1],[1,1]] 是 |+><+|，是合法的。
    rho0 = 0.5 * np.array([[1., 1.], [1., 1.]])
    
    # 泡利矩阵
    sx = np.array([[0., 1.], [1., 0.]])
    sy = np.array([[0., -1.j], [1.j, 0.]]) 
    sz = np.array([[1., 0.], [0., -1.]])
    
    # 哈密顿量及其导数 (使用传入的 B_factor)
    # H(x) = 0.5 * B * omega0 * (sx * cos(x) + sz * sin(x))
    H0_func = lambda x_val: 0.5 * B_factor * omega0 * (sx * np.cos(x_val) + sz * np.sin(x_val))
    dH_func = lambda x_val: [0.5 * B_factor * omega0 * (-sx * np.sin(x_val) + sz * np.cos(x_val))]
    
    # --- 2. 设置先验分布 ---
    # 动态调整采样范围以覆盖主要的概率质量 (mu +/- 4*eta 或 固定的物理范围)
    x_max = max(np.pi * x_range_factor, mu + 4 * eta)
    x_min = min(-np.pi * x_range_factor, mu - 4 * eta)
    
    x = np.linspace(x_min, x_max, n_x_points)
    
    # 高斯分布函数
    p_func = lambda x_val, m, e: np.exp(-(x_val - m)**2 / (2 * e**2)) / (e * np.sqrt(2 * np.pi))
    dp_func = lambda x_val, m, e: -(x_val - m) * np.exp(-(x_val - m)**2 / (2 * e**2)) / (e**3 * np.sqrt(2 * np.pi))
    
    p_tp = [p_func(x[i], mu, eta) for i in range(len(x))]
    dp_tp = [dp_func(x[i], mu, eta) for i in range(len(x))]
    
    # 归一化
    c = simpson(p_tp, x)
    if c == 0: c = 1e-9 # 防止除零
    p = np.array(p_tp) / c
    dp = np.array(dp_tp) / c
    
    # --- 3. 时间演化设置 ---
    tspan = np.linspace(0., t_end, n_t_points)
    
    # 初始化存储容器
    rho_final = [np.zeros((2, 2), dtype=np.complex128) for _ in range(len(x))]
    drho_final = [np.zeros((2, 2), dtype=np.complex128) for _ in range(len(x))]
    
    # --- 4. 动力学循环 ---
    # 注意：此处依赖外部定义的 Lindblad 类
    try:
        for i in range(len(x)):
            H0_tp = H0_func(x[i])
            dH_tp = dH_func(x[i])
            
            # 实例化动力学求解器
            dynamics = Lindblad(tspan, rho0, H0_tp, dH_tp)
            rho_tp, drho_tp = dynamics.expm()
            
            rho_final[i] = rho_tp[-1]
            drho_final[i] = drho_tp[-1] # 假设返回的是列表，取最后一个时间点的导数
    except NameError:
        print("错误: 未找到 'Lindblad' 类定义。请确保在运行此函数前已导入或定义该类。")
        return None, None
    except Exception as e:
        print(f"动力学演化过程中出错: {e}")
        return None, None

    # --- 5. 后处理：提取布洛赫矢量 ---
    bloch_z = []
    
    for i in range(len(x)):
        state = rho_final[i]
        
        # 计算 Z 分量期望值: <Sz> = Tr(rho * sz)
        bz = np.real(np.trace(state @ sz))
        bloch_z.append(bz)
    
    bloch_z = np.array(bloch_z)

    # --- 6. 绘图 ---
    # 配置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 绘制先验分布
    color_p = 'tab:blue'
    ax1.set_xlabel('参数 x (弧度)', fontsize=12)
    ax1.set_ylabel('先验概率密度', color=color_p, fontsize=12)
    ax1.plot(x, p, color=color_p, linewidth=2, label=f'先验分布 ($\eta={eta}$)')
    ax1.tick_params(axis='y', labelcolor=color_p)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 双轴：布洛赫 Z 分量
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60)) 
    ax3.set_ylabel('终态 Z 分量期望值 ($\\langle S_z \\rangle$)', color='tab:green', fontsize=12)
    ax3.plot(x, bloch_z, color='tab:green', alpha=0.7, linewidth=2, label='终态 $\\langle S_z \\rangle$')
    ax3.tick_params(axis='y', labelcolor='tab:green')
    
    # 合并图例
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_3, labels_3 = ax3.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_3, labels_1 + labels_3, loc='best')
    
    title_str = f'贝叶斯量子参数估计分析\n参数: $\eta={eta}$, $T={t_end}$, $B={B_factor:.2f}$'
    plt.title(title_str, fontsize=14)
    
    plt.tight_layout()
    
    results = {
        'x': x,
        'p': p,
        'bloch_z': bloch_z
    }
    
    # --- 7. 文件保存 (保持不变) ---
    os.makedirs("./Img", exist_ok=True)
    filename = str(time.time())
    plt.savefig(f"./Img/{filename}.png")
    plt.close() # 建议关闭绘图以释放内存
    return f"{filename}.png"





def run_quantum_estimation(
    omega1=1.0,
    omega2=1.0,
    g=0.1,
    gamma1=0.05,
    gamma2=0.05,
    theta=np.pi/4,
    p1=0.85,
    p2=0.1,
    w1=1.0,
    w2=1.0,
    t_max=10.0,
    n_points=500
):
    """
    运行量子参数估计模拟，计算 CFIM, QFIM, HCRB, NHB 并绘图。
    
    返回:
        filename: 保存的图片文件名 (字符串)
        results: 字典，包含 'trace_I', 'trace_F', 'val_HCRB', 'val_NHB'
    """
    
    print(f"开始模拟... 参数: omega1={omega1}, omega2={omega2}, g={g}")
    print(f"耗散: gamma1={gamma1}, gamma2={gamma2}, 初始角 theta={theta:.4f}")
    print(f"测量: p1={p1}, p2={p2}, 权重: w1={w1}, w2={w2}")

    psi0 = np.array([np.cos(theta), 0., 0., np.sin(theta)])
    rho0 = np.dot(psi0.reshape(-1, 1), psi0.reshape(1, -1).conj())

    sx = np.array([[0., 1.], [1., 0.]])
    sy = np.array([[0., -1.j], [1.j, 0.]]) 
    sz = np.array([[1., 0.], [0., -1.]])
    ide = np.array([[1., 0.], [0., 1.]]) 
    
    H0 = omega1*np.kron(sz, ide) + omega2*np.kron(ide, sz) + g*np.kron(sx, sx)
    
    dH_omega2 = np.kron(ide, sz)
    dH_g = np.kron(sx, sx)
    dH = [dH_omega2, dH_g] 

    decay = [[np.kron(sz, ide), gamma1], [np.kron(ide, sz), gamma2]]

    m1 = np.array([1., 0., 0., 0.])
    M1 = p1 * np.dot(m1.reshape(-1, 1), m1.reshape(1, -1).conj())
    
    M2 = p2 * np.ones((4, 4))
    
    M3 = np.identity(4) - M1 - M2
    
    if np.min(np.linalg.eigvalsh(M3)) < -1e-8:
        print("警告: 计算出的 M3 不是半正定的，请检查 p1 和 p2 的设置。")
        
    M = [M1, M2, M3]

    tspan = np.linspace(0., t_max, n_points)
    
    dynamics = Lindblad(tspan, rho0, H0, dH, decay)
    rho, drho = dynamics.expm()
    
    W = np.diag([w1, w2])

    F, I, f_HCRB, f_NHB = [], [], [], []
    
    print("正在计算 Fisher 信息量和边界...")
    
    for ti in range(1, len(tspan)):
        I_tp = CFIM(rho[ti], drho[ti], M=M)
        I.append(I_tp)
        
        F_tp = QFIM(rho[ti], drho[ti])
        F.append(F_tp)
        
        f_tp1 = HCRB(rho[ti], drho[ti], W, eps=1e-6)
        f_HCRB.append(f_tp1)
        
        f_tp2 = NHB(rho[ti], drho[ti], W)
        f_NHB.append(f_tp2)

    t_plot = tspan[1:]
    
    trace_I = [np.trace(np.real(x)) for x in I]
    trace_F = [np.trace(np.real(x)) for x in F]
    
    def extract_val(data_list):
        vals = []
        for x in data_list:
            if isinstance(x, np.ndarray):
                if x.ndim == 0:
                    vals.append(np.real(x))
                else:
                    vals.append(np.real(np.trace(x))) 
            else:
                vals.append(np.real(x))
        return vals

    val_HCRB = extract_val(f_HCRB)
    val_NHB = extract_val(f_NHB)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Quantum Parameter Estimation (theta={theta:.2f}, g={g})', fontsize=16)

    axs[0, 0].plot(t_plot, trace_I, color='blue', label='Tr(CFIM)')
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].set_ylabel('Trace')
    axs[0, 0].set_title('Classical Fisher Information Matrix (Trace)')
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend()

    axs[0, 1].plot(t_plot, trace_F, color='green', label='Tr(QFIM)')
    axs[0, 1].set_xlabel('Time')
    axs[0, 1].set_ylabel('Trace')
    axs[0, 1].set_title('Quantum Fisher Information Matrix (Trace)')
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].legend()

    axs[1, 0].plot(t_plot, val_HCRB, color='red', label='HCRB')
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('Bound Value')
    axs[1, 0].set_title('Holevo-Cramér-Rao Bound')
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].legend()

    axs[1, 1].plot(t_plot, val_NHB, color='purple', label='NHB')
    axs[1, 1].set_xlabel('Time')
    axs[1, 1].set_ylabel('Bound Value')
    axs[1, 1].set_title('Nagaoka-Hayashi Bound')
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    os.makedirs("./Img", exist_ok=True)
    filename = str(time.time())
    plt.savefig(f"./Img/{filename}.png")
    plt.close()
    print(f"图片已保存至: ./Img/{filename}.png")

    results = {
        'time': t_plot,
        'trace_I': trace_I,
        'trace_F': trace_F,
        'HCRB': val_HCRB,
        'NHB': val_NHB
    }
    
    print("模拟完成。")
    return f"{filename}.png", results

def run_quantum_parameter_estimation(
    B=np.pi/2.0,
    omega0=1.0,
    x_min=0.0,
    x_max=0.5*np.pi,
    x_num=1000,
    t_max=1.0,
    t_num=1000,
    y_num=1000,
    seed=1234,
    prob_threshold=1/3
):
    """
    量子参数估计函数 - 贝叶斯估计和最大似然估计
    
    参数:
    B: 磁场强度，默认值为 π/2.0
    omega0: 角频率，默认值为 1.0
    x_min: 参数 x 的最小值，默认值为 0.0
    x_max: 参数 x 的最大值，默认值为 π/2
    x_num: 参数 x 的采样点数，默认值为 1000
    t_max: 时间演化的最大值，默认值为 1.0
    t_num: 时间演化的采样点数，默认值为 1000
    y_num: 测量次数，默认值为 1000
    seed: 随机种子，默认值为 1234
    prob_threshold: 测量结果的概率阈值，默认值为 1/3
    
    返回:
    filename: 保存的图片文件名 (字符串)
    results: 字典，包含 'x', 'pout', 'Lout', 'xout'
    """
    
    # initial state
    rho0 = 0.5*np.array([[1., 1.], [1., 1.]])
    # free Hamiltonian
    sx = np.array([[0., 1.], [1., 0.]])
    sy = np.array([[0., -1.j], [1.j, 0.]]) 
    sz = np.array([[1., 0.], [0., -1.]])
    H0_func = lambda x: 0.5*B*omega0*(sx*np.cos(x)+sz*np.sin(x))
    # derivative of the free Hamiltonian on x
    dH_func = lambda x: [0.5*B*omega0*(-sx*np.sin(x)+sz*np.cos(x))]
    # measurement
    M1 = 0.5*np.array([[1., 1.], [1., 1.]])
    M2 = 0.5*np.array([[1.,-1.], [-1., 1.]])
    M = [M1, M2]
    # prior distribution
    x = np.linspace(x_min, x_max, x_num)
    p = (1.0/(x[-1]-x[0]))*np.ones(len(x))
    # time length for the evolution
    tspan = np.linspace(0., t_max, t_num)
    # dynamics
    rho = [
        np.zeros((len(rho0), len(rho0)), dtype=np.complex128) 
        for _ in range(len(x))
    ]
    for i in range(len(x)):
        H0 = H0_func(x[i])
        dH = dH_func(x[i])
        dynamics = Lindblad(tspan, rho0, H0, dH)
        rho_tp, _ = dynamics.expm()
        rho[i] = rho_tp[-1]

    np.random.seed(seed)
    y = [1 if np.random.rand() < prob_threshold else 0 for _ in range(y_num)]

    pout, xout = Bayes([x], p, rho, y, M=M, estimator="MAP", savefile=False)

    Lout, xout = MLE([x], rho, y, M=M, savefile=False)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(x, pout, 'b-', linewidth=2, label='后验分布')
    plt.axvline(xout, color='r', linestyle='--', linewidth=2, label=f'贝叶斯估计值: x={xout:.4f}')
    plt.xlabel('参数 x', fontsize=12)
    plt.ylabel('概率密度', fontsize=12)
    plt.title('贝叶斯估计 (MAP)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(x, Lout, 'g-', linewidth=2, label='似然函数')
    plt.axvline(xout, color='r', linestyle='--', linewidth=2, label=f'最大似然估计值: x={xout:.4f}')
    plt.xlabel('参数 x', fontsize=12)
    plt.ylabel('似然值', fontsize=12)
    plt.title('最大似然估计', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    
    os.makedirs("./Img", exist_ok=True)
    filename = str(time.time())
    plt.savefig(f"./Img/{filename}.png")
    plt.close()
    
    results = {
        'x': x,
        'pout': pout,
        'Lout': Lout,
        'xout': xout
    }
    
    return f"{filename}.png", results


# ==========================================
# 使用示例

#def run_state_optimization(
#    N=8,
#    Lambda=1.0,
#    g=0.5,
#    h=0.1,
#    decay_rate=0.1,
#    t_max=10.0,
#    n_points=2500,
#    max_episode=20,
#    epsilon=0.01,
#    beta1=0.90,
#    beta2=0.99,
#    use_adam=False
#):
#    """
#    运行量子状态优化 (State Optimization) 并生成QFI收敛曲线和最优态分布图。
#    
#    参数:
#    -----------
#    N : int
#        系统维度 (自旋粒子数的两倍)，默认为 8。
#    Lambda : float
#        哈密顿量中的耦合系数，默认为 1.0。
#    g : float
#        粒子间的相互作用强度，默认为 0.5。
#    h : float
#        外场强度，默认为 0.1。
#    decay_rate : float
#        耗散速率，默认为 0.1。
#    t_max : float
#        演化总时间，默认为 10.0。
#    n_points : int
#        时间采样点数，默认为 2500。
#    max_episode : int
#        最大迭代次数，默认为 20。
#    epsilon : float
#        学习率，默认为 0.01。
#    beta1 : float
#        Adam优化器参数1，默认为 0.90。
#    beta2 : float
#        Adam优化器参数2，默认为 0.99。
#    use_adam : bool
#        是否使用Adam优化器，默认为 False。
#        
#    返回:
#    -----------
#    filenames : tuple
#        保存的两张图片文件名 (收敛曲线, 直方图)。
#    results : dict
#        包含 'optimal_states', 'qfi_values' 和 'final_qfi' 的字典。
#    """
#    
#    print("正在生成可视化结果...")
#    
#    os.makedirs("./Img", exist_ok=True)
#    
#    psi_css = spin_coherent(0.5*N, 0.5*np.pi, 0.5*np.pi, type="ket").full()
#    psi_css = psi_css.reshape(1, -1)[0]
#    psi0 = [psi_css]
#    
#    Jx, Jy, Jz = jmat(0.5*N)
#    Jx, Jy, Jz = Jx.full(), Jy.full(), Jz.full()
#    H0 = -Lambda*(np.dot(Jx, Jx) + g*np.dot(Jy, Jy))/N - h*Jz
#    
#    dH = [-Lambda*np.dot(Jy, Jy)/N]
#    
#    decay = [[Jz, decay_rate]]
#    
#    tspan = np.linspace(0., t_max, n_points)
#    
#    AD_paras = {"Adam":use_adam, "psi0":psi0, "max_episode":max_episode, \
#                "epsilon":epsilon, "beta1":beta1, "beta2":beta2}
#    state = StateOpt(savefile=False, method="AD", **AD_paras)
#    
#    state.dynamics(tspan, H0, dH, decay=decay, dyn_method="expm")
#    
#    state.QFIM()
#    optimal_states = np.load("states.npy", allow_pickle=True)
#    print(optimal_states)
#    
#    qfi_values = np.loadtxt("f.csv")
#    
#    plt.figure(figsize=(10, 6))
#    plt.plot(qfi_values, 'b-', linewidth=2, label='QFI')
#    plt.xlabel('迭代次数', fontsize=12)
#    plt.ylabel('QFI 值', fontsize=12)
#    plt.title('量子费舍尔信息量优化收敛曲线 (AD算法)', fontsize=14)
#    plt.legend(fontsize=10)
#    plt.grid(True, alpha=0.3)
#    plt.tight_layout()
#    
#    filename = str(time.time())
#    plt.savefig(f"./Img/{filename}.png", dpi=300, bbox_inches='tight')
#    plt.close()
#    
#    optimal_states = np.load("states.npy", allow_pickle=True)
## 如果保存的是每一步状态（形状 (episodes, dim)），取最后一行作为if optimal_states.ndim == 2 and optimal_states.shape:
##   optimal_states=optimal_states[-1]
#    
##    amplitudes = np.abs(optimal_state)      
##    phases = np.angle(optimal_state)        
#    
##    norm_phases = (phases + np.pi) / (2 * np.pi)   
#    colors = plt.cm.hsv(norm_phases)               
#    
#    plt.figure(figsize=(12, 6))
#    x = np.arange(len(optimal_states))
#    bars = plt.bar(x, amplitudes, color=colors, edgecolor='black', linewidth=0.5)
#    
#    plt.xlabel(r'计算基态 $|i\rangle$', fontsize=12)
#    plt.ylabel('概率幅模长 $|c_i|$', fontsize=12)
#    plt.title('最优态在计算基下的展开 (颜色表示相位)', fontsize=14)
#    
#    n_states = len(optimal_states)
#    max_ticks = 20   
#    if n_states <= max_ticks:
#        plt.xticks(x, [f'$|{i}\\rangle$' for i in x], rotation=45)
#    else:
#        step = n_states // max_ticks
#        ticks = np.arange(0, n_states, step)
#        plt.xticks(ticks, [f'$|{i}\\rangle$' for i in ticks], rotation=45)
#    
#    sm = plt.cm.ScalarMappable(cmap='hsv', norm=plt.Normalize(-np.pi, np.pi))
#    sm.set_array(phases)
#    cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.8)
#    cbar.set_label('相位 (rad)', fontsize=10)
#    
#    plt.tight_layout()
#    hist_filename = filename + "_histogram.png"
#    plt.savefig(f"./Img/{hist_filename}", dpi=300, bbox_inches='tight')
#    plt.close()
#    
#    print(f"收敛曲线已保存至: ./Img/{filename}.png")
#    print(f"最优态分布直方图已保存至: ./Img/{hist_filename}")
#    print(f"最终 QFI 值: {qfi_values[-1]:.6f}")
#    print("可视化完成！")
#    
#    results = {
#        'optimal_states': optimal_states,
#        'qfi_values': qfi_values,
#        'final_qfi': qfi_values[-1]
#    }
#    
#    return (f"{filename}.png", hist_filename), results

import time  # 确保已导入，若顶部已有可忽略

def run_state_optimization(
    N=8,
    Lambda=1.0,
    g=0.5,
    h=0.1,
    decay_rate=0.1,
    t_max=10.0,
    n_points=2500,
    max_episode=20,
    epsilon=0.01,
    beta1=0.90,
    beta2=0.99,
    use_adam=False
):
    """
    运行量子状态优化 (State Optimization) 并生成QFI收敛曲线和最优态分布图。

    参数:
    -----------
    N : int
        系统维度 (自旋粒子数的两倍)，默认为 8。
    Lambda : float
        哈密顿量中的耦合系数，默认为 1.0。
    g : float
        粒子间的相互作用强度，默认为 0.5。
    h : float
        外场强度，默认为 0.1。
    decay_rate : float
        耗散速率，默认为 0.1。
    t_max : float
        演化总时间，默认为 10.0。
    n_points : int
        时间采样点数，默认为 2500。
    max_episode : int
        最大迭代次数，默认为 20。
    epsilon : float
        学习率，默认为 0.01。
    beta1 : float
        Adam优化器参数1，默认为 0.90。
    beta2 : float
        Adam优化器参数2，默认为 0.99。
    use_adam : bool
        是否使用Adam优化器，默认为 False。

    返回:
    -----------
    filenames : list
        保存的两张图片文件名列表 [收敛曲线图, 直方图]。
    results : dict
        包含 'optimal_states', 'qfi_values' 和 'final_qfi' 的字典。
    """

    print("正在生成可视化结果...")

    # 确保输出目录存在
    os.makedirs("./Img", exist_ok=True)

    # 初始态：自旋相干态
    psi_css = spin_coherent(0.5 * N, 0.5 * np.pi, 0.5 * np.pi, type="ket").full()
    psi_css = psi_css.reshape(1, -1)[0]
    psi0 = [psi_css]

    # 哈密顿量构造
    Jx, Jy, Jz = jmat(0.5 * N)
    Jx, Jy, Jz = Jx.full(), Jy.full(), Jz.full()
    H0 = -Lambda * (np.dot(Jx, Jx) + g * np.dot(Jy, Jy)) / N - h * Jz
    dH = [-Lambda * np.dot(Jy, Jy) / N]

    # 耗散算符
    decay = [[Jz, decay_rate]]

    # 时间序列
    tspan = np.linspace(0., t_max, n_points)

    # 优化参数设置
    AD_paras = {
        "Adam": use_adam,
        "psi0": psi0,
        "max_episode": max_episode,
        "epsilon": epsilon,
        "beta1": beta1,
        "beta2": beta2
    }

    # 执行状态优化
    state = StateOpt(savefile=False, method="AD", **AD_paras)
    state.dynamics(tspan, H0, dH, decay=decay, dyn_method="expm")
    state.QFIM()

    # 读取优化结果
    optimal_states = np.load("states.npy", allow_pickle=True)
    qfi_values = np.loadtxt("f.csv")

    # 确保 optimal_states 为一维复数数组
    if optimal_states.ndim == 2 and optimal_states.shape[0] == 1:
        optimal_states = optimal_states[0]

    # 生成唯一的基础文件名
    base_filename = str(time.time())

    # ---------- 图1：QFI 收敛曲线 ----------
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(qfi_values, 'b-', linewidth=2, label='QFI')
    ax1.set_xlabel('迭代次数', fontsize=12)
    ax1.set_ylabel('QFI 值', fontsize=12)
    ax1.set_title('量子费舍尔信息量优化收敛曲线 (AD算法)', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()

    conv_filename = f"{base_filename}.png"
    plt.savefig(f"./Img/{conv_filename}", dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # ---------- 图2：最优态模长与相位直方图 ----------
    amplitudes = np.abs(optimal_states)
    phases = np.angle(optimal_states)

    norm_phases = (phases + np.pi) / (2 * np.pi)
    colors = plt.cm.hsv(norm_phases)

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    x = np.arange(len(optimal_states))
    bars = ax2.bar(x, amplitudes, color=colors, edgecolor='black', linewidth=0.5)

    ax2.set_xlabel(r'计算基态 $|i\rangle$', fontsize=12)
    ax2.set_ylabel('概率幅模长 $|c_i|$', fontsize=12)
    ax2.set_title('最优态在计算基下的展开 (颜色表示相位)', fontsize=14)

    # 处理 x 轴刻度标签
    n_states = len(optimal_states)
    max_ticks = 20
    if n_states <= max_ticks:
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'$|4,{i-4}\\rangle$' for i in x], rotation=45)
    else:
        step = n_states // max_ticks
        ticks = np.arange(0, n_states, step)
        ax2.set_xticks(ticks)
        ax2.set_xticklabels([f'$|{i}\\rangle$' for i in ticks], rotation=45)

    # 颜色条
    sm = plt.cm.ScalarMappable(cmap='hsv', norm=plt.Normalize(-np.pi, np.pi))
    sm.set_array(phases)
    cbar = fig2.colorbar(sm, ax=ax2, shrink=0.8)
    cbar.set_label('相位 (rad)', fontsize=10)

    plt.tight_layout()
    hist_filename = f"{base_filename}_hist.png"
    plt.savefig(f"./Img/{hist_filename}", dpi=300, bbox_inches='tight')
    plt.close(fig2)

    print(f"收敛曲线已保存至: ./Img/{conv_filename}")
    print(f"最优态直方图已保存至: ./Img/{hist_filename}")
    print(f"最终 QFI 值: {qfi_values[-1]:.6f}")
    print("可视化完成！")

    # 构建返回值
    # filenames = [f"./Img/{conv_filename}", f"./Img/{hist_filename}"]
    filenames = [conv_filename, hist_filename]
    results = {
        'optimal_states': optimal_states,
        'qfi_values': qfi_values,
        'final_qfi': qfi_values[-1]
    }

    return filenames, results


def run_projection_measurement(
    omega=1.0,
    rho0=None,
    tspan=None,
    decay_rates=None,
    DE_paras=None,
    savefig=True,
    show_plots=False,
    output_dir="./Img/"
):

    if rho0 is None:
        rho0 = 0.5*np.array([[1., 1.], [1., 1.]])
    
    if tspan is None:
        tspan = np.linspace(0., 10., 25)
    
    if decay_rates is None:
        decay_rates = [0., 0.1]
    
    if DE_paras is None:
        DE_paras = {"p_num":10, "measurement0":[], "max_episode":1000, \
                    "c":1.0, "cr":0.5, "seed":1234}
    
    sx = np.array([[0., 1.], [1., 0.]])
    sy = np.array([[0., -1.j], [1.j, 0.]]) 
    sz = np.array([[1., 0.], [0., -1.]])
    H0 = 0.5*omega*sz
    dH = [0.5*sz]
    
    sp = np.array([[0., 1.], [0., 0.]])  
    sm = np.array([[0., 0.], [1., 0.]]) 
    decay = [[sp, decay_rates[0]], [sm, decay_rates[1]]]
    
    dim = np.shape(rho0)[0]
    POVM_basis = SIC(dim)
    
    m = MeasurementOpt(mtype="projection", minput=[], savefile=False, method="DE", **DE_paras)
    m.dynamics(tspan, rho0, H0, dH, decay=decay, dyn_method="expm")
    m.CFIM()

    M = np.load("measurements.npy")[-1]
    print(M)
    # 生成统一的基础文件名
    base_filename = str(time.time())

    # ---- 绘制测量矩阵热图 ----
    fig1 = plt.figure(figsize=(15, 5))

    if len(M.shape) == 3:
        for i in range(min(M.shape[0], 4)):
            plt.subplot(1, 4, i + 1)
            plt.imshow(np.real(M[i]), cmap='viridis')
            plt.title(f'Projection Measurement {i+1}')
            plt.colorbar()
    elif len(M.shape) == 2:
        plt.subplot(1, 1, 1)
        plt.imshow(np.real(M), cmap='viridis')
        plt.title('Projection Measurement')
        plt.colorbar()

    plt.tight_layout()

    if savefig:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"{base_filename}.png"))
    if show_plots:
        plt.show()
    plt.close(fig1)

    # ---- 绘制 3D 直方图 (模长 + 相位) ----
    fig2 = plt.figure(figsize=(10, 8))
    ax = fig2.add_subplot(111, projection='3d')

    n = M.shape[0]
    x, y = np.meshgrid(np.arange(n), np.arange(n))
    x = x.flatten()
    y = y.flatten()
    z = np.zeros_like(x)
    dz = np.abs(M.flatten())
    phase = np.angle(M.flatten())

    norm_phase = (phase + np.pi) / (2 * np.pi)
    cmap = plt.cm.hsv
    colors = cmap(norm_phase)

    bar_width = 0.5
    bar_depth = 0.5
    x_corner = x - bar_width / 2
    y_corner = y - bar_depth / 2

    ax.bar3d(
        x_corner, y_corner, z,
        dx=bar_width, dy=bar_depth, dz=dz,
        color=colors, shade=True, alpha=0.9
    )

    ax.set_xlabel('|j> (right vector)')
    ax.set_ylabel('<i| (left vector)')
    ax.set_zlabel('Modulus')
    ax.set_title('3D Histogram: Height = Modulus, Color = Phase')

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels([f'|{i}>' for i in range(n)])
    ax.set_yticklabels([f'<{i}|' for i in range(n)])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-np.pi, np.pi))
    sm.set_array(phase)
    cbar = fig2.colorbar(sm, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Phase (radians)')

    plt.tight_layout()

    if savefig:
        plt.savefig(os.path.join(output_dir, f"{base_filename}_3d.png"))
    if show_plots:
        plt.show()
    plt.close(fig2)

    # 构建结果字典
    results = {
        'measurement': M,
        'type': "Projection Input Measurement"
    }

    return f"{base_filename}_3d.png", results

def run_lc_input_measurement(
    omega=1.0,
    rho0=None,
    tspan=None,
    decay_rates=None,
    DE_paras=None,
    savefig=True,
    show_plots=False,
    output_dir="./Img/"
):
    if rho0 is None:
        rho0 = 0.5*np.array([[1., 1.], [1., 1.]])
    
    if tspan is None:
        tspan = np.linspace(0., 10., 25)
    
    if decay_rates is None:
        decay_rates = [0., 0.1]
    
    if DE_paras is None:
        DE_paras = {"p_num":10, "measurement0":[], "max_episode":1000, \
                    "c":1.0, "cr":0.5, "seed":1234}
    
    sx = np.array([[0., 1.], [1., 0.]])
    sy = np.array([[0., -1.j], [1.j, 0.]]) 
    sz = np.array([[1., 0.], [0., -1.]])
    H0 = 0.5*omega*sz
    dH = [0.5*sz]
    
    sp = np.array([[0., 1.], [0., 0.]])  
    sm = np.array([[0., 0.], [1., 0.]]) 
    decay = [[sp, decay_rates[0]], [sm, decay_rates[1]]]
    
    dim = np.shape(rho0)[0]
    POVM_basis = SIC(dim)
    
    M_num = 2
    m = MeasurementOpt(mtype="input", minput=["LC", POVM_basis, M_num], savefile=False, method="DE", **DE_paras)
    m.dynamics(tspan, rho0, H0, dH, decay=decay, dyn_method="expm")
    m.CFIM()
    M = np.load("measurements.npy")[-1]
    
    # 生成统一的基础文件名
    base_filename = str(time.time())

    # ---- 绘制测量矩阵热图 ----
    fig1 = plt.figure(figsize=(15, 5))

    if len(M.shape) == 3:
        for i in range(min(M.shape[0], 4)):
            plt.subplot(1, 4, i + 1)
            plt.imshow(np.real(M[i]), cmap='viridis')
            plt.title(f'LC Input Measurement {i+1}')
            plt.colorbar()
    elif len(M.shape) == 2:
        plt.subplot(1, 1, 1)
        plt.imshow(np.real(M), cmap='viridis')
        plt.title('LC Input Measurement')
        plt.colorbar()

    plt.tight_layout()

    if savefig:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"{base_filename}.png"))
    if show_plots:
        plt.show()
    plt.close(fig1)

    # ---- 绘制 3D 直方图 (模长 + 相位) ----
    fig2 = plt.figure(figsize=(10, 8))
    ax = fig2.add_subplot(111, projection='3d')

    n = M.shape[0]
    x, y = np.meshgrid(np.arange(n), np.arange(n))
    x = x.flatten()
    y = y.flatten()
    z = np.zeros_like(x)
    dz = np.abs(M.flatten())
    phase = np.angle(M.flatten())

    norm_phase = (phase + np.pi) / (2 * np.pi)
    cmap = plt.cm.hsv
    colors = cmap(norm_phase)

    bar_width = 0.5
    bar_depth = 0.5
    x_corner = x - bar_width / 2
    y_corner = y - bar_depth / 2

    ax.bar3d(
        x_corner, y_corner, z,
        dx=bar_width, dy=bar_depth, dz=dz,
        color=colors, shade=True, alpha=0.9
    )

    ax.set_xlabel('|j> (right vector)')
    ax.set_ylabel('<i| (left vector)')
    ax.set_zlabel('Modulus')
    ax.set_title('3D Histogram: Height = Modulus, Color = Phase')

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels([f'|{i}>' for i in range(n)])
    ax.set_yticklabels([f'<{i}|' for i in range(n)])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-np.pi, np.pi))
    sm.set_array(phase)
    cbar = fig2.colorbar(sm, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Phase (radians)')

    plt.tight_layout()

    if savefig:
        plt.savefig(os.path.join(output_dir, f"{base_filename}_3d.png"))
    if show_plots:
        plt.show()
    plt.close(fig2)

    # 构建结果字典
    results = {
        'measurement': M,
        'type': "LC Input Measurement"
    }

    return f"{base_filename}_3d.png", results

def run_rotation_input_measurement(
    omega=1.0,
    rho0=None,
    tspan=None,
    decay_rates=None,
    DE_paras=None,
    savefig=True,
    show_plots=False,
    output_dir="./Img/"
):
    """
    运行量子旋转输入测量优化，并生成测量矩阵的热图与 3D 直方图。

    Parameters:
    -----------
    omega : float
        频率参数，默认为1.0
    rho0 : numpy.ndarray or None
        初始密度矩阵，默认为0.5*np.array([[1., 1.], [1., 1.]])
    tspan : numpy.ndarray or None
        时间范围数组，默认为np.linspace(0., 10., 25)
    decay_rates : list or None
        衰减率列表[gamma1, gamma2]，默认为[0., 0.1]
    DE_paras : dict or None
        差分进化参数字典，默认包含p_num=10, max_episode=1000等
    savefig : bool
        是否保存图像，默认为True
    show_plots : bool
        是否显示图像，默认为False
    output_dir : str
        图像输出目录，默认为"./Img/"

    Returns:
    --------
    filename : str
        保存的图片基础文件名（不含扩展名）
    results : dict
        包含优化结果，如 'measurement' 测量矩阵，'type' 测量类型
    """

    if rho0 is None:
        rho0 = 0.5 * np.array([[1., 1.], [1., 1.]])

    if tspan is None:
        tspan = np.linspace(0., 10., 25)

    if decay_rates is None:
        decay_rates = [0., 0.1]

    if DE_paras is None:
        DE_paras = {
            "p_num": 10,
            "measurement0": [],
            "max_episode": 1000,
            "c": 1.0,
            "cr": 0.5,
            "seed": 1234
        }

    sx = np.array([[0., 1.], [1., 0.]])
    sy = np.array([[0., -1.j], [1.j, 0.]])
    sz = np.array([[1., 0.], [0., -1.]])

    H0 = 0.5 * omega * sz
    dH = [0.5 * sz]

    sp = np.array([[0., 1.], [0., 0.]])
    sm = np.array([[0., 0.], [1., 0.]])
    decay = [[sp, decay_rates[0]], [sm, decay_rates[1]]]

    dim = np.shape(rho0)[0]
    POVM_basis = SIC(dim)

    m = MeasurementOpt(
        mtype="input",
        minput=["rotation", POVM_basis],
        savefile=False,
        method="DE",
        **DE_paras
    )
    m.dynamics(tspan, rho0, H0, dH, decay=decay, dyn_method="expm")
    m.CFIM()
    M = np.load("measurements.npy")[-1]
    print(M)
    # 生成统一的基础文件名
    base_filename = str(time.time())

    # ---- 绘制测量矩阵热图 ----
    fig1 = plt.figure(figsize=(15, 5))

    if len(M.shape) == 3:
        for i in range(min(M.shape[0], 4)):
            plt.subplot(1, 4, i + 1)
            plt.imshow(np.real(M[i]), cmap='viridis')
            plt.title(f'Rotation Input Measurement {i+1}')
            plt.colorbar()
    elif len(M.shape) == 2:
        plt.subplot(1, 1, 1)
        plt.imshow(np.real(M), cmap='viridis')
        plt.title('Rotation Input Measurement')
        plt.colorbar()

    plt.tight_layout()

    if savefig:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"{base_filename}.png"))
    if show_plots:
        plt.show()
    plt.close(fig1)

    # ---- 绘制 3D 直方图 (模长 + 相位) ----
    fig2 = plt.figure(figsize=(10, 8))
    ax = fig2.add_subplot(111, projection='3d')

    n = M.shape[0]
    x, y = np.meshgrid(np.arange(n), np.arange(n))
    x = x.flatten()
    y = y.flatten()
    z = np.zeros_like(x)
    dz = np.abs(M.flatten())
    phase = np.angle(M.flatten())

    norm_phase = (phase + np.pi) / (2 * np.pi)
    cmap = plt.cm.hsv
    colors = cmap(norm_phase)

    bar_width = 0.5
    bar_depth = 0.5
    x_corner = x - bar_width / 2
    y_corner = y - bar_depth / 2

    ax.bar3d(
        x_corner, y_corner, z,
        dx=bar_width, dy=bar_depth, dz=dz,
        color=colors, shade=True, alpha=0.9
    )

    ax.set_xlabel('|j> (right vector)')
    ax.set_ylabel('<i| (left vector)')
    ax.set_zlabel('Modulus')
    ax.set_title('3D Histogram: Height = Modulus, Color = Phase')

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels([f'|{i}>' for i in range(n)])
    ax.set_yticklabels([f'<{i}|' for i in range(n)])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-np.pi, np.pi))
    sm.set_array(phase)
    cbar = fig2.colorbar(sm, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Phase (radians)')

    plt.tight_layout()

    if savefig:
        plt.savefig(os.path.join(output_dir, f"{base_filename}_3d.png"))
    if show_plots:
        plt.show()
    plt.close(fig2)

    # 构建结果字典
    results = {
        'measurement': M,
        'type': "Rotation Input Measurement"
    }

    return f"{base_filename}_3d.png", results