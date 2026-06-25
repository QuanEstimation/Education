from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
import numpy as np
import time
import os
import quantum_estimation
# 引入量子估计函数
from quantum_estimation import run_quantum_estimation 
from quantum_estimation import run_bayesian_quantum_estimation
from quantum_estimation import run_quantum_parameter_estimation
from quantum_estimation import run_projection_measurement, run_lc_input_measurement, run_rotation_input_measurement
from quantum_estimation import run_state_optimization

app = FastAPI()

# 自动创建必要文件夹
os.makedirs("./Img", exist_ok=True)
os.makedirs("./Doc", exist_ok=True)

# 挂载静态文件 (保持不变)
app.mount("/Img", StaticFiles(directory="./Img"), name="Img")
# 挂载静态文件 (保持不变)
app.mount("/Doc", StaticFiles(directory="./Doc"), name="Doc")

# --- 原有接口：保持不变 ---
@app.get("/createImg", response_class=HTMLResponse)
async def create_img(
    omega: float = Query(1.0, description="系统本征频率 (omega)"),
    t_end: float = Query(50.0, description="演化时间终点 (t_end)"),
    steps: int = Query(2000, description="时间步数 (steps)"),
    cam: float = Query(0.5, description="耗散系数 (cam): 0~1")
):
    # 1. 根据参数生成 tspan
    tspan = np.linspace(0., t_end, steps)
    
    
    # 2. 调用修改后的 MakeImg 函数
    imgname = quantum_estimation.MakeImg(omega=omega, tspan=tspan, cam=cam)
    
    # 3. 返回 HTML 图片标签
    imgsrc = f"/Img/{imgname}"
    return f"<img src=\"{imgsrc}\" />"

# --- 新增接口：支持 Tab2 (双量子比特系统) ---
@app.get("/createImgQuantum", response_class=HTMLResponse)
async def create_img_quantum(
    omega1: float = Query(1.0, description="粒子1频率"),
    omega2: float = Query(1.0, description="粒子2频率"),
    g: float = Query(0.1, description="耦合强度"),
    gamma: float = Query(0.05, description="耗散速率 (gamma1=gamma2)"),
    theta: float = Query(0.79, description="初始态角度"),
    t_max: float = Query(10.0, description="最大演化时间"),
    n_points: int = Query(500, description="采样点数")
):
    """
    处理双量子比特参数估计请求。
    调用 run_quantum_estimation 函数并返回图片 HTML。
    """
    try:
        # 调用 Python 函数
        filename, _ = run_quantum_estimation(
            omega1=omega1,
            omega2=omega2,
            g=g,
            gamma1=gamma,      
            gamma2=gamma,      
            theta=theta,
            t_max=t_max,
            n_points=n_points
        )
        
        # 构建图片 URL
        imgsrc = f"/Img/{filename}"
        
        # 返回 HTML 图片标签
        return f"<img src=\"{imgsrc}\" />"
        
    except Exception as e:
        raise e

# --- 【新增】接口：支持 Tab3 (贝叶斯量子参数估计) ---
@app.get("/createImgBayesian", response_class=HTMLResponse)
async def create_img_bayesian(
    eta: float = Query(0.2, description="先验不确定度 (eta)"),
    t_end: float = Query(1.0, description="演化总时间 (t_end)"),
    B_factor: float = Query(0.5, description="磁场强度因子 (B_factor, 单位pi)"),
    omega0: float = Query(1.0, description="特征频率 (omega0)"),
    mu: float = Query(0.0, description="先验均值 (mu)"),
    x_range_factor: float = Query(0.5, description="采样范围系数"),
    n_x_points: int = Query(100, description="x轴采样点数"),
    n_t_points: int = Query(1000, description="时间采样点数")
):
    """
    处理贝叶斯量子参数估计请求 (Tab 3)。
    调用 run_bayesian_quantum_estimation 函数并返回图片 HTML。
    """
    if run_bayesian_quantum_estimation is None:
        raise RuntimeError("贝叶斯估计函数未找到，请检查导入路径。")

    try:
        # 将前端的 B_factor (数值) 转换为物理值 (乘以 pi)
        # 文档中函数定义 B_factor 参数直接接受数值，但前端注释提到 "单位: π"
        # 查看文档函数定义: def run_bayesian_quantum_estimation(..., B_factor=0.5*np.pi, ...)
        # 前端传递的是 0.5 (代表 0.5pi)，所以这里需要乘以 np.pi
        B_physical = B_factor * np.pi
        
        # 调用 Python 函数
        # 注意：文档中该函数只返回文件名字符串
        filename = run_bayesian_quantum_estimation(
            eta=eta,
            t_end=t_end,
            B_factor=B_physical,  # 传入物理值
            omega0=omega0,
            mu=mu,
            x_range_factor=x_range_factor,
            n_x_points=n_x_points,
            n_t_points=n_t_points
        )
        
        # 构建图片 URL
        imgsrc = f"/Img/{filename}"
        
        # 返回 HTML 图片标签
        return f"<img src=\"{imgsrc}\" />"
        
    except Exception as e:
        # 抛出异常让前端捕获
        raise e 

# --- 【新增】接口：支持 Tab4 (量子参数估计 - 贝叶斯和最大似然估计) ---
@app.get("/createImgQuantumParameter", response_class=HTMLResponse)
async def create_img_quantum_parameter(
    B_factor: float = Query(0.5, description="磁场强度因子 (B_factor, 单位pi)"),
    omega0: float = Query(1.0, description="角频率 (omega0)"),
    x_min: float = Query(0.0, description="参数 x 的最小值 (单位pi)"),
    x_max_factor: float = Query(0.5, description="参数 x 的最大值因子 (单位pi)"),
    x_num: int = Query(1000, description="参数 x 的采样点数"),
    t_max: float = Query(1.0, description="时间演化的最大值"),
    t_num: int = Query(1000, description="时间演化的采样点数"),
    y_num: int = Query(1000, description="测量次数"),
    seed: int = Query(1234, description="随机种子"),
    prob_threshold: float = Query(0.333, description="测量结果的概率阈值")
):
    """
    处理量子参数估计请求 (Tab 4)。
    调用 run_quantum_parameter_estimation 函数并返回图片 HTML。
    """
    if run_quantum_parameter_estimation is None:
        raise RuntimeError("量子参数估计函数未找到，请检查导入路径。")

    try:
        B_physical = B_factor * np.pi
        x_min_physical = x_min * np.pi
        x_max_physical = x_max_factor * np.pi
        
        filename, _ = run_quantum_parameter_estimation(
            B=B_physical,
            omega0=omega0,
            x_min=x_min_physical,
            x_max=x_max_physical,
            x_num=x_num,
            t_max=t_max,
            t_num=t_num,
            y_num=y_num,
            seed=seed,
            prob_threshold=prob_threshold
        )
        
        imgsrc = f"/Img/{filename}"
        
        return f"<img src=\"{imgsrc}\" />"
        
    except Exception as e:
        raise e 

# --- 【新增】接口：支持 Tab5 (测量优化算法) ---
# --- 【新增】接口：支持 Tab5-1 (投影测量优化) ---
@app.get("/createImgProjection", response_class=HTMLResponse)
async def create_img_projection(
    omega: float = Query(1.0, description="系统频率 (omega)"),
    t_end: float = Query(10.0, description="演化时间终点 (t_end)"),
    t_points: int = Query(25, description="时间采样点数 (t_points)"),
    decay_rate_1: float = Query(0.0, description="衰减率1 (decay_rate_1)"),
    decay_rate_2: float = Query(0.1, description="衰减率2 (decay_rate_2)"),
    max_episode: int = Query(1000, description="最大迭代次数 (max_episode)"),
    p_num: int = Query(10, description="粒子数量 (p_num)")
):
    """
    处理投影测量优化请求 (Tab 5-1)。
    调用 run_projection_measurement 函数并返回两张图片的 URL。
    """
    if run_projection_measurement is None:
        raise RuntimeError("投影测量优化函数未找到，请检查导入路径。")

    try:
        tspan = np.linspace(0., t_end, t_points)
        
        decay_rates = [decay_rate_1, decay_rate_2]
        
        DE_paras = {
            "p_num": p_num,
            "measurement0": [],
            "max_episode": max_episode,
            "c": 1.0,
            "cr": 0.5,
            "seed": 1234
        }
        
        filename, _  = run_projection_measurement(
            omega=omega,
            tspan=tspan,
            decay_rates=decay_rates,
            DE_paras=DE_paras,
            savefig=True,
            show_plots=False,
            output_dir="./Img/"
        )
        print(filename)
        imgsrc = f"/Img/{filename}"
        print(imgsrc)
        return f"<img src=\"{imgsrc}\" />"
        
    except Exception as e:
        raise e 

# --- 【新增】接口：支持 Tab5-2 (LC输入测量优化) ---
@app.get("/createImgLCInput", response_class=HTMLResponse)
async def create_img_lc_input(
    omega: float = Query(1.0, description="系统频率 (omega)"),
    t_end: float = Query(10.0, description="演化时间终点 (t_end)"),
    t_points: int = Query(25, description="时间采样点数 (t_points)"),
    decay_rate_1: float = Query(0.0, description="衰减率1 (decay_rate_1)"),
    decay_rate_2: float = Query(0.1, description="衰减率2 (decay_rate_2)"),
    max_episode: int = Query(1000, description="最大迭代次数 (max_episode)"),
    p_num: int = Query(10, description="粒子数量 (p_num)")
):
    """
    处理LC输入测量优化请求 (Tab 5-2)。
    调用 run_lc_input_measurement 函数并返回图片 HTML。
    """
    if run_lc_input_measurement is None:
        raise RuntimeError("LC输入测量优化函数未找到，请检查导入路径。")

    try:
        tspan = np.linspace(0., t_end, t_points)
        
        decay_rates = [decay_rate_1, decay_rate_2]
        
        DE_paras = {
            "p_num": p_num,
            "measurement0": [],
            "max_episode": max_episode,
            "c": 1.0,
            "cr": 0.5,
            "seed": 1234
        }
        
        filename, _ = run_lc_input_measurement(
            omega=omega,
            tspan=tspan,
            decay_rates=decay_rates,
            DE_paras=DE_paras,
            savefig=True,
            show_plots=False
        )
        
        imgsrc = f"/Img/{filename}"
        
        return f"<img src=\"{imgsrc}\" />"
        
    except Exception as e:
        raise e 

# --- 【新增】接口：支持 Tab5-3 (旋转输入测量优化) ---
@app.get("/createImgRotationInput", response_class=HTMLResponse)
async def create_img_rotation_input(
    omega: float = Query(1.0, description="系统频率 (omega)"),
    t_end: float = Query(10.0, description="演化时间终点 (t_end)"),
    t_points: int = Query(25, description="时间采样点数 (t_points)"),
    decay_rate_1: float = Query(0.0, description="衰减率1 (decay_rate_1)"),
    decay_rate_2: float = Query(0.1, description="衰减率2 (decay_rate_2)"),
    max_episode: int = Query(1000, description="最大迭代次数 (max_episode)"),
    p_num: int = Query(10, description="粒子数量 (p_num)")
):
    """
    处理旋转输入测量优化请求 (Tab 5-3)。
    调用 run_rotation_input_measurement 函数并返回图片 HTML。
    """
    if run_rotation_input_measurement is None:
        raise RuntimeError("旋转输入测量优化函数未找到，请检查导入路径。")

    try:
        tspan = np.linspace(0., t_end, t_points)
        
        decay_rates = [decay_rate_1, decay_rate_2]
        
        DE_paras = {
            "p_num": p_num,
            "measurement0": [],
            "max_episode": max_episode,
            "c": 1.0,
            "cr": 0.5,
            "seed": 1234
        }
        
        filename, _ = run_rotation_input_measurement(
            omega=omega,
            tspan=tspan,
            decay_rates=decay_rates,
            DE_paras=DE_paras,
            savefig=True,
            show_plots=False
        )
        
        imgsrc = f"/Img/{filename}"
        
        return f"<img src=\"{imgsrc}\" />"
        
    except Exception as e:
        raise e 

# --- 【新增】接口：支持 Tab6 (状态优化算法) ---

@app.get("/createImgStateOpt")
async def create_img_state_opt(
    N: int = Query(8, description="系统维度 (N)"),
    Lambda: float = Query(1.0, description="耦合系数 (Lambda)"),
    g: float = Query(0.5, description="相互作用强度 (g)"),
    h: float = Query(0.1, description="外场强度 (h)"),
    decay_rate: float = Query(0.1, description="耗散速率 (decay_rate)"),
    t_max: float = Query(10.0, description="演化时间 (t_max)"),
    n_points: int = Query(2500, description="时间采样点数 (n_points)"),
    max_episode: int = Query(20, description="最大迭代次数 (max_episode)"),
    epsilon: float = Query(0.01, description="学习率 (epsilon)"),
    beta1: float = Query(0.90, description="Adam参数1 (beta1)"),
    beta2: float = Query(0.99, description="Adam参数2 (beta2)"),
    use_adam: bool = Query(False, description="是否使用Adam优化器 (use_adam)")
):
    """
    处理状态优化算法请求 (Tab 6)。
    调用 run_state_optimization 函数并返回两张图片的 URL。
    """
    if run_state_optimization is None:
        raise RuntimeError("状态优化函数未找到，请检查导入路径。")

    try:
        filenames, _ = run_state_optimization(
            N=N,
            Lambda=Lambda,
            g=g,
            h=h,
            decay_rate=decay_rate,
            t_max=t_max,
            n_points=n_points,
            max_episode=max_episode,
            epsilon=epsilon,
            beta1=beta1,
            beta2=beta2,
            use_adam=use_adam
        )
        
        imgsrc1 = f"/Img/{filenames[0]}"
        imgsrc2 = f"/Img/{filenames[1]}"
        
        return JSONResponse(content={
            "image1": imgsrc1,
            "image2": imgsrc2
        })
        
    except Exception as e:
        raise e 

# --- 原有根路由：保持不变 ---
@app.get("/")
async def root():
    return FileResponse("./index.html")