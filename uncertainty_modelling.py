import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.optimize import minimize_scalar
from sklearn.mixture import GaussianMixture
import pandas as pd
from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt
from scipy.stats import norm
import warnings
from scipy.interpolate import interp1d

warnings.filterwarnings('ignore')

'''
 4.1.添加机会约束相关函数，现在先只写一个在这里放着
 假设每个时刻的IW实际值满足以预测值为均值,预测值的2%为标准差的正态分布
 4.2.利用一组数据负荷预测误差的直方图分布，用EM算法求出其GMM模型，再利用ppf函数反解分位点
 7.9 直接抄原来的代码
'''
def solve_quantile(mean,std,epsilon):
    standard_normal_quantile = norm.ppf(1-epsilon)
    quantile=mean+std*standard_normal_quantile
    return quantile


def gmm_pdf(x, weights, means, covariances):
    """
    计算GMM在点x处的概率密度函数值
    """
    pdf_values = [weights[i] * multivariate_normal.pdf(x, mean=means[i], cov=covariances[i]) 
                  for i in range(len(weights))]
    return np.sum(pdf_values)

def gmm_quantile(probability, weights, means, covariances, n_samples=100000):
    """
    计算给定置信水平下的分位点
    """
    dim = len(means[0])
    
    # 从GMM中抽样
    samples = np.concatenate([multivariate_normal.rvs(mean=mean, cov=cov, size=int(weights[i]*n_samples))
                             for i, (weight, mean, cov) in enumerate(zip(weights, means, covariances))])

    # 定义目标函数，寻找满足累计概率等于指定置信水平的分位点
    def objective_function(x):
        return np.abs(np.cumsum(gmm_pdf(samples, weights, means, covariances).reshape(-1, 1) <= x) - probability)

    # 初始猜测值，可以选择样本集的中位数
    initial_guess = np.median(samples, axis=0)

    # 使用优化算法找到分位点
    result = minimize_scalar(objective_function, bounds=[0, 1], method='bounded', args=(dim,))
    
    # 返回找到的分位点
    quantile_point = result.x
    return quantile_point

def fit_gmm(data, n_components, max_iter=100, covariance_type='diag'):
    """
    使用scikit-learn的GaussianMixture模型通过EM算法拟合高斯混合模型
    :param data: 输入的二维numpy数组，每一行代表一个样本
    :param n_components: 高斯混合模型的组件数
    :param max_iter: EM算法的最大迭代次数，默认为100
    :param covariance_type: 协方差矩阵类型，可选['full', 'tied', 'diag', 'spherical']
    :return: 拟合好的GaussianMixture模型对象
    """
    # 初始化GMM模型
    gmm = GaussianMixture(n_components=n_components, max_iter=max_iter, covariance_type=covariance_type)
    # 使用数据拟合模型
    gmm.fit(data)
    return gmm

def EM_fit_withN(X,N):
    # 创建并初始化GMM模型，假设数据由3个高斯分布混合而成
    gmm = GaussianMixture(n_components=N, random_state=42)
    # 使用EM算法拟合数据
    gmm.fit(X)
    return gmm.means_,gmm.covariances_,gmm.weights_

def EM_fit(X):
    # 可能的分量数量列表
    n_components = range(1, 11)  # 假设我们想尝试从1到9个分量
    # 设置参数网格
    param_grid = {'n_components': n_components}
    # 创建GMM模型对象
    
    # 网格搜索寻找最佳参数
    aics,bics=[],[]
    for i in n_components:
        gmm = GaussianMixture(n_components=i,covariance_type='diag')
        gmm.fit(X)
        aics.append(gmm.aic(X))
        bics.append(gmm.bic(X))
        best_n_components_bic = n_components[np.argmin(bics)]
        best_n_components_aic = n_components[np.argmin(aics)]
    # 以BIC为标准选取最合适的K值
    best_k=best_n_components_bic 
    print(f"最优的高斯分布数量: {best_k}")
    # 最后你可以用这个最优的K值重新拟合GMM模型
    best_gmm = GaussianMixture(n_components=best_k)
    best_gmm.fit(X)
    return best_gmm,best_gmm.means_,best_gmm.covariances_,best_gmm.weights_

def Draw_EM_fit(best_model,X):
    # 计算每个样本点的新概率密度
    densities = best_model.predict_proba(X)  # 每个样本点对于各个高斯分布的概率

    # 绘制原始数据的概率密度曲线
    plt.hist(X, bins='auto', density=True, alpha=0.5, label='Original Data Density')

    # 创建网格用于绘制概率密度函数
    x_grid = np.linspace(X.min(), X.max(), 1000)
    grid_densities = best_model.score_samples(x_grid[:, np.newaxis])

    # 将概率密度函数转换为概率密度
    pdf = np.exp(grid_densities) / (np.sum(np.exp(grid_densities)) * (x_grid[1] - x_grid[0]))

    # 绘制新的概率密度曲线
    plt.plot(x_grid, pdf, label='Fitted GMM Density')

    plt.legend()
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Original Data Density vs Fitted GMM Density')
    plt.show()

def gmm_cdf(x, weights, means, covariances):
    total_cdf = 0
    for weight, mean, covariance in zip(weights, means, covariances):
        # 计算单个高斯分布的累积分布函数
        single_gaussian_cdf = norm.cdf(x, loc=mean, scale=np.sqrt(np.diag(covariance)).squeeze())
        # 累加所有高斯分量的累积分布函数
        total_cdf += weight * single_gaussian_cdf
    return total_cdf

def solve_gmm_quantile(data,weights,means,covariances,epsilon,n_samples=20000):
    x_points=np.linspace(data.min(),data.max(),n_samples)
    approaching_target=1-epsilon
    for i in range(n_samples):
        if abs(gmm_cdf(x_points[i],weights,means,covariances)-approaching_target)<=abs(gmm_cdf(x_points[i+1],weights,means,covariances)-approaching_target):
            return x_points[i]

def get_96point_WT_quantiles(forecast_value,real_value,confidence_level):
    WT_quantiles=[]
    for i in range(96):
        delta_WT=real_value[i]-forecast_value[i]
        delta_WT=np.expand_dims(delta_WT,axis=1)
        gmm_WT,gmm_WT.means_,gmm_WT.covariances_,gmm_WT.weights_=EM_fit(delta_WT)
        WT_quantile=solve_gmm_quantile(delta_WT,gmm_WT.weights_,gmm_WT.means_,gmm_WT.covariances_,confidence_level)
        WT_quantiles.append(WT_quantile)
        print(f'第{i+1}个分位点计算完成',i)
    return np.array(WT_quantiles)

#写一个函数，把这些数据分成96份，对应一天中96个时间点，并输出每个时间点对应的GMM分位点
def get_96_subarrays(data):
    if not isinstance(data, np.ndarray):
            raise ValueError("Input must be a NumPy array.")
    
    # 获取数组的长度
    length = len(data)
    
    # 创建一个空列表，用于存储子数组
    subarrays = []
    
    # 循环96次，每次创建一个子数组
    for i in range(96):
        # 计算这个子数组的索引
        indices = slice(i, length, 96)
        
        # 使用索引从原始数组中提取子数组
        subarray = data[indices]
        
        # 将子数组添加到列表中
        subarrays.append(subarray)
    
    return subarrays

#下面求出96点中每个时间点的WT、数据处理负荷预测偏差的分位点

def get_96point_PV_quantiles(forecast_value,real_value,confidence_level):
    PV_quantiles=[]
    for i in range(96):
        delta_PV=real_value[i]-forecast_value[i]
        delta_PV=np.expand_dims(delta_PV,axis=1)
        gmm_PV,gmm_PV.means_,gmm_PV.covariances_,gmm_PV.weights_=EM_fit(delta_PV)
        PV_quantile=solve_gmm_quantile(delta_PV,gmm_PV.weights_,gmm_PV.means_,gmm_PV.covariances_,confidence_level)
        PV_quantiles.append(PV_quantile)
        print(f'第{i+1}个分位点计算完成:',PV_quantile)
    return np.array(PV_quantiles)

def get_96point_load_quantiles(forecast_value,real_value,confidence_level):
    load_quantiles=[]
    for i in range(96):
        delta_load=forecast_value[i]-real_value[i]
        delta_load=delta_load[(delta_load >= np.percentile(delta_load, 25) - 1.5 * (np.percentile(delta_load, 75) - np.percentile(delta_load, 25))) & 
                        (delta_load<= np.percentile(delta_load, 75) + 1.5 * (np.percentile(delta_load, 75) - np.percentile(delta_load, 25)))]
        delta_load=np.expand_dims(delta_load,axis=1)
        gmm_load,gmm_load.means_,gmm_load.covariances_,gmm_load.weights_=EM_fit(delta_load)
        load_quantile=solve_gmm_quantile(delta_load,gmm_load.weights_,gmm_load.means_,gmm_load.covariances_,confidence_level)
        load_quantiles.append(load_quantile)
        print(f'第{i+1}个分位点计算完成: ',load_quantile)
    return np.array(load_quantiles)

def interpolate_data(data):
    # Ensure input data length is 24
    if len(data) != 24:
        raise ValueError("Input array must have exactly 24 elements.")
    
    # Original x coordinates, from 0 to 23
    x_original = np.arange(24)
    
    # New x coordinates, from 0 to 95, total 96 points
    x_new = np.linspace(0, 24, 96)
    
    # Use linear interpolation function
    f = interp1d(x_original, data, kind='linear', bounds_error=False, fill_value="extrapolate")
    
    # Interpolate and get new data
    interpolated_data = f(x_new[:-3])  # Handle the first 93 points
    
    # Special handling for the last three points, using the first and last point of the original array for interpolation
    last_point = data[-1]
    first_point = data[0]
    extended_x = np.array([24, 25, 26])
    extended_y = np.array([last_point, first_point, first_point])
    
    # Extended interpolation function
    f_extended = interp1d(extended_x, extended_y, kind='linear', bounds_error=False, fill_value="extrapolate")
    
    # Get the data for the last three points
    last_three_points = f_extended(x_new[-3:])
    
    # Combine the two parts of data
    final_data = np.concatenate((interpolated_data, last_three_points))
    
    return final_data

WT_data=pd.read_csv('ods031.csv').fillna(0)
WT_forecast=np.flip(WT_data['Measured & Upscaled'].to_numpy())*2
WT_real_time=np.flip(WT_data['Most recent forecast'].to_numpy())*2


WT_96point_forecast=get_96_subarrays(WT_forecast)
WT_96point_real_time=get_96_subarrays(WT_real_time)



WT_96points_quantiles=get_96point_WT_quantiles(WT_96point_forecast,WT_96point_real_time,1-0.05)


load_data=pd.read_csv('ods001.csv').fillna(0)
load_forecast=np.flip(load_data['Most recent forecast'].to_numpy())/10
load_real_time=np.flip(load_data['Total Load'].to_numpy())/10


load_96point_forecast=get_96_subarrays(load_forecast)
load_96point_real_time=get_96_subarrays(load_real_time)



load_96points_quantiles=get_96point_load_quantiles(load_96point_forecast,load_96point_real_time,0.05)


WT_simulation=np.flip(WT_data['Measured & Upscaled'][-96:].to_numpy())*2

PV_data=pd.read_csv('ods032.csv').fillna(0)
PV_forecast=np.flip(PV_data['Measured & Upscaled'][:-96].to_numpy())
PV_real_time=np.flip(PV_data['Most recent forecast'][:-96].to_numpy())

PV_96point_forecast=get_96_subarrays(PV_forecast)
PV_96point_real_time=get_96_subarrays(PV_real_time)



PV_96points_quantiles=get_96point_PV_quantiles(PV_96point_forecast,PV_96point_real_time,1-0.05)
PV_simulation=np.flip(PV_data['Measured & Upscaled'][-96:].to_numpy())
