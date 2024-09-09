import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB 
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# Define the function for interpolation
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


#parameter configuration
#unit:kW
P_idle=0.4
P_peak=0.75
eta=1.75
cooling_coef=0.972
cooling_g=1/200
cooling_COP=10

P_grid_max=30

L_rate=7
C_DT=0.32
A_max=2.5e3
T=96
TD=16   #16个15分钟，即四小时
C_RES=32
C_reduce_max=1
C_reduce_min=0
C_BW=2.2/1e2
C_IW=1.8/1e2
C_GC=100
C_punish=250
redundant_ratio=0.01

#unit:MWh
kk=2.5
delta_W_res_fix=30*kk
W_res_fix=176*kk
W_res_max=206*kk

s_reg=0.987
R_mil=2.92

P_res_max=np.array([0,0,0,0,0,0,10,14,18,20,22,24,25,23,20,17,13,0,0,0,0,0,0,0])*kk
electricity_price=np.array([29,26,23,23,24,25,31,30,32,33,34,35,37,41,44,52,62,50,40,36,38,34,32,30])
reserve_price=np.array([10,8,7,6,7,6,12,6,6,6,7,6,8,10,13,22,31,21,13,10,12,8,10,9])


alpha=np.array([-0.6,-0.6,-0.8,-0.3,-0.3,-0.2,0,0.25,0.25,0.5,0.5,0.5,0.5,0.5,0.8,0.8,0.7,0.5,0.5,0.5,0.93,0.2,0.1,0])
beta=np.array([-0.25,-0.12,0.2,-0.05,-0.1,-0.15,0.2,-0.45,0.2,-0.1,0,0.1,0,-0.12,0.05,0.1,-0.15,-0.13,0.25,0,-0.2,0.25,0.1,-0.1])
T_env=np.array([26,25,24,23,22,23,
                24,25,26,27,28,30,
                32,34,34,32,31,30,
                30,29,28,27,26,26])
T_env=interpolate_data(T_env)
Cp=1.00545
m_air=2.26



