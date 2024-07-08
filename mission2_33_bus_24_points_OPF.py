import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import seaborn as sns
import matplotlib.pyplot as plt
from data import *

#任务2 以最优潮流问题为基础的24点优化调度
#负荷比例：24个小时中，每个时间点每个节点的有功/无功负荷相对基准值（即开头表中值）的比例
#负荷根据实际需要任意修改即可，这里只做实验用
pq_ratio=np.array([0.8,0.7,0.6,0.5,0.4,0.45,0.5,0.55,0.65,0.75,0.85,1.1,
                   1.2,1.3,1.4,1.5,1.5,1.6,1.7,1.55,1.35,1.15,1.05,0.95])
model2=gp.Model('24 point OPF')
#声明各个决策变量，每一个24维的变量表示该节点/电机/线路24个小时中每小时的功率
gen_Ps,gen_Qs,bus_square_voltages,branch_P,branch_Q,branch_square_currents=[],[],[],[],[],[]
for i in range(3):
    tmp_p=model2.addVars(24,lb=0,ub=2000,vtype=GRB.CONTINUOUS,name=f'generator_p_{i+1}')
    tmp_q=model2.addVars(24,lb=0,ub=2000,vtype=GRB.CONTINUOUS,name=f'generator_q_{i+1}')
    gen_Ps.append(tmp_p)
    gen_Qs.append(tmp_q)
for i in range(32):
    tmp_bus_square_voltage=model2.addVars(24,lb=Vmin*Vmin,ub=Vmax*Vmax,vtype=GRB.CONTINUOUS,name=f'bus_square_voltage_{i+1}')
    tmp_branch_p=model2.addVars(24,lb=-GRB.INFINITY,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name=f'branch_P_{i+1}')
    tmp_branch_q=model2.addVars(24,lb=-GRB.INFINITY,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name=f'branch_Q_{i+1}')
    tmp_branch_square_current=model2.addVars(24,lb=0,ub=4e4,vtype=GRB.CONTINUOUS,name=f'branch_square_{i+1}')
    bus_square_voltages.append(tmp_bus_square_voltage)
    branch_P.append(tmp_branch_p)
    branch_Q.append(tmp_branch_q)
    branch_square_currents.append(tmp_branch_square_current) 
    
#发电机爬坡功率约束,设发电机爬坡能力为400kW/h
for i in range(3):
    for j in range(1,24):
        model2.addConstr(gen_Ps[i][j]-gen_Ps[i][j-1]<=400)
        model2.addConstr(gen_Ps[i][j]-gen_Ps[i][j-1]>=-400)
        model2.addConstr(gen_Qs[i][j]-gen_Qs[i][j-1]<=400)
        model2.addConstr(gen_Qs[i][j]-gen_Qs[i][j-1]>=-400)
#主线
#节点2
#后一个节点电压=前一个节点电压和功率、电流的关系
#后面统一循环变量i表示时间点，循环变量j表示节点
#发电机的懒得改了
for i in range(24):
    model2.addConstr(1e6*bus_square_voltages[0][i]==1e6*Vm*Vm-2*1e3*(branch_r[0]*branch_P[0][i]+branch_x[0]*branch_Q[0][i])
                    +(branch_r[0]*branch_r[0]+branch_x[0]*branch_x[0])*branch_square_currents[0][i])
    #流入节点功率+节点自身功率=流出节点功率+负荷
    model2.addConstr(branch_P[0][i]+gen_Ps[0][i]-branch_square_currents[0][i]*branch_r[0]/1e3==bus_Pd[0]*pq_ratio[i]+branch_P[1][i]+branch_P[17][i])
    model2.addConstr(branch_Q[0][i]+gen_Qs[0][i]-branch_square_currents[0][i]*branch_x[0]/1e3==bus_Qd[0]*pq_ratio[i]+branch_Q[1][i]+branch_Q[17][i])
    #线上电流关系
    model2.addQConstr(4*branch_P[0][i]**2+4*branch_Q[0][i]**2+(branch_square_currents[0][i]-Vm*Vm)**2<=
                    (branch_square_currents[0][i]+Vm*Vm)**2)
#节点3
for i in range(24):
    model2.addConstr(1e6*bus_square_voltages[1][i]==1e6*bus_square_voltages[0][i]-2*1e3*(branch_r[1]*branch_P[1][i]+branch_x[1]*branch_Q[1][i])
                    +(branch_r[1]*branch_r[1]+branch_x[1]*branch_x[1])*branch_square_currents[1][i])
    #流入节点功率+节点自身功率=流出节点功率+负荷
    model2.addConstr(branch_P[1][i]+gen_Ps[1][i]-branch_square_currents[1][i]*branch_r[1]/1e3==bus_Pd[1]*pq_ratio[i]+branch_P[2][i]+branch_P[21][i])
    model2.addConstr(branch_Q[1][i]+gen_Qs[1][i]-branch_square_currents[1][i]*branch_x[1]/1e3==bus_Qd[1]*pq_ratio[i]+branch_Q[2][i]+branch_Q[21][i])
    #线上电流关系
    model2.addQConstr(4*branch_P[1][i]**2+4*branch_Q[1][i]**2+(branch_square_currents[1][i]-bus_square_voltages[0][i])**2<=
                    (branch_square_currents[1][i]+bus_square_voltages[0][i])**2)
#节点4
for i in range(24):
    model2.addConstr(1e6*bus_square_voltages[2][i]==1e6*bus_square_voltages[1][i]-2*1e3*(branch_r[2]*branch_P[2][i]+branch_x[2]*branch_Q[2][i])
                    +(branch_r[2]*branch_r[2]+branch_x[2]*branch_x[2])*branch_square_currents[2][i])
    #流入节点功率+节点自身功率=流出节点功率+负荷
    model2.addConstr(branch_P[2][i]-branch_square_currents[2][i]*branch_r[2]/1e3==bus_Pd[2]*pq_ratio[i]+branch_P[3][i])
    model2.addConstr(branch_Q[2][i]-branch_square_currents[2][i]*branch_x[2]/1e3==bus_Qd[2]*pq_ratio[i]+branch_Q[3][i])
    #线上电流关系
    model2.addQConstr(4*branch_P[2][i]**2+4*branch_Q[2][i]**2+(branch_square_currents[2][i]-bus_square_voltages[1][i])**2<=
                    (branch_square_currents[2][i]+bus_square_voltages[1][i])**2)
#节点5
for i in range(24):
    model2.addConstr(1e6*bus_square_voltages[3][i]==1e6*bus_square_voltages[2][i]-2*1e3*(branch_r[3]*branch_P[3][i]+branch_x[3]*branch_Q[3][i])
                    +(branch_r[3]*branch_r[3]+branch_x[3]*branch_x[3])*branch_square_currents[3][i])
    #流入节点功率+节点自身功率=流出节点功率+负荷
    model2.addConstr(branch_P[3][i]-branch_square_currents[3][i]*branch_r[3]/1e3==bus_Pd[3]*pq_ratio[i]+branch_P[4][i])
    model2.addConstr(branch_Q[3][i]-branch_square_currents[3][i]*branch_x[3]/1e3==bus_Qd[3]*pq_ratio[i]+branch_Q[4][i])
    #线上电流关系
    model2.addQConstr(4*branch_P[3][i]**2+4*branch_Q[3][i]**2+(branch_square_currents[3][i]-bus_square_voltages[2][i])**2<=
                    (branch_square_currents[3][i]+bus_square_voltages[2][i])**2)
#节点6
for i in range(24):
    model2.addConstr(1e6*bus_square_voltages[4][i]==1e6*bus_square_voltages[3][i]-2*1e3*(branch_r[4]*branch_P[4][i]+branch_x[4]*branch_Q[4][i])
                    +(branch_r[4]*branch_r[4]+branch_x[4]*branch_x[4])*branch_square_currents[4][i])
    #流入节点功率+节点自身功率=流出节点功率+负荷
    model2.addConstr(branch_P[4][i]+gen_Ps[2][i]-branch_square_currents[4][i]*branch_r[4]/1e3==bus_Pd[4]*pq_ratio[i]+branch_P[5][i]+branch_P[24][i])
    model2.addConstr(branch_Q[4][i]+gen_Qs[2][i]-branch_square_currents[4][i]*branch_x[4]/1e3==bus_Qd[4]*pq_ratio[i]+branch_Q[5][i]+branch_Q[24][i])
    #线上电流关系
    model2.addQConstr(4*branch_P[4][i]**2+4*branch_Q[4][i]**2+(branch_square_currents[4][i]-bus_square_voltages[3][i])**2<=
                    (branch_square_currents[4][i]+bus_square_voltages[3][i])**2)
#节点7到17
#再次强调：循环变量i表示时间点，循环变量j表示节点和线路的编号
for j in range(7,18):
    for i in range(24):
        model2.addConstr(1e6*bus_square_voltages[j-2][i]==1e6*bus_square_voltages[j-3][i]-2*1e3*(branch_r[j-2]*branch_P[j-2][i]+branch_x[j-2]*branch_Q[j-2][i])
                    +(branch_r[j-2]*branch_r[j-2]+branch_x[j-2]*branch_x[j-2])*branch_square_currents[j-2][i])
        #流入节点功率+节点自身功率=流出节点功率+负荷
        model2.addConstr(branch_P[j-2][i]-branch_square_currents[j-2][i]*branch_r[j-2]/1e3==bus_Pd[j-2]*pq_ratio[i]+branch_P[j-1][i])
        model2.addConstr(branch_Q[j-2][i]-branch_square_currents[j-2][i]*branch_x[j-2]/1e3==bus_Qd[j-2]*pq_ratio[i]+branch_Q[j-1][i])
        #线上电流关系
        model2.addQConstr(4*branch_P[j-2][i]**2+4*branch_Q[j-2][i]**2+(branch_square_currents[j-2][i]-bus_square_voltages[j-3][i])**2<=
                        (branch_square_currents[j-2][i]+bus_square_voltages[j-3][i])**2)
#节点18
for i in range(24):
    model2.addConstr(1e6*bus_square_voltages[16][i]==1e6*bus_square_voltages[15][i]-2*1e3*(branch_r[16]*branch_P[16][i]+branch_x[16]*branch_Q[16][i])
                    +(branch_r[16]*branch_r[16]+branch_x[16]*branch_x[16])*branch_square_currents[16][i])
    #流入节点功率+节点自身功率=流出节点功率+负荷
    model2.addConstr(branch_P[16][i]-branch_square_currents[16][i]*branch_r[16]/1e3==bus_Pd[16]*pq_ratio[i])
    model2.addConstr(branch_Q[16][i]-branch_square_currents[16][i]*branch_x[16]/1e3==bus_Qd[16]*pq_ratio[i])
    #线上电流关系
    model2.addQConstr(4*branch_P[16][i]**2+4*branch_Q[16][i]**2+(branch_square_currents[16][i]-bus_square_voltages[15][i])**2<=
                    (branch_square_currents[16][i]+bus_square_voltages[15][i])**2)
    
#支路1：节点19到节点22
#节点19
for i in range(24):
    model2.addConstr(1e6*bus_square_voltages[17][i]==1e6*bus_square_voltages[0][i]-2*1e3*(branch_r[17]*branch_P[17][i]+branch_x[17]*branch_Q[17][i])
                    +(branch_r[17]*branch_r[17]+branch_x[17]*branch_x[17])*branch_square_currents[17][i])
    model2.addConstr(branch_P[17][i]-branch_square_currents[17][i]*branch_r[17]/1e3==bus_Pd[17]*pq_ratio[i]+branch_P[18][i])
    model2.addConstr(branch_Q[17][i]-branch_square_currents[17][i]*branch_x[17]/1e3==bus_Qd[17]*pq_ratio[i]+branch_Q[18][i])
    model2.addQConstr(4*branch_P[17][i]**2+4*branch_Q[17][i]**2+(branch_square_currents[17][i]-bus_square_voltages[0][i])**2<=
                    (branch_square_currents[17][i]+bus_square_voltages[0][i])**2)
#节点20
for i in range(24):
    model2.addConstr(1e6*bus_square_voltages[18][i]==1e6*bus_square_voltages[17][i]-2*1e3*(branch_r[18]*branch_P[18][i]+branch_x[18]*branch_Q[18][i])
                    +(branch_r[18]*branch_r[18]+branch_x[18]*branch_x[18])*branch_square_currents[18][i])
    model2.addConstr(branch_P[18][i]-branch_square_currents[18][i]*branch_r[18]/1e3==bus_Pd[18]*pq_ratio[i]+branch_P[19][i])
    model2.addConstr(branch_Q[18][i]-branch_square_currents[18][i]*branch_x[18]/1e3==bus_Qd[18]*pq_ratio[i]+branch_Q[19][i])
    model2.addQConstr(4*branch_P[18][i]**2+4*branch_Q[18][i]**2+(branch_square_currents[18][i]-bus_square_voltages[17][i])**2<=
                    (branch_square_currents[18][i]+bus_square_voltages[17][i])**2)
#节点21
for i in range(24):
    model2.addConstr(1e6*bus_square_voltages[19][i]==1e6*bus_square_voltages[18][i]-2*1e3*(branch_r[19]*branch_P[19][i]+branch_x[19]*branch_Q[19][i])
                    +(branch_r[19]*branch_r[19]+branch_x[19]*branch_x[19])*branch_square_currents[19][i])
    model2.addConstr(branch_P[19][i]-branch_square_currents[19][i]*branch_r[19]/1e3==bus_Pd[19]*pq_ratio[i]+branch_P[20][i])
    model2.addConstr(branch_Q[19][i]-branch_square_currents[19][i]*branch_x[19]/1e3==bus_Qd[19]*pq_ratio[i]+branch_Q[20][i])
    model2.addQConstr(4*branch_P[19][i]**2+4*branch_Q[19][i]**2+(branch_square_currents[19][i]-bus_square_voltages[18][i])**2<=
                    (branch_square_currents[19][i]+bus_square_voltages[18][i])**2)
#节点22
for i in range(24):
    model2.addConstr(1e6*bus_square_voltages[20][i]==1e6*bus_square_voltages[19][i]-2*1e3*(branch_r[20]*branch_P[20][i]+branch_x[20]*branch_Q[20][i])
                    +(branch_r[20]*branch_r[20]+branch_x[20]*branch_x[20])*branch_square_currents[20][i])
    model2.addConstr(branch_P[20][i]-branch_square_currents[20][i]*branch_r[20]/1e3==bus_Pd[20]*pq_ratio[i])
    model2.addConstr(branch_Q[20][i]-branch_square_currents[20][i]*branch_x[20]/1e3==bus_Qd[20]*pq_ratio[i])
    model2.addQConstr(4*branch_P[20][i]**2+4*branch_Q[20][i]**2+(branch_square_currents[20][i]-bus_square_voltages[19][i])**2<=
                    (branch_square_currents[20][i]+bus_square_voltages[19][i])**2)

#支路2：节点23到节点25
#节点23
for i in range(24):
    model2.addConstr(1e6*bus_square_voltages[21][i]==1e6*bus_square_voltages[1][i]-2*1e3*(branch_r[21]*branch_P[21][i]+branch_x[21]*branch_Q[21][i])
                    +(branch_r[21]*branch_r[21]+branch_x[21]*branch_x[21])*branch_square_currents[21][i])
    model2.addConstr(branch_P[21][i]-branch_square_currents[21][i]*branch_r[21]/1e3==bus_Pd[21]*pq_ratio[i]+branch_P[22][i])
    model2.addConstr(branch_Q[21][i]-branch_square_currents[21][i]*branch_x[21]/1e3==bus_Qd[21]*pq_ratio[i]+branch_Q[22][i])
    model2.addQConstr(4*branch_P[21][i]**2+4*branch_Q[21][i]**2+(branch_square_currents[21][i]-bus_square_voltages[1][i])**2<=
                    (branch_square_currents[21][i]+bus_square_voltages[1][i])**2)
#节点24
for i in range(24):
    model2.addConstr(1e6*bus_square_voltages[22][i]==1e6*bus_square_voltages[21][i]-2*1e3*(branch_r[22]*branch_P[22][i]+branch_x[22]*branch_Q[22][i])
                    +(branch_r[22]*branch_r[22]+branch_x[22]*branch_x[22])*branch_square_currents[22][i])
    model2.addConstr(branch_P[22][i]-branch_square_currents[22][i]*branch_r[22]/1e3==bus_Pd[22]*pq_ratio[i]+branch_P[23][i])
    model2.addConstr(branch_Q[22][i]-branch_square_currents[22][i]*branch_x[22]/1e3==bus_Qd[22]*pq_ratio[i]+branch_Q[23][i])
    model2.addQConstr(4*branch_P[22][i]**2+4*branch_Q[22][i]**2+(branch_square_currents[22][i]-bus_square_voltages[21][i])**2<=
                    (branch_square_currents[22][i]+bus_square_voltages[21][i])**2)
#节点25
for i in range(24):
    model2.addConstr(1e6*bus_square_voltages[23][i]==1e6*bus_square_voltages[22][i]-2*1e3*(branch_r[23]*branch_P[23][i]+branch_x[23]*branch_Q[23][i])
                    +(branch_r[23]*branch_r[23]+branch_x[23]*branch_x[23])*branch_square_currents[23][i])
    model2.addConstr(branch_P[23][i]-branch_square_currents[23][i]*branch_r[23]/1e3==bus_Pd[23]*pq_ratio[i])
    model2.addConstr(branch_Q[23][i]-branch_square_currents[23][i]*branch_x[23]/1e3==bus_Qd[23]*pq_ratio[i])
    model2.addQConstr(4*branch_P[23][i]**2+4*branch_Q[23][i]**2+(branch_square_currents[23][i]-bus_square_voltages[22][i])**2<=
                    (branch_square_currents[23][i]+bus_square_voltages[22][i])**2)

#支路3：节点26到节点33
#节点26
for i in range(24):
    model2.addConstr(1e6*bus_square_voltages[24][i]==1e6*bus_square_voltages[4][i]-2*1e3*(branch_r[24]*branch_P[24][i]+branch_x[24]*branch_Q[24][i])
                    +(branch_r[24]*branch_r[24]+branch_x[24]*branch_x[24])*branch_square_currents[24][i])
    model2.addConstr(branch_P[24][i]-branch_square_currents[24][i]*branch_r[24]/1e3==bus_Pd[24]*pq_ratio[i]+branch_P[25][i])
    model2.addConstr(branch_Q[24][i]-branch_square_currents[24][i]*branch_x[24]/1e3==bus_Qd[24]*pq_ratio[i]+branch_Q[25][i])
    model2.addQConstr(4*branch_P[24][i]**2+4*branch_Q[24][i]**2+(branch_square_currents[24][i]-bus_square_voltages[4][i])**2<=
                    (branch_square_currents[24][i]+bus_square_voltages[4][i])**2)
#节点27到节点32
for j in range(27,33):
    for i in range(24):
        model2.addConstr(1e6*bus_square_voltages[j-2][i]==1e6*bus_square_voltages[j-3][i]-2*1e3*(branch_r[j-2]*branch_P[j-2][i]+branch_x[j-2]*branch_Q[j-2][i])
                    +(branch_r[j-2]*branch_r[j-2]+branch_x[j-2]*branch_x[j-2])*branch_square_currents[j-2][i])
        #流入节点功率+节点自身功率=流出节点功率+负荷
        model2.addConstr(branch_P[j-2][i]-branch_square_currents[j-2][i]*branch_r[j-2]/1e3==bus_Pd[j-2]*pq_ratio[i]+branch_P[j-1][i])
        model2.addConstr(branch_Q[j-2][i]-branch_square_currents[j-2][i]*branch_x[j-2]/1e3==bus_Qd[j-2]*pq_ratio[i]+branch_Q[j-1][i])
        #线上电流关系
        model2.addQConstr(4*branch_P[j-2][i]**2+4*branch_Q[j-2][2]**2+(branch_square_currents[j-2][i]-bus_square_voltages[j-3][i])**2<=
                        (branch_square_currents[j-2][i]+bus_square_voltages[j-3][i])**2)
#节点33
for i in range(24):
    model2.addConstr(1e6*bus_square_voltages[31][i]==1e6*bus_square_voltages[30][i]-2*1e3*(branch_r[31]*branch_P[31][i]+branch_x[31]*branch_Q[31][i])
                    +(branch_r[31]*branch_r[31]+branch_x[31]*branch_x[31])*branch_square_currents[31][i])
    model2.addConstr(branch_P[31][i]-branch_square_currents[31][i]*branch_r[31]/1e3==bus_Pd[31]*pq_ratio[i])
    model2.addConstr(branch_Q[31][i]-branch_square_currents[31][i]*branch_x[31]/1e3==bus_Qd[31]*pq_ratio[i])
    model2.addQConstr(4*branch_P[31][i]**2+4*branch_Q[31][i]**2+(branch_square_currents[31][i]-bus_square_voltages[30][i])**2<=
                    (branch_square_currents[31][i]+bus_square_voltages[30][i])**2)
    
#先优化网损，别的目标函数（如经济性相关）可以自己随便调整
model2.setObjective(gp.quicksum(gp.quicksum(branch_square_currents[j][i]*branch_r[j] for j in range(32)) for i in range(24)))
model2.optimize()

gen_active_power_1,gen_active_power_2,gen_active_power_3,gen_reactive_power_1,gen_reactive_power_2,gen_reactive_power_3=[],[],[],[],[],[]
for i in range(24):
    gen_active_power_1.append(gen_Ps[0][i].X)
    gen_active_power_2.append(gen_Ps[1][i].X)
    gen_active_power_3.append(gen_Ps[2][i].X)
    gen_reactive_power_1.append(gen_Qs[0][i].X)
    gen_reactive_power_2.append(gen_Qs[1][i].X)
    gen_reactive_power_3.append(gen_Qs[2][i].X)
utility_grid_active_power_exchange,utility_grid_reactive_power_exchange=[],[]
for i in range(24):
    utility_grid_active_power_exchange.append(branch_P[0][i].X)
    utility_grid_reactive_power_exchange.append(branch_Q[0][i].X)

plt.rcParams['font.sans-serif'] = ['SimHei'] 
# 解决保存图像是负号'-'显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

#绘制三个发电机的出力曲线
fig1, ax1 = plt.subplots(figsize=(12,7))
# 绘制左侧Y轴的有功功率数据
ax1.set_xlabel('时间/h')
ax1.set_ylabel('有功功率/KW')
ax2=ax1.twinx()
ax2.set_ylabel('无功功率/KVar') 

ax1.plot(gen_active_power_1,marker='d',color='r',label='发电机1有功功率')
ax2.plot(gen_reactive_power_1,marker='o',color='r',label='发电机1无功功率')
ax1.plot(gen_active_power_2,marker='d',color='b',label='发电机2有功功率')
ax2.plot(gen_reactive_power_2,marker='o',color='b',label='发电机2无功功率')
ax1.plot(gen_active_power_3,marker='d',color='g',label='发电机3有功功率')
ax2.plot(gen_reactive_power_3,marker='o',color='g',label='发电机3无功功率')

lines1,labels1=ax1.get_legend_handles_labels()
lines2,labels2=ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2,labels1+labels2,loc='best')
plt.show()

#绘制主网交互功率曲线
fig1, ax1 = plt.subplots()
# 绘制左侧Y轴的有功功率数据
ax1.set_xlabel('时间/h')
ax1.set_ylabel('有功功率/KW')
ax2=ax1.twinx()
ax2.set_ylabel('无功功率/KVar') 

ax1.plot(gen_active_power_1,marker='d',color='b',label='主网交互有功功率')
ax2.plot(gen_reactive_power_1,marker='o',color='g',label='主网交互无功功率')

lines1,labels1=ax1.get_legend_handles_labels()
lines2,labels2=ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2,labels1+labels2,loc='best')
plt.show()