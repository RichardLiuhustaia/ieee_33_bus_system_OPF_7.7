import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import seaborn as sns
import matplotlib.pyplot as plt
from data import *

model=gp.Model()
    
bus_square_voltages=[model.addVar(lb=Vmin*Vmin, ub=Vmax*Vmax, vtype=GRB.CONTINUOUS, name=f'bus_square_voltage_{i+1}') for i in range(32)]
gen_Ps=[model.addVar(lb=0,ub=2000,vtype=GRB.CONTINUOUS,name=f'generator_p_{i+1}') for i in range(3)]
gen_Qs=[model.addVar(lb=0,ub=2000,vtype=GRB.CONTINUOUS,name=f'generator_q_{i+1}') for i in range(3)]
branch_P=[model.addVar(lb=-GRB.INFINITY,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name=f'branch_P_{i+1}') for i in range(32)]
branch_Q=[model.addVar(lb=-GRB.INFINITY,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name=f'branch_Q_{i+1}') for i in range(32)]
branch_square_currents=[model.addVar(lb=0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name=f'branch_square_currents_{i+1}') for i in range(32)]

####
#branch_P/Q[0]-branch_P/Q[16]:1->2至17->18
#branch_P/Q[17]-branch[20]:2->19至21->22
#branch[21]-branch[23]:3->23至24->25
#branch[24]-branch[31]:6->26至32->33

#主线
#节点2
#后一个节点电压=前一个节点电压和功率、电流的关系
model.addConstr(1e6*bus_square_voltages[0]==1e6*Vm*Vm-2*1e3*(branch_r[0]*branch_P[0]+branch_x[0]*branch_Q[0])
                +(branch_r[0]*branch_r[0]+branch_x[0]*branch_x[0])*branch_square_currents[0])
#流入节点功率+节点自身功率=流出节点功率+负荷
model.addConstr(branch_P[0]+gen_Ps[0]==bus_Pd[0]+branch_P[1]+branch_P[17])
model.addConstr(branch_Q[0]+gen_Qs[0]==bus_Qd[0]+branch_Q[1]+branch_Q[17])
#线上电流关系
model.addQConstr(4*branch_P[0]**2+4*branch_Q[0]**2+(branch_square_currents[0]-Vm*Vm)**2<=
                 (branch_square_currents[0]+Vm*Vm)**2)
#节点3
model.addConstr(1e6*bus_square_voltages[1]==1e6*bus_square_voltages[0]-2*1e3*(branch_r[1]*branch_P[1]+branch_x[1]*branch_Q[1])
                +(branch_r[1]*branch_r[1]+branch_x[1]*branch_x[1])*branch_square_currents[1])
#流入节点功率+节点自身功率=流出节点功率+负荷
model.addConstr(branch_P[1]+gen_Ps[1]==bus_Pd[1]+branch_P[2]+branch_P[21])
model.addConstr(branch_Q[1]+gen_Qs[1]==bus_Qd[1]+branch_Q[2]+branch_Q[21])
#线上电流关系
model.addQConstr(4*branch_P[1]**2+4*branch_Q[1]**2+(branch_square_currents[1]-bus_square_voltages[0])**2<=
                 (branch_square_currents[1]+bus_square_voltages[0])**2)
#节点4
model.addConstr(1e6*bus_square_voltages[2]==1e6*bus_square_voltages[1]-2*1e3*(branch_r[2]*branch_P[2]+branch_x[2]*branch_Q[2])
                +(branch_r[2]*branch_r[2]+branch_x[2]*branch_x[2])*branch_square_currents[2])
#流入节点功率+节点自身功率=流出节点功率+负荷
model.addConstr(branch_P[2]==bus_Pd[2]+branch_P[3])
model.addConstr(branch_Q[2]==bus_Qd[2]+branch_Q[3])
#线上电流关系
model.addQConstr(4*branch_P[2]**2+4*branch_Q[2]**2+(branch_square_currents[2]-bus_square_voltages[1])**2<=
                 (branch_square_currents[2]+bus_square_voltages[1])**2)
#节点5
model.addConstr(1e6*bus_square_voltages[3]==1e6*bus_square_voltages[2]-2*1e3*(branch_r[3]*branch_P[3]+branch_x[3]*branch_Q[3])
                +(branch_r[3]*branch_r[3]+branch_x[3]*branch_x[3])*branch_square_currents[3])
#流入节点功率+节点自身功率=流出节点功率+负荷
model.addConstr(branch_P[3]==bus_Pd[3]+branch_P[4])
model.addConstr(branch_Q[3]==bus_Qd[3]+branch_Q[4])
#线上电流关系
model.addQConstr(4*branch_P[3]**2+4*branch_Q[3]**2+(branch_square_currents[3]-bus_square_voltages[2])**2<=
                 (branch_square_currents[3]+bus_square_voltages[2])**2)
#节点6
model.addConstr(1e6*bus_square_voltages[4]==1e6*bus_square_voltages[3]-2*1e3*(branch_r[4]*branch_P[4]+branch_x[4]*branch_Q[4])
                +(branch_r[4]*branch_r[4]+branch_x[4]*branch_x[4])*branch_square_currents[4])
#流入节点功率+节点自身功率=流出节点功率+负荷
model.addConstr(branch_P[4]+gen_Ps[2]==bus_Pd[4]+branch_P[5]+branch_P[24])
model.addConstr(branch_Q[4]+gen_Qs[2]==bus_Qd[4]+branch_Q[5]+branch_Q[24])
#线上电流关系
model.addQConstr(4*branch_P[4]**2+4*branch_Q[4]**2+(branch_square_currents[4]-bus_square_voltages[3])**2<=
                 (branch_square_currents[4]+bus_square_voltages[3])**2)
#节点7到17
for i in range(7,18):
    model.addConstr(1e6*bus_square_voltages[i-2]==1e6*bus_square_voltages[i-3]-2*1e3*(branch_r[i-2]*branch_P[i-2]+branch_x[i-2]*branch_Q[i-2])
                +(branch_r[i-2]*branch_r[i-2]+branch_x[i-2]*branch_x[i-2])*branch_square_currents[i-2])
    #流入节点功率+节点自身功率=流出节点功率+负荷
    model.addConstr(branch_P[i-2]==bus_Pd[i-2]+branch_P[i-1])
    model.addConstr(branch_Q[i-2]==bus_Qd[i-2]+branch_Q[i-1])
    #线上电流关系
    model.addQConstr(4*branch_P[i-2]**2+4*branch_Q[i-2]**2+(branch_square_currents[i-2]-bus_square_voltages[i-3])**2<=
                     (branch_square_currents[i-2]+bus_square_voltages[i-3])**2)
#节点18
model.addConstr(1e6*bus_square_voltages[16]==1e6*bus_square_voltages[15]-2*1e3*(branch_r[16]*branch_P[16]+branch_x[16]*branch_Q[16])
                +(branch_r[16]*branch_r[16]+branch_x[16]*branch_x[16])*branch_square_currents[16])
#流入节点功率+节点自身功率=流出节点功率+负荷
model.addConstr(branch_P[16]==bus_Pd[16])
model.addConstr(branch_Q[16]==bus_Qd[16])
#线上电流关系
model.addQConstr(4*branch_P[16]**2+4*branch_Q[16]**2+(branch_square_currents[16]-bus_square_voltages[15])**2<=
                 (branch_square_currents[16]+bus_square_voltages[15])**2)

#支路1：节点19到节点22
#节点19
model.addConstr(1e6*bus_square_voltages[17]==1e6*bus_square_voltages[0]-2*1e3*(branch_r[17]*branch_P[17]+branch_x[17]*branch_Q[17])
                +(branch_r[17]*branch_r[17]+branch_x[17]*branch_x[17])*branch_square_currents[17])
model.addConstr(branch_P[17]==bus_Pd[17]+branch_P[18])
model.addConstr(branch_Q[17]==bus_Qd[17]+branch_Q[18])
model.addQConstr(4*branch_P[17]**2+4*branch_Q[17]**2+(branch_square_currents[17]-bus_square_voltages[0])**2<=
                 (branch_square_currents[17]+bus_square_voltages[0])**2)
#节点20
model.addConstr(1e6*bus_square_voltages[18]==1e6*bus_square_voltages[17]-2*1e3*(branch_r[18]*branch_P[18]+branch_x[18]*branch_Q[18])
                +(branch_r[18]*branch_r[18]+branch_x[18]*branch_x[18])*branch_square_currents[18])
model.addConstr(branch_P[18]==bus_Pd[18]+branch_P[19])
model.addConstr(branch_Q[18]==bus_Qd[18]+branch_Q[19])
model.addQConstr(4*branch_P[18]**2+4*branch_Q[18]**2+(branch_square_currents[18]-bus_square_voltages[17])**2<=
                 (branch_square_currents[18]+bus_square_voltages[17])**2)
#节点21
model.addConstr(1e6*bus_square_voltages[19]==1e6*bus_square_voltages[18]-2*1e3*(branch_r[19]*branch_P[19]+branch_x[19]*branch_Q[19])
                +(branch_r[19]*branch_r[19]+branch_x[19]*branch_x[19])*branch_square_currents[19])
model.addConstr(branch_P[19]==bus_Pd[19]+branch_P[20])
model.addConstr(branch_Q[19]==bus_Qd[19]+branch_Q[20])
model.addQConstr(4*branch_P[19]**2+4*branch_Q[19]**2+(branch_square_currents[19]-bus_square_voltages[18])**2<=
                 (branch_square_currents[19]+bus_square_voltages[18])**2)
#节点22
model.addConstr(1e6*bus_square_voltages[20]==1e6*bus_square_voltages[19]-2*1e3*(branch_r[20]*branch_P[20]+branch_x[20]*branch_Q[20])
                +(branch_r[20]*branch_r[20]+branch_x[20]*branch_x[20])*branch_square_currents[20])
model.addConstr(branch_P[20]==bus_Pd[20])
model.addConstr(branch_Q[20]==bus_Qd[20])
model.addQConstr(4*branch_P[20]**2+4*branch_Q[20]**2+(branch_square_currents[20]-bus_square_voltages[19])**2<=
                 (branch_square_currents[20]+bus_square_voltages[19])**2)

#支路2：节点23到节点25
#节点23
model.addConstr(1e6*bus_square_voltages[21]==1e6*bus_square_voltages[1]-2*1e3*(branch_r[21]*branch_P[21]+branch_x[21]*branch_Q[21])
                +(branch_r[21]*branch_r[21]+branch_x[21]*branch_x[21])*branch_square_currents[21])
model.addConstr(branch_P[21]==bus_Pd[21]+branch_P[22])
model.addConstr(branch_Q[21]==bus_Qd[21]+branch_Q[22])
model.addQConstr(4*branch_P[21]**2+4*branch_Q[21]**2+(branch_square_currents[21]-bus_square_voltages[1])**2<=
                 (branch_square_currents[21]+bus_square_voltages[1])**2)
#节点24
model.addConstr(1e6*bus_square_voltages[22]==1e6*bus_square_voltages[21]-2*1e3*(branch_r[22]*branch_P[22]+branch_x[22]*branch_Q[22])
                +(branch_r[22]*branch_r[22]+branch_x[22]*branch_x[22])*branch_square_currents[22])
model.addConstr(branch_P[22]==bus_Pd[22]+branch_P[23])
model.addConstr(branch_Q[22]==bus_Qd[22]+branch_Q[23])
model.addQConstr(4*branch_P[22]**2+4*branch_Q[22]**2+(branch_square_currents[22]-bus_square_voltages[21])**2<=
                 (branch_square_currents[22]+bus_square_voltages[21])**2)
#节点25
model.addConstr(1e6*bus_square_voltages[23]==1e6*bus_square_voltages[22]-2*1e3*(branch_r[23]*branch_P[23]+branch_x[23]*branch_Q[23])
                +(branch_r[23]*branch_r[23]+branch_x[23]*branch_x[23])*branch_square_currents[23])
model.addConstr(branch_P[23]==bus_Pd[23])
model.addConstr(branch_Q[23]==bus_Qd[23])
model.addQConstr(4*branch_P[23]**2+4*branch_Q[23]**2+(branch_square_currents[23]-bus_square_voltages[22])**2<=
                 (branch_square_currents[23]+bus_square_voltages[22])**2)

#支路3：节点26到节点33
#节点26
model.addConstr(1e6*bus_square_voltages[24]==1e6*bus_square_voltages[4]-2*1e3*(branch_r[24]*branch_P[24]+branch_x[24]*branch_Q[24])
                +(branch_r[24]*branch_r[24]+branch_x[24]*branch_x[24])*branch_square_currents[24])
model.addConstr(branch_P[24]==bus_Pd[24]+branch_P[25])
model.addConstr(branch_Q[24]==bus_Qd[24]+branch_Q[25])
model.addQConstr(4*branch_P[24]**2+4*branch_Q[24]**2+(branch_square_currents[24]-bus_square_voltages[4])**2<=
                 (branch_square_currents[24]+bus_square_voltages[4])**2)
#节点27到节点32
for i in range(27,33):
    model.addConstr(1e6*bus_square_voltages[i-2]==1e6*bus_square_voltages[i-3]-2*1e3*(branch_r[i-2]*branch_P[i-2]+branch_x[i-2]*branch_Q[i-2])
                +(branch_r[i-2]*branch_r[i-2]+branch_x[i-2]*branch_x[i-2])*branch_square_currents[i-2])
    #流入节点功率+节点自身功率=流出节点功率+负荷
    model.addConstr(branch_P[i-2]==bus_Pd[i-2]+branch_P[i-1])
    model.addConstr(branch_Q[i-2]==bus_Qd[i-2]+branch_Q[i-1])
    #线上电流关系
    model.addQConstr(4*branch_P[i-2]**2+4*branch_Q[i-2]**2+(branch_square_currents[i-2]-bus_square_voltages[i-3])**2<=
                     (branch_square_currents[i-2]+bus_square_voltages[i-3])**2)
#节点33
model.addConstr(1e6*bus_square_voltages[31]==1e6*bus_square_voltages[30]-2*1e3*(branch_r[31]*branch_P[31]+branch_x[31]*branch_Q[31])
                +(branch_r[31]*branch_r[31]+branch_x[31]*branch_x[31])*branch_square_currents[31])
model.addConstr(branch_P[31]==bus_Pd[31])
model.addConstr(branch_Q[31]==bus_Qd[31])
model.addQConstr(4*branch_P[31]**2+4*branch_Q[31]**2+(branch_square_currents[31]-bus_square_voltages[30])**2<=
                 (branch_square_currents[31]+bus_square_voltages[30])**2)

#先做一个最简单的优化网损的目标函数。有了上面的模型之后，可以方便的使用各种目标函数，并往节点上挂各种设备或者可再生能源
model.setObjective(gp.quicksum(branch_square_currents[i]*branch_r[i] for i in range(32)))
model.optimize()

print(branch_P[0].X,branch_Q[0].X)
print(branch_P[1].X,branch_Q[1].X)
print(branch_P[4].X,branch_Q[4].X)
print(gen_Ps[0].X,gen_Qs[1].X)
print(gen_Ps[1].X,gen_Qs[1].X)
print(gen_Ps[2].X,gen_Qs[2].X)