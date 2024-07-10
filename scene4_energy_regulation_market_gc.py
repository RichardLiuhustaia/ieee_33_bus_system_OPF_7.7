import numpy as np
import gurobipy as gp
from gurobipy import GRB
from parameters import *
import matplotlib.pyplot as plt
from chance_constrain import *
from IW_data_preprocess import gmm_quantile,norm_quantile

TD=4
model1=gp.Model('solution_with_reg')
#variable declaration
A_BW_t=model1.addVars(24,lb=0,ub=A_max,vtype=GRB.INTEGER,name='A_BW_t')
A_IW_t=model1.addVars(24,lb=0,ub=A_max,vtype=GRB.INTEGER,name='A_IW_t')
A_R_t=model1.addVars(24,lb=0,ub=A_max,vtype=GRB.INTEGER,name='A_R_t')
L_BW_t=model1.addVars(24,lb=0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name='L_BW_t')
L_IW_t=model1.addVars(24,lb=0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name='L_IW_t')
P_res_t=model1.addVars(24,lb=0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name='P_res_t')
P_d_t=model1.addVars(24,lb=-GRB.INFINITY,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name='P_d_t')
W_res=model1.addVar(lb=0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name='W_res')
delta_W_res=model1.addVar(lb=-GRB.INFINITY,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name='delta_W_res')
#variables about reserve market and regulation support
P_reg_t=model1.addVars(24,lb=0,ub=100,vtype=GRB.CONTINUOUS,name='P_reg_t')
#P_d_t=model1.addVars(24,lb=0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name='P_d_t')
#constraints
#power balance constraint:DC's power supply comes from utility grid and PV generation
model1.addConstrs(1/1000*(P_idle+0.75*P_peak)*(A_BW_t[i]+A_IW_t[i])+1/1000*P_idle*A_R_t[i]+1/1000*(L_BW_t[i]+L_IW_t[i])*(P_peak-P_idle)/L_rate+beta[i]*P_reg_t[i]
                 ==P_d_t[i]+P_res_t[i] for i in range(T))
model1.addConstrs(P_reg_t[i]<=1/1000*(P_idle+0.75*P_peak)*(A_max-A_BW_t[i]-A_IW_t[i]-A_R_t[i]) for i in range(T))
model1.addConstrs(1/1000*(P_idle+0.75*P_peak)*(A_BW_t[i]+A_IW_t[i]+A_R_t[i])+1/1000*(L_BW_t[i]+L_IW_t[i])*(P_peak-P_idle)/L_rate-P_reg_t[i]>=
                  1/1000*P_idle*redundant_ratio*A_max for i in range(T))

#QoS constraint of interactive loads
model1.addConstrs(A_IW_t[i]>=L_IW_t[i]/(L_rate-1/C_DT) for i in range(T))
#interactive loads must be immediately satisfied
model1.addConstrs(L_IW_t[i]>=predicted_IWs[i]+gmm_quantile for i in range(T))
#enough processing resource for batch loads
model1.addConstrs(L_rate*A_BW_t[i]>=L_BW_t[i] for i in range(T))
#batch loads constraints
for j in range(T):
    model1.addConstr(gp.quicksum(L_BW_t[i] for i in range(0,j))<=gp.quicksum(predicted_BWs[i] for i in range(0,j)))
#assume TD=4,further modification allowed
for j in range(T-TD):
    model1.addConstr(gp.quicksum(L_BW_t[i] for i in range(0,j+TD))>=gp.quicksum(predicted_BWs[i] for i in range(0,j)))
for j in range(T-TD,T):
    model1.addConstr(gp.quicksum(L_BW_t[i] for i in range(0,T))>=gp.quicksum(predicted_BWs[i] for i in range(0,j)))

model1.addConstrs(A_R_t[i]>=redundant_ratio*A_max for i in range(T))
model1.addConstrs(A_IW_t[i]+A_BW_t[i]+A_R_t[i]<=A_max for i in range(T))
#green certificate related constraints
model1.addConstrs(P_res_t[i]<=P_res_max[i] for i in range(T))
model1.addConstrs(P_d_t[i]<=P_grid_max for i in range(T))
model1.addConstrs(P_d_t[i]>=-P_grid_max for i in range(T))
model1.addConstr(gp.quicksum(P_res_t[i] for i in range(T))==W_res)
model1.addConstr(W_res<=W_res_max)
model1.addConstr(delta_W_res==W_res_fix-W_res)
#auxiliary variable z
z=model1.addVar(vtype=GRB.BINARY,name='z')
M=1e5
model1.addConstr(delta_W_res<=delta_W_res_fix+M*z)
model1.addConstr(delta_W_res>=delta_W_res_fix-M*(1-z))

model1.setObjective(gp.quicksum(L_BW_t[i]*C_BW+L_IW_t[i]*C_IW for i in range(T))
                   +gp.quicksum(reg_cap_price[i]*s_reg*P_reg_t[i]+reg_mil_price[i]*s_reg*R_mil*P_reg_t[i] for i in range(T))
                   -gp.quicksum(P_d_t[i]*electricity_price[i] for i in range(T))
                   -C_RES*gp.quicksum(P_res_t[i] for i in range(T))
                   -C_GC*delta_W_res-C_punish*(delta_W_res-delta_W_res_fix)*z,GRB.MAXIMIZE)
model1.optimize()

P_reg_t_res=np.array([P_reg_t[i].X for i in range(T)])
P_reg_t_rspd=np.array([P_reg_t[i].X*beta[i] for i in range(T)])
A_BW_t_res=np.array([A_BW_t[i].X for i in range(T)])
A_IW_t_res=np.array([A_IW_t[i].X for i in range(T)])
A_R_t_res=np.array([A_R_t[i].X for i in range(T)])
L_BW_t_res=np.array([L_BW_t[i].X for i in range(T)])
L_IW_t_res=np.array([L_IW_t[i].X for i in range(T)])
P_res_t_res=np.array([P_res_t[i].X for i in range(T)])
P_d_t_res=np.array([P_d_t[i].X for i in range(T)])
'''
plt.plot(L_IW_t_res,marker='o',color='r')
plt.plot(L_BW_t_res,marker='o',color='g')
plt.show()
plt.plot(P_d_t_res,marker='o',color='r')
plt.plot(electricity_price,marker='o',color='g')
plt.show()
plt.plot(P_res_t_res,marker='o',color='b')
plt.plot(P_reg_t_res,marker='o',color='g')
plt.show()
print(W_res.X/sum(P_res_max))
print(L_BW_t_res*C_BW+L_IW_t_res*C_IW)
#仿真做一下绿证对新能源消纳的影响分析
'''
plt.plot(A_BW_t_res,marker='o',color='g',label="active servers for batch loads")
plt.plot(A_IW_t_res,marker='d',color='r',label="active servers for interactive loads")
plt.plot(A_R_t_res,marker='x',color='b',label="reserve servers")
plt.title("scene4_server_allocation")
plt.xlabel("time/h")
plt.legend()
plt.savefig('scene4_servers.pdf',dpi=800)
plt.show()

bottom=np.array([max(0,P_d_t_res[i]) for i in range(T)])
plt.bar(x=np.arange(24),height=P_d_t_res,color='b',label="energy bought/sold")
plt.bar(x=np.arange(24),height=P_reg_t_res,color='orange',bottom=bottom,label="regulation capacity")
plt.axhline(y=0,color='black',linewidth=0.4)
plt.title("scene4_energy_and_regulation")
plt.xlabel("time/h")
plt.ylabel("power/MW")
plt.legend()
plt.savefig('scene4_energy_market.pdf',dpi=800)
plt.show()

plt.plot(P_reg_t_res,marker='d',color='r',label="regulation_capacity")
plt.bar(x=np.arange(24),height=P_reg_t_rspd,color='b',label="regulation_responded")
plt.axhline(y=0,color='black',linewidth=0.4)
plt.title("scene4_regulation_capacity_responded")
plt.xlabel("time/h")
plt.ylabel("power/MW")
plt.legend()
plt.savefig('scene4_regulation.pdf',dpi=800)
plt.show()

print(sum(P_res_t_res)/sum(P_res_max))
print(sum(L_BW_t_res[i]*C_BW+L_IW_t_res[i]*C_IW for i in range(T)))
print(sum(P_d_t_res[i]*electricity_price[i] for i in range(T)))
print(sum(reg_cap_price[i]*s_reg*P_reg_t_res[i]+reg_mil_price[i]*s_reg*R_mil*P_reg_t_res[i] for i in range(T)))
print(C_RES*sum(P_res_t_res[i] for i in range(T))
                   +C_GC*delta_W_res.X+C_punish*(delta_W_res.X-delta_W_res_fix)*z.X)