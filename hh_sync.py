import math 
import numpy as np
def alpha_n(V):
    return 0.01*(10.0-V)/ (np.exp(1.0-0.1*V )-1)
def beta_n(V):
    return 0.125*np.exp(-V/80.0)
def alpha_m(V):
    return  0.1*(25-V) / (np.exp(2.5 - 0.1*V)-1)
def beta_m(V): 
    return 4.0*np.exp(-V/18.0)
def alpha_h(V): 
    return 0.07*np.exp(-V/20.0)
def beta_h(V): 
    return 1/(1+np.exp(3.0-0.1*V))

Cm = 1; # uF/cm^2
E_Na = 120  # mV
E_K = -12  #mV
E_Leak = 10.6; # mV
g_Na = 120; # mS/cm^2
g_K = 36; # mS/cm^2
g_Leak = 0.3; # mS/cm^2


# V = 0;s = 0 
m=0;n=0;h=0
V = np.zeros(100) #np.ones(100)*-65
s = np.zeros(100)
dt = 0.01

I_DBS = np.random.normal(9,10,100)
ew = np.random.normal(0.1,0.02,(100,100))

def HH(V,m,n,h,s,I_DBS):
    I_Na = g_Na*m**3*h*(V-E_Na);
    I_K = g_K*n**4*(V-E_K);
    I_Leak = g_Leak*(V-E_Leak);
    # Isyn_exci = (-75-V)*np.squeeze((np.dot(ew, np.array([s]).T)))/200
    # Isyn_inhi = (20-V)*np.squeeze((np.dot(ew, np.array([s]).T)))/200
    Isyn = ((20-V)*np.squeeze((np.dot(ew, np.array([s]).T))))/100
    # Input  = (I_DBS-(I_Na+I_K+I_Leak))
    Input  = (I_DBS-(I_Na+I_K+I_Leak)+Isyn)# _exci+Isyn_inhi)
    V  = V + dt *Input* (1/Cm) # -spk*(v+60)
    m = m + dt*((alpha_m(V)*(1-m))-beta_m(V)*m)
    n = n + dt*((alpha_n(V)*(1-n))-beta_n(V)*n)
    h = h + dt*((alpha_h(V)*(1-h))-beta_h(V)*h)
    s =  s + dt* (-s + ((1-s)*5)/(1+np.exp(-(V+3)/8)))
    return V,m,n,h,s,Isyn #_exci,Isyn_inhi

import multiprocessing as mp 
quant_proc = np.max((mp.cpu_count()-1,1)) # 47 output 
pool = mp.Pool(processes=quant_proc) 

####################
l = 1 # I want to change this from 1 to 10000
s_pre = np.zeros((l*5000,100))
t_pre = np.zeros((l*5000,100))
V1 = np.zeros((l*5000,100))
s1 = np.zeros((l*5000,100))
Isyn_all = np.zeros((l*5000,100))
# without multiprocessing working code 
# import time
# st = time.time()
# for t in range(l*5000):
#     V,m,n,h,s,Isyn=  HH(V,m,n,h,s,I_DBS)
#     V1[t,:] = V
#     Isyn_all[t] = Isyn
#     s1[t] = s
#     if t>2:
#         for i in range(100):
#             if V1[t-2,i]<V1[t-1,i] and V1[t-1,i]> V1[t,i]:
#                 t_pre[t,i] = t 
#                 s_pre[t,i] =1 
# et = time.time()
# print(et-st)

############something wrong in implementation ???????
import time
st = time.time()
def OneTrial(void):
    for t in range(l*5000):
        V,m,n,h,s,Isyn=  HH(V,m,n,h,s,I_DBS)
        V1[t,:] = V
        Isyn_all[t] = Isyn
        s1[t] = s
        if t>2:
            for i in range(100):
                if V1[t-2,i]<V1[t-1,i] and V1[t-1,i]> V1[t,i]:
                    t_pre[t,i] = t 
                    s_pre[t,i] =1 
    return s_pre
All_spikes = pool.map(OneTrial,np.zeros(l))
et = time.time()
print(et-st)

# lets make a rastor plot 
import torch
import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
fig,ax = plt.subplots()
# it is synchronise but not the perfect sync
pre = torch.from_numpy(np.array(s_pre))
#post = torch.from_numpy(np.array(spike2[:,:50]))
splt.raster(pre[-4000:], ax, s=25, c="blue",marker='|')
#splt.raster(post, ax, s=25, c="red",marker='|')
# plt.savefig('./batista/P0.1withoutSTDP_pertur.png')
plt.show()
