import array as arr
import numpy as np
import pandas as pds
import random
import time
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from pyswarm import pso
from matplotlib.gridspec import GridSpec
from itertools import product
from mpi4py import MPI
from itertools import islice

file =('Trial.xlsx')
newData = pds.read_excel(file)
newData
newData1 = pds.read_excel(file,index_col = 0, header = 0)
newData1
lat_lon = pds.read_excel(file,index_col = 0, header = 0, usecols = 'A:C')
lat_lon
mddl = pds.read_excel(file,index_col = 0, header = 0, usecols = [0,3])
mddl
frl = pds.read_excel(file,index_col = 0, header = 0, usecols = [0,4])
frl
live_storage = pds.read_excel(file,index_col = 0, header = 0, usecols = [0,5])
live_storage
dead_storage = pds.read_excel(file,index_col = 0, header = 0, usecols = [0,6])
dead_storage
river_bed_level = pds.read_excel(file,index_col = 0, header = 0, usecols = [0,7])
river_bed_level
newData2 = newData1.assign(max_height=newData1['F.R.L(m)/Pond Level(PL)'] - newData1['River Bed Level(m)'])
newData2 = newData2.assign(min_height=newData2['MDDL(m)'] - newData2['River Bed Level(m)'])
newData2 = newData2.assign(max_storage=newData2['Live Storage(MCM)'] + newData2['Dead Storage(MCM)'])
newData2 = newData2.assign(min_storage=newData2['Dead Storage(MCM)'])

##dataframe to list
max_storage_list = newData2['max_storage'].tolist()
min_storage_list = newData2['min_storage'].tolist()

###convert to xarray
result = newData2.to_xarray()

class Dam:
    MDDL = 768.544
    FRL = 814.625
    Gross_Storage = 115.53
    Live_Storage = 112.13
    Dead_Storage = 3.4
    River_Bed_Level = 752
    Lenth_of_Dam = 1211
    Height_of_Dam = 65.1
    Catchment_Area = 48450
    n = np.array([[1,1],[1,1]])
    release_per_month_Mm3 = np.array([[200, 400],[200,400]])
    head_causing_flow = ([[20, 40],[20, 40]])
    Demand_per_month_Mm3 = ([[400,800],[400,800]])
    
    
    def Hydropower(self):
        P = 2725*(np.multiply(self.release_per_month_Mm3,self.head_causing_flow,self.n))
        print(P)
        
    def Irrigation(self):
        SQDV = [((x-y)/y)**2 for x,y in zip(self.release_per_month_Mm3,self.Demand_per_month_Mm3)]
        print('SQDV:',SQDV)
        totalSQDV = sum(SQDV)
        print('totalSQDV: ', totalSQDV)

def objective_function(O):
        r = np.ones((11, 12))
        for j in range(11):
            for i in range(12):
                r[j, i] = O[(12 * j) + i]
        
        s = np.ones((11, 13))
        s_initial = max_storage_list
        for i in range(len(s_initial)):
            s[i, 0] = s_initial[i]
            s[i,12] = s[i,0]

        s_lb_bound = np.ones((11,13))
        for j in range(13):
            s_lb_bound[:,j] = min_storage_list
                
        s_up_bound = np.ones((11,13))
        for j in range(13):
            s_up_bound[:,j] = max_storage_list

                
        inflow = np.zeros((11, 12))
        inflow[0] = ([0.5, 1, 2, 3, 3.5, 2.5, 2, 1.25, 1.25, 0.75, 1.75, 1])
        inflow[1] = ([0.4, 0.7, 2, 2, 4, 3.5, 3, 2.5, 1.3, 1.2, 1, 0.7])
        inflow[2] = 0.8*np.ones(12)
        inflow[4] = ([1.5,2,2.5,2.5,3,3.5,3.5,3,2.5,2.5,2,1.5])
        inflow[5] = ([.32,.81,1.53,2.16,2.31,4.32,4.81,2.24,1.63,1.91,.8,.46])
        inflow[7] = ([.71,.83,1,1.25,1.67,2.5,2.8,1.87,1.45,1.2,.93,.81])
                
        evap = np.zeros((11,12))
        
        h = np.zeros((11, 12))
        M = 80
        
        m11 = np.array([[-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, -1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, -1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 1, -1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, -1]])
        
        for j in range(11):
            for i in range(12):
                s[j, i+1] = s[j, i] + inflow[j, i] - evap[j,i] + (m11 @ r[:, i])[j]
                if s[j, i+1] > s_up_bound[j,i+1]:
                    h[j][i] = abs((s[j, i + 1] - s_up_bound[j,i+1])) * M  
                
                elif s[j, i+1] < s_lb_bound[j,i+1]:
                    h[j][i] = abs((s[j, i + 1] - s_lb_bound[j,i+1])) * M
                
        d = np.ones((11, 1))
        d[:,0] = [6, 6, 3, 8,8,7,15,6,5,15,15]
        g = np.ones(11)
        for i in range(11):
            if (s[i][12]-d[i][0]) <= 0:
                g[i] = ((s[i][12]-d[i][0])**2)*60
            if (s[i][12]-d[i][0]) > 0:
                g[i] = 0
                
        b_hp = np.array([[1.1, 1, 1, 1.2, 1.8, 2.5, 2.2, 2, 1.8, 2.2, 1.8, 1.4],[1.4, 1.1, 1, 1, 1.2, 1.8, 2.5, 2.2, 2, 1.8, 2.2, 1.8],[1, 1, 1.2, 1.8, 2.5, 2.2, 2, 1.5, 2.2, 1.8, 1.4, 1.1],[1.1,1,1,1.2,1.8,2.5,2.2,2,1.8,2.2,1.8,1.4],[1,1.1,1.2,1.3,1.4,1.5,1.67,1.56,1.45,1.34,1.25,1.14],[1.4,1.1,1,1,1.2,1.8,2.5,2.2,2,1.8,2.2,1.8],[2.6,2.9,3.6,4.4,4.2,4,3.8,4.1,3.6,3.1,2.7,2.5],[1,1.1,1.2,1.3,1.4,1.5,1.67,1.56,1.45,1.34,1.25,1.14],[1,1,1.2,1.8,2.5,2.2,2,1.8,2.2,1.8,1.4,1.1],[2.7,3,2.8,3.2,2.9,3.9,4,3.6,3.7,2.8,3.5,2.1],[2.7,3,2.8,3.2,2.9,3.9,4,3.6,3.7,2.8,3.5,2.1]])
                    
        hp = np.multiply(b_hp, r)
                
        retsum = (np.sum(hp)) - np.sum(g) - np.sum(h)
        return -retsum  

lb = list(0.005*np.ones(48)) + list(0.006*np.ones(24)) + list(0.01*np.ones(12)) + list(0.008*np.ones(24)) + list(0.01*np.ones(12)) + list(0.01*np.ones(12))
ub = list(4*np.ones(12)) + list(4.5*np.ones(12)) + list(2.12*np.ones(12)) + list(7*np.ones(12)) +list(6.43*np.ones(12)) + list(4.21*np.ones(12)) +list(17.1*np.ones(12)) + list(3.1*np.ones(12)) + list(4.2*np.ones(12)) + list(18.9*np.ones(12)) + list(18.9*np.ones(12))
    
xopt, fopt = pso(objective_function, lb, ub, swarmsize=100, omega=.9, phip=1.2, phig=1.2, maxiter=1000)
    
print(fopt)
    
      
 














