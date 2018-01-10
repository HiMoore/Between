# _*_ coding: utf-8 -*-

import numpy as np
import pandas as pd
from math import *
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号




class STK_Simulation:
	
	def __init__(self, number=50, step=120):
		self.number = number
		self.step = step
		
	
	def get_data(self, number=50):
		'''读取STK仿真数据，默认50个，单位转换为标准制'''
		SA_origin = pd.read_csv("STK/Sat_A J2000_Origin.csv")[:number]
		SA_dm0_pos = pd.read_csv("STK/Sat_A J2000_+DM0.csv")[:number]
		SA_dm0_neg = pd.read_csv("STK/Sat_A J2000_-DM0.csv")[:number]
		SA_dm1_pos = pd.read_csv("STK/Sat_A J2000_+DM1.csv")[:number]
		SA_dm1_neg = pd.read_csv("STK/Sat_A J2000_-DM1.csv")[:number]
		SA_de_pos = pd.read_csv("STK/Sat_A J2000_+De.csv")[:number]
		SA_de_neg = pd.read_csv("STK/Sat_A J2000_-De.csv")[:number]
		SB_origin = pd.read_csv("STK/Sat_b J2000_Origin.csv")[:number]
		SB_dm0_pos = pd.read_csv("STK/Sat_B J2000_+DM0.csv")[:number]
		SB_dm0_neg = pd.read_csv("STK/Sat_B J2000_-DM0.csv")[:number]
		SB_dm1_pos = pd.read_csv("STK/Sat_B J2000_+DM1.csv")[:number]
		SB_dm1_neg = pd.read_csv("STK/Sat_B J2000_-DM1.csv")[:number]
		SB_de_pos = pd.read_csv("STK/Sat_B J2000_+De.csv")[:number]
		SB_de_neg = pd.read_csv("STK/Sat_B J2000_-De.csv")[:number]
		sat_list = [ SA_origin, SA_dm0_pos, SA_dm0_neg, SA_dm1_pos, SA_dm1_neg, SA_de_pos, SA_de_neg, \
					 SB_origin, SB_dm0_pos, SB_dm0_neg, SB_dm1_pos, SB_dm1_neg, SB_de_pos, SB_de_neg ]
		for ele in sat_list:
			del ele["Time (UTCG)"]
			ele *= 1000
			ele.columns = ['x (m)', 'y (m)', 'z (m)', 'vx (m/sec)', 'vy (m/sec)', 'vz (m/sec)']
		return sat_list
	
	
	def b_minus_a(self, sat_list):
		'''计算不同变动参数下的rBA = rB - rA'''
		minus_list = [sat_list[i+7] - sat_list[i] for i in range(7)]
		return minus_list  # [origin, dm0_posi, dm0_nega, dm1_posi, dm1_nega, de_posi, de_nega]
		
		
	def line_of_nodes(self, sat_info):
		'''分别取r和v的第一行，叉乘计算h; 再叉乘uz计算升降交点连线方向N'''
		r = sat_info[['x (m)', 'y (m)', 'z (m)']].values[0]
		v = sat_info[['vx (m/sec)', 'vy (m/sec)', 'vz (m/sec)']].values[0]
		h = np.cross(r, v)
		uz = np.array([0, 0, 1])
		N = np.cross(uz, h)
		N_unit = N / sqrt(np.dot(N, N))
		return N_unit
		
		
	def modr_angle(self, minus_data, N_unit):
		'''计算卫星位置矢量的模，以及与节点矢量N_unit的夹角'''
		ori_r = minus_data[['x (m)', 'y (m)', 'z (m)']].values  #多维数组
		mod_r = np.array([sqrt(np.dot(arr, arr)) for arr in ori_r])     #多维数组中依次取一维进行点乘求模
		angle = np.arccos(np.dot(ori_r, N_unit) / mod_r) * (pi/180)     #N*3 dot 3*1
		return [mod_r, angle]
		
		
	def delta_modr_angle(self, minus_list, N_unit):
		'''计算rBA的长度变化，以及rBA与节点矢量的夹角变化'''
		modr_angles = [ self.modr_angle(minus_list[i], N_unit) for i in range(7) ]  # [ [modr, angle], [modr, angle] ]
		delta_modr = [ modr_angles[i][0] - modr_angles[0][0] for i in range(1, 7) ] # 与origin的modr, angle作差，求变化值
		delta_angle = [ modr_angles[i][1] - modr_angles[0][1] for i in range(1, 7) ]
		return [delta_modr, delta_angle]
		
		
	def draw_change(self, delta_modr, delta_angle):
		Time = self.number * self.step
		plt.figure(1); 
		plt.subplot(211); plt.xlabel("Time / s", fontsize=18); plt.ylabel("△ρ / m", fontsize=18)
		plt.plot(range(0, Time, self.step), np.abs(delta_modr[0]), "g-", label=r"$△M_{0A} = △M_{0B} = +0.02°$")
		plt.plot(range(0, Time, self.step), np.abs(delta_modr[1]), "r--", label=r"$△M_{0A} = -△M_{0B} = -0.004°$")
		plt.legend(fontsize=18)
		
		plt.subplot(212); plt.xlabel("Time / s", fontsize=18); plt.ylabel("△ψ / °", fontsize=18)
		plt.plot(range(0, Time, self.step), np.abs(delta_angle[0]), "g-", label=r"$△M_{0A} = △M_{0B}=+0.02°$")
		plt.plot(range(0, Time, self.step), np.abs(delta_angle[1]), "r--", label=r"$△M_{0A} = -△M_{0B}=-0.004°$")
		plt.legend(fontsize=18)
		
		plt.figure(2); 
		plt.subplot(211); plt.xlabel("Time / s", fontsize=18); plt.ylabel("△ρ / m", fontsize=18)
		plt.plot(range(0, Time, self.step), np.abs(delta_modr[2]), "g-", label=r"$△M_{1A} = △M_{1B} = +3e7 rad/s$")
		plt.plot(range(0, Time, self.step), np.abs(delta_modr[3]), "r--", label=r"$△M_{1A} = -△M_{1B} = -1e7 rad/s$")
		plt.legend(fontsize=18)
		
		plt.subplot(212); plt.xlabel("Time / s", fontsize=18); plt.ylabel("△ψ / °", fontsize=18)
		plt.plot(range(0, Time, self.step), np.abs(delta_angle[2]), "g-", label=r"$△M_{1A} = △M_{1B} = +3e7 rad/s$")
		plt.plot(range(0, Time, self.step), np.abs(delta_angle[3]), "r--", label=r"$△M_{1A} = -△M_{1B} = -1e7 rad/s$")
		plt.legend(fontsize=18)
		
		plt.figure(3); 
		plt.subplot(211); plt.xlabel("Time / s", fontsize=18); plt.ylabel("△ρ / m", fontsize=18)
		plt.plot(range(0, Time, self.step), delta_modr[4], "g-", label=r"$△e_A = △e_B = +0.0001$")
		plt.plot(range(0, Time, self.step), delta_modr[5], "r--", label=r"$△e_A = -△e_B = 0.00002$")
		plt.legend(fontsize=18)
       
		plt.subplot(212); plt.xlabel("Time / s", fontsize=18); plt.ylabel("△ψ / °", fontsize=18)
		plt.plot(range(0, Time, self.step), delta_angle[4], "g-", label=r"$△e_A = △e_B = +0.0001$")
		plt.plot(range(0, Time, self.step), delta_angle[5], "r--", label=r"$△e_A = -△e_B = 0.00002$")
		plt.legend(fontsize=18)
		plt.show()
		
		
		




if __name__ == "__main__":
	stk = STK_Simulation()
	Sat_list = stk.get_data()
	minus_list = [origin, dm0_posi, dm0_nega, dm1_posi, dm1_nega, de_posi, de_nega] = stk.b_minus_a(Sat_list)
	N_unit = stk.line_of_nodes(Sat_list[0])
	[delta_modr, delta_angle] = stk.delta_modr_angle(minus_list, N_unit)
	stk.draw_change(delta_modr, delta_angle)

	
	