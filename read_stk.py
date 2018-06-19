# _*_ coding: utf-8 -*-

import numpy as np
import pandas as pd
from math import *
from matplotlib import pyplot as plt
from orbit_km import Orbit
import time 
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['NSimSun', 'Times New Roman'] # 指定默认字体
mpl.rcParams['font.family']='sans-serif'
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
mpl.rcParams["figure.figsize"] = (3.2, 3); mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['legend.fontsize'] = 6; mpl.rcParams['axes.labelsize'] = 8;
mpl.rcParams['xtick.labelsize'] = 8; mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['savefig.dpi'] = 400; mpl.rcParams['figure.dpi'] = 400



class STK_Simulation:
	
	def __init__(self, number=50, step=120):
		self.number = number
		self.step = step
		
	
	def get_data(self, number=50):
		'''读取STK仿真数据，默认50个，单位转换为标准制'''
		SA_origin = pd.read_csv("STK/Part_1/Sat_A J2000_Origin.csv", nrows=number, usecols=range(1,7)).values * 1000
		SA_dm0_pos = pd.read_csv("STK/Part_1/Sat_A J2000_+DM0.csv", nrows=number, usecols=range(1,7)).values * 1000
		SA_dm0_neg = pd.read_csv("STK/Part_1/Sat_A J2000_-DM0.csv", nrows=number, usecols=range(1,7)).values * 1000
		SA_dm1_pos = pd.read_csv("STK/Part_1/Sat_A J2000_+DM1.csv", nrows=number, usecols=range(1,7)).values * 1000
		SA_dm1_neg = pd.read_csv("STK/Part_1/Sat_A J2000_-DM1.csv", nrows=number, usecols=range(1,7)).values * 1000
		SA_de_pos = pd.read_csv("STK/Part_1/Sat_A J2000_+De.csv", nrows=number, usecols=range(1,7)).values * 1000
		SA_de_neg = pd.read_csv("STK/Part_1/Sat_A J2000_-De.csv", nrows=number, usecols=range(1,7)).values * 1000
		SA_dw_pos = pd.read_csv("STK/Part_1/Sat_A J2000_+Dw.csv", nrows=number, usecols=range(1,7)).values * 1000
		SA_dw_neg = pd.read_csv("STK/Part_1/Sat_A J2000_-Dw.csv", nrows=number, usecols=range(1,7)).values * 1000
		
		SB_origin = pd.read_csv("STK/Part_1/Sat_B J2000_Origin.csv", nrows=number, usecols=range(1,7)).values * 1000
		SB_dm0_pos = pd.read_csv("STK/Part_1/Sat_B J2000_+DM0.csv", nrows=number, usecols=range(1,7)).values * 1000
		SB_dm0_neg = pd.read_csv("STK/Part_1/Sat_B J2000_-DM0.csv", nrows=number, usecols=range(1,7)).values * 1000
		SB_dm1_pos = pd.read_csv("STK/Part_1/Sat_B J2000_+DM1.csv", nrows=number, usecols=range(1,7)).values * 1000
		SB_dm1_neg = pd.read_csv("STK/Part_1/Sat_B J2000_-DM1.csv", nrows=number, usecols=range(1,7)).values * 1000
		SB_de_pos = pd.read_csv("STK/Part_1/Sat_B J2000_+De.csv", nrows=number, usecols=range(1,7)).values * 1000
		SB_de_neg = pd.read_csv("STK/Part_1/Sat_B J2000_-De.csv", nrows=number, usecols=range(1,7)).values * 1000
		SB_dw_pos = pd.read_csv("STK/Part_1/Sat_B J2000_+Dw.csv", nrows=number, usecols=range(1,7)).values * 1000
		SB_dw_neg = pd.read_csv("STK/Part_1/Sat_B J2000_-Dw.csv", nrows=number, usecols=range(1,7)).values * 1000
		sat_list = np.array([ SA_origin, SA_dm0_pos, SA_dm0_neg, SA_dm1_pos, SA_dm1_neg, SA_de_pos, SA_de_neg, SA_dw_pos, SA_dw_neg,\
							  SB_origin, SB_dm0_pos, SB_dm0_neg, SB_dm1_pos, SB_dm1_neg, SB_de_pos, SB_de_neg, SB_dw_pos, SB_dw_neg ])
		return sat_list
	
	
	def b_minus_a(self, sat_list):
		'''计算不同变动参数下的rBA = rB - rA'''
		minus_list = [sat_list[i+9] - sat_list[i] for i in range(9)]
		return minus_list  # [origin, dm0_posi, dm0_nega, dm1_posi, dm1_nega, de_posi, de_nega, dw_posi, dw_nega]
		
		
	def line_of_nodes(self, sat_info):
		'''分别取r和v的第一行，叉乘计算h; 再叉乘uz计算升降交点连线方向N'''
		r = sat_info[0, 0:3]
		v = sat_info[0, 3:6]
		h = np.cross(r, v)
		uz = np.array([0, 0, 1])
		N = np.cross(uz, h)
		N_unit = N / sqrt(np.dot(N, N))
		return N_unit
		
		
	def modr_angle(self, minus_data, N_unit):
		'''计算卫星位置矢量的模，以及与节点矢量N_unit的夹角'''
		ori_r = minus_data[:, 0:3]  #多维数组
		mod_r = np.array([ np.linalg.norm(arr, 2) for arr in ori_r])     #多维数组中依次取一维进行点乘求模
		angle = np.arccos(np.dot(ori_r, N_unit) / mod_r) * (180/pi)    #N*3 dot 3*1
		return [mod_r, angle]
		
		
	def delta_modr_angle(self, minus_list, N_unit):
		'''计算rBA的长度变化，以及rBA与节点矢量的夹角变化'''
		modr_angles = [ self.modr_angle(minus_list[i], N_unit) for i in range(9) ]  # [ [modr, angle], [modr, angle] ]
		delta_modr = [ modr_angles[i][0] - modr_angles[0][0] for i in range(1, 9) ] # 与origin的modr, angle作差，求变化值
		delta_angle = [ modr_angles[i][1] - modr_angles[0][1] for i in range(1, 9) ]
		return [delta_modr, delta_angle]
		
		
	def draw_change(self, delta_modr, delta_angle):
		Time = self.number * self.step
		plt.figure(1); 
		plt.subplot(211); plt.ylabel("△ρ / m")
		plt.plot(range(0, Time, self.step), np.abs(delta_modr[1]), "g-", label=r"$△M_{0A} = △M_{0B} = 0.02°$")
		plt.plot(range(0, Time, self.step), np.abs(delta_modr[0]), "r--", label=r"$△M_{0A} = -△M_{0B} = 0.004°$")
		plt.legend()
		
		plt.subplot(212); plt.xlabel("Time / s"); plt.ylabel("△ψ / °")
		plt.plot(range(0, Time, self.step), np.abs(delta_angle[0]), "g-", label=r"$△M_{0A} = △M_{0B}=0.02°$")
		plt.plot(range(0, Time, self.step), np.abs(delta_angle[1]), "r--", label=r"$△M_{0A} = -△M_{0B}=0.004°$")
		plt.tight_layout(); plt.savefig("Figure/M0.png", dpi=400, bbox_inches='tight')
		
		plt.figure(2); 
		plt.subplot(211); plt.ylabel("△ρ / m")
		plt.plot(range(0, Time, self.step), np.abs(delta_modr[2]), "g-", label=r"$△M_{1A} = △M_{1B} = 3e^{7} rad/s$")
		plt.plot(range(0, Time, self.step), np.abs(delta_modr[3]), "r--", label=r"$△M_{1A} = -△M_{1B} = 1e^{7} rad/s$")
		plt.legend()
		
		plt.subplot(212); plt.xlabel("Time / s"); plt.ylabel("△ψ / °")
		plt.plot(range(0, Time, self.step), np.abs(delta_angle[2]), "g-", label=r"$△M_{1A} = △M_{1B} = 3e^{7} rad/s$")
		plt.plot(range(0, Time, self.step), np.abs(delta_angle[3]), "r--", label=r"$△M_{1A} = -△M_{1B} = 1e^{7} rad/s$")
		plt.tight_layout(); plt.savefig("Figure/M1.png", dpi=400, bbox_inches='tight')
		
		plt.figure(3); 
		plt.subplot(211); plt.ylabel("△ρ / m")
		plt.plot(range(0, Time, self.step), delta_modr[4], "g-", label=r"$△e_A = △e_B = 0.01$")
		plt.plot(range(0, Time, self.step), delta_modr[5], "r--", label=r"$△e_A = -△e_B = 0.005$")
		plt.legend()
       
		plt.subplot(212); plt.xlabel("Time / s"); plt.ylabel("△ψ / °")
		plt.plot(range(0, Time, self.step), delta_angle[4], "g-", label=r"$△e_A = △e_B = 0.01$")
		plt.plot(range(0, Time, self.step), delta_angle[5], "r--", label=r"$△e_A = -△e_B = 0.005$")
		plt.tight_layout(); plt.savefig("Figure/e.png", dpi=400, bbox_inches='tight')
		
		plt.figure(4); 
		plt.subplot(211); plt.ylabel("△ρ / m")
		plt.plot(range(0, Time, self.step), np.abs(delta_modr[6]), "g-", label=r"$△w_{0A} = △w_{0B} = 0.1°$")
		plt.plot(range(0, Time, self.step), np.abs(delta_modr[7]), "r--", label=r"$△w_{0A} = -△w_{0B} = 0.1°$")
		plt.legend()
		
		plt.subplot(212); plt.xlabel("Time / s"); plt.ylabel("△ψ / °")
		plt.plot(range(0, Time, self.step), np.abs(delta_angle[6]), "g-", label=r"$△M_{0A} = △M_{0B}=0.1°$")
		plt.plot(range(0, Time, self.step), np.abs(delta_angle[7]), "r--", label=r"$△M_{0A} = -△M_{0B}=0.1°$")
		plt.tight_layout(); plt.savefig("Figure/w.png", dpi=400, bbox_inches='tight')
		
		plt.show()
		



if __name__ == "__main__":
	stk = STK_Simulation()
	number = 50
	# sat_list = stk.get_data()
	# minus_list = [origin, dm0_posi, dm0_nega, dm1_posi, dm1_nega, de_posi, de_nega, dw_posi, dw_nega] = stk.b_minus_a(sat_list)
	# N_unit = stk.line_of_nodes(sat_list[0])
	# [delta_modr, delta_angle] = stk.delta_modr_angle(minus_list, N_unit)
	# stk.draw_change(delta_modr, delta_angle)
	SA_origin = pd.read_csv("STK/Part_1/Sat_A J2000_Origin.csv", nrows=number, usecols=range(1,7)).values * 1000
	SA_dm0_pos = pd.read_csv("STK/Part_1/Sat_A J2000_+DM0.csv", nrows=number, usecols=range(1,7)).values * 1000
	SA_dm0_neg = pd.read_csv("STK/Part_1/Sat_A J2000_-DM0.csv", nrows=number, usecols=range(1,7)).values * 1000
	SA_dm1_pos = pd.read_csv("STK/Part_1/Sat_A J2000_+DM1.csv", nrows=number, usecols=range(1,7)).values * 1000
	SA_dm1_neg = pd.read_csv("STK/Part_1/Sat_A J2000_-DM1.csv", nrows=number, usecols=range(1,7)).values * 1000
	SA_de_pos = pd.read_csv("STK/Part_1/Sat_A J2000_+De.csv", nrows=number, usecols=range(1,7)).values * 1000
	SA_de_neg = pd.read_csv("STK/Part_1/Sat_A J2000_-De.csv", nrows=number, usecols=range(1,7)).values * 1000
	SA_dw_pos = pd.read_csv("STK/Part_1/Sat_A J2000_+Dw.csv", nrows=number, usecols=range(1,7)).values * 1000
	SA_dw_neg = pd.read_csv("STK/Part_1/Sat_A J2000_-Dw.csv", nrows=number, usecols=range(1,7)).values * 1000
	
	SB_origin = pd.read_csv("STK/Part_1/Sat_B J2000_Origin.csv", nrows=number, usecols=range(1,7)).values * 1000
	SB_dm0_pos = pd.read_csv("STK/Part_1/Sat_B J2000_+DM0.csv", nrows=number, usecols=range(1,7)).values * 1000
	SB_dm0_neg = pd.read_csv("STK/Part_1/Sat_B J2000_-DM0.csv", nrows=number, usecols=range(1,7)).values * 1000
	SB_dm1_pos = pd.read_csv("STK/Part_1/Sat_B J2000_+DM1.csv", nrows=number, usecols=range(1,7)).values * 1000
	SB_dm1_neg = pd.read_csv("STK/Part_1/Sat_B J2000_-DM1.csv", nrows=number, usecols=range(1,7)).values * 1000
	SB_de_pos = pd.read_csv("STK/Part_1/Sat_B J2000_+De.csv", nrows=number, usecols=range(1,7)).values * 1000
	SB_de_neg = pd.read_csv("STK/Part_1/Sat_B J2000_-De.csv", nrows=number, usecols=range(1,7)).values * 1000
	SB_dw_pos = pd.read_csv("STK/Part_1/Sat_B J2000_+Dw.csv", nrows=number, usecols=range(1,7)).values * 1000
	SB_dw_neg = pd.read_csv("STK/Part_1/Sat_B J2000_-Dw.csv", nrows=number, usecols=range(1,7)).values * 1000
	sat_list = np.array([ SA_origin, SA_dm0_pos, SA_dm0_pos, SA_dm1_pos, SA_dm1_pos, SA_de_pos, SA_de_pos, SA_dw_pos, SA_dw_pos,\
						  SB_origin, SB_dm0_pos, SB_dm0_neg, SB_dm1_pos, SB_dm1_neg, SB_de_pos, SB_de_neg, SB_dw_pos, SB_dw_neg ])
	minus_list = [ sat_list[i+9] - sat_list[i] for i in range(9) ]
	N_unit = stk.line_of_nodes(sat_list[0])
	[delta_modr, delta_angle] = stk.delta_modr_angle(minus_list, N_unit)
	stk.draw_change(delta_modr, delta_angle)
	