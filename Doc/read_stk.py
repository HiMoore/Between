# _*_ coding: utf-8 -*-

import numpy as np
import pandas as pd
from math import *
from matplotlib import pyplot as plt
import orbit
from orbit import Orbit
import time 
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
		SA_dw_pos = pd.read_csv("STK/Sat_A J2000_+Dw.csv")[:number]
		SA_dw_neg = pd.read_csv("STK/Sat_A J2000_-Dw.csv")[:number]
		
		SB_origin = pd.read_csv("STK/Sat_B J2000_Origin.csv")[:number]
		SB_dm0_pos = pd.read_csv("STK/Sat_B J2000_+DM0.csv")[:number]
		SB_dm0_neg = pd.read_csv("STK/Sat_B J2000_-DM0.csv")[:number]
		SB_dm1_pos = pd.read_csv("STK/Sat_B J2000_+DM1.csv")[:number]
		SB_dm1_neg = pd.read_csv("STK/Sat_B J2000_-DM1.csv")[:number]
		SB_de_pos = pd.read_csv("STK/Sat_B J2000_+De.csv")[:number]
		SB_de_neg = pd.read_csv("STK/Sat_B J2000_-De.csv")[:number]
		SB_dw_pos = pd.read_csv("STK/Sat_B J2000_+Dw.csv")[:number]
		SB_dw_neg = pd.read_csv("STK/Sat_B J2000_-Dw.csv")[:number]
		sat_list = [ SA_origin, SA_dm0_pos, SA_dm0_neg, SA_dm1_pos, SA_dm1_neg, SA_de_pos, SA_de_neg, SA_dw_pos, SA_dw_neg, \
					 SB_origin, SB_dm0_pos, SB_dm0_neg, SB_dm1_pos, SB_dm1_neg, SB_de_pos, SB_de_neg, SB_dw_pos, SB_dw_neg]
		for ele in sat_list:
			del ele["Time (UTCG)"]
			ele *= 1000
			ele.columns = ['x (m)', 'y (m)', 'z (m)', 'vx (m/sec)', 'vy (m/sec)', 'vz (m/sec)']
		return sat_list
	
	
	def b_minus_a(self, sat_list):
		'''计算不同变动参数下的rBA = rB - rA'''
		minus_list = [sat_list[i+7] - sat_list[i] for i in range(9)]
		return minus_list  # [origin, dm0_posi, dm0_nega, dm1_posi, dm1_nega, de_posi, de_nega, dw_posi, dw_nega]
		
		
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
		modr_angles = [ self.modr_angle(minus_list[i], N_unit) for i in range(9) ]  # [ [modr, angle], [modr, angle] ]
		delta_modr = [ modr_angles[i][0] - modr_angles[0][0] for i in range(1, 9) ] # 与origin的modr, angle作差，求变化值
		delta_angle = [ modr_angles[i][1] - modr_angles[0][1] for i in range(1, 9) ]
		return [delta_modr, delta_angle]
		
		
	def draw_change(self, delta_modr, delta_angle):
		Time = self.number * self.step
		plt.figure(1); 
		plt.subplot(211); plt.xlabel("Time / s", fontsize=18); plt.ylabel("△ρ / m", fontsize=18)
		plt.plot(range(0, Time, self.step), np.abs(delta_modr[1]), "g-", label=r"$△M_{0A} = △M_{0B} = +0.02°$")
		plt.plot(range(0, Time, self.step), np.abs(delta_modr[0]), "r--", label=r"$△M_{0A} = -△M_{0B} = -0.004°$")
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
		
		plt.figure(4); 
		plt.subplot(211); plt.xlabel("Time / s", fontsize=18); plt.ylabel("△ρ / m", fontsize=18)
		plt.plot(range(0, Time, self.step), np.abs(delta_modr[6]), "g-", label=r"$△w_{0A} = △w_{0B} = +0.02°$")
		plt.plot(range(0, Time, self.step), np.abs(delta_modr[7]), "r--", label=r"$△w_{0A} = -△w_{0B} = -0.004°$")
		plt.legend(fontsize=18)
		
		plt.subplot(212); plt.xlabel("Time / s", fontsize=18); plt.ylabel("△ψ / °", fontsize=18)
		plt.plot(range(0, Time, self.step), np.abs(delta_angle[6]), "g-", label=r"$△M_{0A} = △M_{0B}=+0.02°$")
		plt.plot(range(0, Time, self.step), np.abs(delta_angle[7]), "r--", label=r"$△M_{0A} = -△M_{0B}=-0.004°$")
		plt.legend(fontsize=18)
		
		
		plt.show()
		

	def draw_accelerate(self, number=200, step=120):
		data = pd.read_csv("STK/Moon.csv")[:number]
		del data["Time (UTCG)"]
		data *= 1000
		data.columns = ['x (m)', 'y (m)', 'z (m)', 'vx (m/sec)', 'vy (m/sec)', 'vz (m/sec)']
		r_array = data[['x (m)', 'y (m)', 'z (m)']].values
		orb = Orbit()
		a0 = np.array([ np.linalg.norm(orb.centreGravity(r_sat=r_sat, miu=orbit.MIU_M, Re=1.738e+06), 2) for r_sat in r_array ])
		utc_list = orb.generate_time(start_t="20171231", end_t="20180101")
		a1 = np.array([ np.linalg.norm(orb.nonspherGravity(r_sat, time_utc)) for (r_sat, time_utc) in zip(r_array, utc_list) ])
		a_solar = np.array([ np.linalg.norm(orb.solarPress(beta_last=0.75, time_utc=time_utc, r_sat=r_sat)) \
				for (r_sat, time_utc) in zip(r_array, utc_list) ])
		a_sun = np.array([ np.linalg.norm(orb.thirdSun(time_utc=time_utc, r_sat=r_sat, miu=orbit.MIU_S)) \
				for (r_sat, time_utc) in zip(r_array, utc_list) ])
		a_earth = np.array([ np.linalg.norm(orb.thirdEarth(time_utc=time_utc, r_sat=r_sat, miu=orbit.MIU_E)) \
				for (r_sat, time_utc) in zip(r_array, utc_list) ])
		time_range = range(0, number*step, step)
		plt.figure(1)
		plt.xlabel("Time / s", fontsize=18); plt.ylabel(r"$a / (m/s^2)$", fontsize=18)
		plt.plot(time_range, a0, "r-", label="centre gravity")
		a = a0+a1+a_solar+a_sun+a_earth
		plt.plot(time_range, a, "g-", label="Combined acceleration")
		plt.legend(fontsize=18); 
		
		plt.figure(2)
		plt.xlabel("Time / s", fontsize=18); plt.ylabel(r"$a / (m/s^2)$", fontsize=18)
		plt.plot(time_range, a1, "g--", label="nonspher gravity")
		plt.plot(time_range, a_sun, "r-", label="third-sun gravity")
		plt.plot(time_range, a_earth, "y-", label="third-earth gravity")
		plt.plot(time_range, a_solar, "b-", label="solar pressure gravity")
		plt.legend(fontsize=18); plt.show()
		
		
	def numberic(self, r1, r2):
		miu = 3.986004415e+14
		r1_norm = np.linalg.norm(r1, 2)
		r2_norm = np.linalg.norm(r2, 2)
		B1 = miu/r1_norm**5 * (3*np.outer(r1, r1) - np.dot(r1, r1)*np.identity(3))
		B2 = miu/r2_norm**5 * (3*np.outer(r2, r2) - np.dot(r2, r2)*np.identity(3))
		
		



if __name__ == "__main__":
	stk = STK_Simulation()
	Sat_list = stk.get_data()
	minus_list = [origin, dm0_posi, dm0_nega, dm1_posi, dm1_nega, de_posi, de_nega, dw_posi, dw_nega] = stk.b_minus_a(Sat_list)
	N_unit = stk.line_of_nodes(Sat_list[0])
	[delta_modr, delta_angle] = stk.delta_modr_angle(minus_list, N_unit)
	stk.draw_change(delta_modr, delta_angle)
	#stk.draw_accelerate()
	# cop_1, cop_2 = np.array([6832.842724, 801.612273, 435.239952]), np.array([6726.550816, 1327.179878, 720.599879])	#共面但不相碰
	# cross_1, cross_2 = np.array([6832.842724, 801.612273, 435.239952]), np.array([6832.842724, 698.746779, 586.318165])		#不共面但同时经过交线
	# non_1, non_2 = np.array([6832.842724, 801.612273, 435.239952]), np.array([6726.550816, 1156.871840, 970.730735])	#不共面也不同时经过交线
	
	