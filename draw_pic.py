# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import *
from orbit_km import *
from navigation import *

import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['NSimSun', 'Times New Roman'] # 指定默认字体
mpl.rcParams['font.family']='sans-serif'
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
mpl.rcParams["figure.figsize"] = (15, 9); mpl.rcParams['lines.linewidth'] = 4
mpl.rcParams['legend.fontsize'] = 30; mpl.rcParams['axes.labelsize'] = 30;
mpl.rcParams['xtick.labelsize'] = 30;mpl.rcParams['ytick.labelsize'] = 30


NUMBER = 1440; orb = Orbit(); nav = Navigation()	# 实例化类
data = ( pd.read_csv("STK/Part_2/1_Inertial_HPOP_660.csv", nrows=NUMBER, usecols=range(1,7)) ).values
# 不同轨道高度的卫星数据
Low = ( pd.read_csv("STK/different_height/Sat_1_LowOrbit_120s.csv", nrows=NUMBER, usecols=range(1,7)) ).values
Mid = ( pd.read_csv("STK/different_height/Sat_1_MidOrbit_120s.csv", nrows=NUMBER, usecols=range(1,7)) ).values
High = ( pd.read_csv("STK/different_height/Sat_1_HighOrbit_120s.csv", nrows=NUMBER, usecols=range(1,7)) ).values
# 基础卫星数据，轨道倾角分别为0和28.5°
basic_i0A = ( pd.read_csv("STK/basic/Sat_1_i0.csv", nrows=NUMBER, usecols=range(1,7)) ).values
basic_i0B = ( pd.read_csv("STK/basic/Sat_1_i0.csv", nrows=NUMBER, usecols=range(1,7)) ).values
basic_i28A = ( pd.read_csv("STK/basic/Sat_1_i28.csv", nrows=NUMBER, usecols=range(1,7)) ).values
basic_i28B = ( pd.read_csv("STK/basic/Sat_1_i28.csv", nrows=NUMBER, usecols=range(1,7)) ).values

##################################################################################################

utc_array = (orb.generate_time(start_t="20180101", end_t="20180131"))[:NUMBER]
utcJD_list = [ time_utc.to_julian_date() for time_utc in utc_array ]
tdbJD_list = [ time_utc.to_julian_date() + 69.184/86400 for time_utc in utc_array ]


class Second_Chapter:

	def __init__(self):
		return
	
	
	def j2_nonspher_acc(self, number=360, step=120):
		'''绘制J2和30阶次非球形摄动加速度三维矢量图'''
		a_j2 = np.array([ orb.nonspher_J2(r_sat) for r_sat in data[:number] ]) * 1000
		a_30 = np.array([ orb.nonspher_moon(r_sat, tdb_jd, lm=30) for (r_sat, tdb_jd) in zip(data[:number], tdbJD_list[:number]) ]) * 1000
		time_range = np.arange(0, number*step/9000, step/9000)
		plt.figure(1)
		plt.xlabel("轨道数量 / 个"); plt.ylabel(r"加速度 / $(\mathrm{m/s^2})$")
		plt.plot(time_range, a_j2[:number, 0], "g--", label="J2, X方向")
		plt.plot(time_range, a_30[:number, 0], "r-", label="30阶次, X方向")
		plt.legend()
		
		plt.figure(2)
		plt.xlabel("轨道数量 / 个"); plt.ylabel(r"加速度 / $(\mathrm{m/s^2})$")
		plt.plot(time_range, a_j2[:number, 1], "g--", label="J2, Y方向")
		plt.plot(time_range, a_30[:number, 1], "r-", label="30阶次, Y方向")
		plt.legend(); 
		
		plt.figure(3)
		plt.xlabel("轨道数量 / 个"); plt.ylabel(r"加速度 / $(\mathrm{m/s^2})$")
		plt.plot(time_range, a_j2[:number, 2], "g--", label="J2, Z方向")
		plt.plot(time_range, a_30[:number, 2], "r-", label="30阶次, Z方向")
		plt.legend()
		
		plt.show()
	
		
	def diff_order_acc(self, number=200, step=120):
		'''绘制10阶, 30阶, 80阶的摄动加速度量级对比'''
		a_10 = np.array([ np.linalg.norm(orb.nonspher_moon(r_sat, tdb_jd, lm=10)) for (r_sat, tdb_jd) in zip(data[:number], tdbJD_list[:number]) ]) * 1000
		a_30 = np.array([ np.linalg.norm(orb.nonspher_moon(r_sat, tdb_jd, lm=30)) for (r_sat, tdb_jd) in zip(data[:number], tdbJD_list[:number]) ]) * 1000
		a_20 = np.array([ np.linalg.norm(orb.nonspher_moon(r_sat, tdb_jd, lm=20)) for (r_sat, tdb_jd) in zip(data[:number], tdbJD_list[:number]) ]) * 1000
		a_80 = np.array([ np.linalg.norm(orb.nonspher_moon(r_sat, tdb_jd, lm=80)) for (r_sat, tdb_jd) in zip(data[:number], tdbJD_list[:number]) ]) * 1000
		
		time_range = np.arange(0, number*step/9000, step/9000)
		plt.figure(1)
		plt.xlabel("轨道数量 / 个"); plt.ylabel(r"加速度 / $(\mathrm{m/s^2})$")
		plt.plot(time_range, a_10, "r-", label="10阶非球形摄动加速度")
		plt.plot(time_range, a_20, "g--", label="20阶非球形摄动加速度")
		plt.plot(time_range, a_30, "b-.", label="30阶非球形摄动加速度")
		plt.plot(time_range, a_80, "m:", label="80阶非球形摄动加速度")
		plt.legend(); 
		plt.show()

		
	def diff_height_orbit(self, number=number, step=120):
		'''绘制低轨, 中轨, 高轨的月球卫星三维轨道图'''
		number = number // 5
		fig = plt.figure(1)
		ax = Axes3D(fig)
		ax.plot(Low[:number, 0], Low[:number, 1], Low[:number, 2], "c1-", label='低轨卫星')
		ax.plot(Mid[:number, 0], Mid[:number, 1], Mid[:number, 2], "g--", label='中轨卫星')
		number = int(number * 3.2)
		ax.plot(High[:number, 0], High[:number, 1], High[:number, 2], "b-.", label='高轨卫星')
		ax.scatter(0, 0, 0, marker="x", s=50, c='r', label='月心')
		ax.set_xlabel("x轴 / ($\mathrm{km}$)", fontsize=22)
		ax.set_ylabel("y轴 / ($\mathrm{km}$)", fontsize=22)
		ax.set_zlabel("z轴 / ($\mathrm{km}$)", fontsize=22)
		# ax.set_xticklabels(fontdict={"fontsize": 18} )
		# ax.set_yticklabels(fontdict={"fontsize": 18} )
		# ax.set_zticklabels(fontdict={"fontsize": 20} )
		ax.legend()
		plt.show()
		
		
	def diff_height_acc(self, number=360, step=120):
		'''绘制低轨, 中轨, 高轨的月球卫星受到的各种摄动加速度量级'''
		num_l, num_m, num_h = number, number*2, number*4
		time_low = np.arange(0, num_l*step/9898, step/9898)
		time_mid = np.arange(0, num_m*step/34623, step/34623)
		time_high = np.arange(0, num_h*step/93802, step/93802)
		time_range = np.arange(0, number*step, step)
		# # 低轨卫星
		Low_non = np.array([ np.linalg.norm(orb.nonspher_moon(r_sat, tdb_jd)) for (r_sat, tdb_jd) in zip(Low[:num_l], tdbJD_list[:num_l]) ]) * 1000
		Low_earth = np.array([ np.linalg.norm(orb.thirdEarth(r_sat, tdb_jd)) for (r_sat, tdb_jd) in zip(Low[:num_l], tdbJD_list[:num_l]) ]) * 1000
		Low_sun = np.array([ np.linalg.norm(orb.thirdSun(r_sat, tdb_jd)) for (r_sat, tdb_jd) in zip(Low[:num_l], tdbJD_list[:num_l]) ]) * 1000
		Low_solar = np.array([ np.linalg.norm(orb.solarPress(r_sat, tdb_jd)) for (r_sat, tdb_jd) in zip(Low[:num_l], tdbJD_list[:num_l]) ]) * 1000
		plt.figure(1)
		plt.xlabel("轨道数量 / 个"); plt.ylabel(r"加速度 / $(\mathrm{m/s^2})$")
		plt.plot(time_low, Low_non, "r-", label="非球形摄动加速度")
		plt.plot(time_low, Low_earth, "g--", label="地球引力摄动加速度")
		plt.plot(time_low, Low_sun, "b-.", label="太阳引力摄动加速度")
		plt.plot(time_low, Low_solar, "m-", label="太阳光压摄动加速度")
		plt.legend(); 
		# 中轨卫星
		Mid_non = np.array([ np.linalg.norm(orb.nonspher_moon(r_sat, tdb_jd)) for (r_sat, tdb_jd) in zip(Mid[:num_m], tdbJD_list[:num_m]) ]) * 1000
		Mid_earth = np.array([ np.linalg.norm(orb.thirdEarth(r_sat, tdb_jd)) for (r_sat, tdb_jd) in zip(Mid[:num_m], tdbJD_list[:num_m]) ]) * 1000
		Mid_sun = np.array([ np.linalg.norm(orb.thirdSun(r_sat, tdb_jd)) for (r_sat, tdb_jd) in zip(Mid[:num_m], tdbJD_list[:num_m]) ]) * 1000
		Mid_solar = np.array([ np.linalg.norm(orb.solarPress(r_sat, tdb_jd)) for (r_sat, tdb_jd) in zip(Mid[:num_m], tdbJD_list[:num_m]) ]) * 1000
		plt.figure(2)
		plt.xlabel("轨道数量 / 个"); plt.ylabel(r"加速度 / $(\mathrm{m/s^2})$")
		plt.plot(time_mid, Mid_non, "r-", label="非球形摄动加速度")
		plt.plot(time_mid, Mid_earth, "g--", label="地球引力摄动加速度")
		plt.plot(time_mid, Mid_sun, "b-.", label="太阳引力摄动加速度")
		plt.plot(time_mid, Mid_solar, "m:", label="太阳光压摄动加速度")
		plt.legend(); 
		# 高轨卫星
		High_non = np.array([ np.linalg.norm(orb.nonspher_moon(r_sat, tdb_jd)) for (r_sat, tdb_jd) in zip(High[:num_h], tdbJD_list[:num_h]) ]) * 1000
		High_earth = np.array([ np.linalg.norm(orb.thirdEarth(r_sat, tdb_jd)) for (r_sat, tdb_jd) in zip(High[:num_h], tdbJD_list[:num_h]) ]) * 1000
		High_sun = np.array([ np.linalg.norm(orb.thirdSun(r_sat, tdb_jd)) for (r_sat, tdb_jd) in zip(High[:num_h], tdbJD_list[:num_h]) ]) * 1000
		High_solar = np.array([ np.linalg.norm(orb.solarPress(r_sat, tdb_jd)) for (r_sat, tdb_jd) in zip(High[:num_h], tdbJD_list[:num_h]) ]) * 1000
		plt.figure(3)
		plt.xlabel("轨道数量 / 个"); plt.ylabel(r"加速度 / $(\mathrm{m/s^2})$")
		plt.plot(time_high, High_non, "r-", label="非球形摄动加速度")
		plt.plot(time_high, High_earth, "g--", label="地球引力摄动加速度")
		plt.plot(time_high, High_sun, "b-.", label="太阳引力摄动加速度")
		plt.plot(time_high, High_solar, "m:", label="太阳光压摄动加速度")
		plt.legend(); 

		# 不同轨道高度, 非球形摄动和地球引力摄动量级变化图
		# 非球形引力摄动加速度在不同高度的变化曲线
		# plt.figure(4)
		# plt.xlabel("时间 / 2 min"); plt.ylabel(r"加速度 / $(\mathrm{m/s^2})$")
		# plt.plot(Low_non, "r-", label="a=2300 km")
		# plt.plot(Mid_non, "g--", label="a=5300 km")
		# plt.plot(High_non, "b-.", label="a=10300 km")
		# plt.legend(); 
		
		# plt.figure(5)
		# plt.xlabel("时间 / 2 min"); plt.ylabel(r"加速度 / $(\mathrm{m/s^2})$")
		# plt.plot(Low_earth, "r-", label="a=2300 km")
		# plt.plot(Mid_earth, "g--", label="a=5300 km")
		# plt.plot(High_earth, "b-.", label="a=10300 km")
		# plt.legend(); 
		
		plt.show()
		
		
	def add_third(self, number=360, step=120):
		'''绘制加入第三体引力摄动加速度的摄动量级曲线'''
		a0 =  np.array([ np.linalg.norm(orb.centreGravity(r_sat), 2) for r_sat in r_array ]) * 1000
		Low_non = np.array([ orb.nonspher_moon(r_sat, tdb_jd) for (r_sat, tdb_jd) in zip(Low[:number], tdbJD_list[:number]) ]) * 1000
		Low_earth = np.array([ orb.thirdEarth(r_sat, tdb_jd) for (r_sat, tdb_jd) in zip(Low[:number], tdbJD_list[:number]) ]) * 1000
		Low_sun = np.array([ orb.thirdSun(r_sat, tdb_jd) for (r_sat, tdb_jd) in zip(Low[:number], tdbJD_list[:number]) ]) * 1000
		Low_solar = np.array([ orb.solarPress(r_sat, tdb_jd) for (r_sat, tdb_jd) in zip(Low[:number], tdbJD_list[:number]) ]) * 1000
		non_earth = Low_non + Low_earth
		All = non_earth + Low_sun + Low_solar
		norm_non = np.array([ np.linalg.norm(a, 2) for a in Low_non ])
		norm_addEarth = np.array([ np.linalg.norm(b, 2) for b in non_earth ])
		norm_addAll = np.array([ np.linalg.norm(c, 2) for c in All ])
		time_low = np.arange(0, number*step/9898, step/9898)
		plt.figure(1)
		plt.xlabel("轨道数量 / 个"); plt.ylabel(r"加速度 / $(\mathrm{m/s^2})$")
		plt.plot(time_low, norm_non, "r-", label="非球形摄动加速度")
		# plt.plot(time_low, norm_addEarth, "g--", label="非球形+地球三体")
		plt.plot(time_low, norm_addAll, "g--", label="所有摄动之和")
		plt.legend()
		plt.show()
		
		
	def j2_orbit(self, number=1440, step=120):
		'''绘制J2摄动加速度对不同轨道高度的轨道精度影响曲线'''
		time_range = np.arange(0, number*120/3600, 120/3600)
		low_hpop = Low[:number].T; mid_hpop = Mid[:number].T; high_hpop = High[:number].T
		low_0, mid_0, high_0 = low_hpop[:, 0], mid_hpop[:, 0], high_hpop[:, 0]
		low_two = orb.integrate_tworbody(low_0, number)
		mid_two = orb.integrate_tworbody(mid_0, number)
		high_two = orb.integrate_tworbody(high_0, number)
		low_j2 = orb.integrate_J2(low_0, number)
		mid_j2 = orb.integrate_J2(mid_0, number)
		high_j2 = orb.integrate_J2(high_0, number)
		
		delta_low = (low_hpop - low_two).T
		delta_low_non = (low_hpop - low_j2).T
		delta_mid = (mid_hpop - mid_two).T
		delta_mid_non = (mid_hpop - mid_j2).T
		delta_high = (high_hpop - high_two).T
		delta_high_non = (high_hpop - high_j2).T
		
		delta_low = np.array([ np.linalg.norm(x, 2) for x in delta_low ])
		delta_low_non = np.array([ np.linalg.norm(x, 2) for x in delta_low_non ])
		delta_mid = np.array([ np.linalg.norm(x, 2) for x in delta_mid ])
		delta_mid_non = np.array([ np.linalg.norm(x, 2) for x in delta_mid_non ])
		delta_high = np.array([ np.linalg.norm(x, 2) for x in delta_high ])
		delta_high_non = np.array([ np.linalg.norm(x, 2) for x in delta_high_non ])
		
		plt.figure(1)
		plt.xlabel("时间 / $/mathrm{h}$"); plt.ylabel("位置误差 / ($\mathrm{km}$)")
		plt.plot(time_range, delta_low, "r-", label="a = 2300km")
		plt.plot(time_range, delta_mid, "g--", label="a = 5300km")
		plt.plot(time_range, delta_high, "b-.", label="a = 10300km")
		plt.legend()
		
		plt.figure(2)
		plt.xlabel("时间 / $\mathrm{h}$"); plt.ylabel("位置误差 / ($\mathrm{km}$)")
		plt.plot(time_range, delta_low_non, "r-", label="a = 2300km")
		plt.plot(time_range, delta_mid_non, "g--", label="a = 5300km")
		plt.plot(time_range, delta_high_non, "b-.", label="a = 10300km")
		plt.legend()
		
		plt.show()
		
		
	def nonspher_orbit(self, number=720, step=120):
		'''非球形引力摄动加速度对不同轨道高度的卫星的轨道精度影响曲线'''
		time_range = np.arange(0, number*120/3600, 120/3600)
		low_hpop = Low[:number].T; mid_hpop = Mid[:number].T; high_hpop = High[:number].T
		low_0, mid_0, high_0 = low_hpop[:, 0], mid_hpop[:, 0], high_hpop[:, 0]
		low_non = orb.integrate_orbit(low_0, number)
		mid_non = orb.integrate_orbit(mid_0, number)
		high_non = orb.integrate_orbit(high_0, number)
		
		delta_low_non = (low_hpop - low_non).T
		delta_mid_non = (mid_hpop - mid_non).T
		delta_high_non = (high_hpop - high_non).T
		
		delta_low_non = np.array([ np.linalg.norm(x, 2) for x in delta_low_non ])
		delta_mid_non = np.array([ np.linalg.norm(x, 2) for x in delta_mid_non ])
		delta_high_non = np.array([ np.linalg.norm(x, 2) for x in delta_high_non ])
		
		plt.figure(2)
		plt.xlabel("时间 / $\mathrm{h}$"); plt.ylabel("位置误差 / ($\mathrm{km}$)")
		plt.plot(time_range, delta_low_non, "r-", label="a = 2300km")
		plt.plot(time_range, delta_mid_non, "g--", label="a = 5300km")
		plt.plot(time_range, delta_high_non, "b-.", label="a = 10300km")
		plt.legend()
		
		plt.show()
		
		
	def Third_orbit(self, number=720, step=120):
		'''地球或太阳摄动对不同轨道高度的卫星的轨道精度影响曲线, 分别取F = F0+F2(或F3)'''
		time_range = np.arange(0, number*120/3600, 120/3600)
		low_hpop = Low[:number].T; mid_hpop = Mid[:number].T; high_hpop = High[:number].T
		low_0, mid_0, high_0 = low_hpop[:, 0], mid_hpop[:, 0], high_hpop[:, 0]
		low_non = orb.integrate_orbit(low_0, number)
		mid_non = orb.integrate_orbit(mid_0, number)
		high_non = orb.integrate_orbit(high_0, number)
		
		delta_low_non = (low_hpop - low_non).T
		delta_mid_non = (mid_hpop - mid_non).T
		delta_high_non = (high_hpop - high_non).T
		
		delta_low_non = np.array([ np.linalg.norm(x, 2) for x in delta_low_non ])
		delta_mid_non = np.array([ np.linalg.norm(x, 2) for x in delta_mid_non ])
		delta_high_non = np.array([ np.linalg.norm(x, 2) for x in delta_high_non ])
		
		plt.figure(2)
		plt.xlabel("时间 / $\mathrm{h}$"); plt.ylabel("位置误差 / ($\mathrm{km}$)")
		plt.plot(time_range, delta_low_non, "r-", label="a = 2300km")
		plt.plot(time_range, delta_mid_non, "g--", label="a = 5300km")
		plt.plot(time_range, delta_high_non, "b-.", label="a = 10300km")
		plt.legend()
		
		plt.show()
		
	
	def complete_orbit(self, number=720, step=120):
		'''考虑完整摄动对不同轨道高度的卫星位置精度的影响曲线'''
		time_range = np.arange(0, number*120/3600, 120/3600)
		low_hpop = Low[:number].T; mid_hpop = Mid[:number].T; high_hpop = High[:number].T
		low_0, mid_0, high_0 = low_hpop[:, 0], mid_hpop[:, 0], high_hpop[:, 0]
		low_non = orb.integrate_orbit(low_0, number)
		mid_non = orb.integrate_orbit(mid_0, number)
		high_non = orb.integrate_orbit(high_0, number)
		
		delta_low_non = (low_hpop - low_non).T
		delta_mid_non = (mid_hpop - mid_non).T
		delta_high_non = (high_hpop - high_non).T
		
		delta_low_non = np.array([ np.linalg.norm(x, 2) for x in delta_low_non ])
		delta_mid_non = np.array([ np.linalg.norm(x, 2) for x in delta_mid_non ])
		delta_high_non = np.array([ np.linalg.norm(x, 2) for x in delta_high_non ])
		
		plt.figure(2)
		plt.xlabel("时间 / $\mathrm{h}$"); plt.ylabel("位置误差 / ($\mathrm{km}$)")
		plt.plot(time_range, delta_low_non, "r-", label="a = 2300km")
		plt.plot(time_range, delta_mid_non, "g--", label="a = 5300km")
		plt.plot(time_range, delta_high_non, "b-.", label="a = 10300km")
		plt.legend()
		
		plt.show()
		
		
	


		
		
if __name__ == "__main__":
	
	sec = Second_Chapter()
	sec.complete_orbit()
	
	# X = nav.extend_kf(number)
	X = nav.unscented_kf(number)
	np.save("npy/error_200m.npy", X)
	nav.plot_filter(X, number)