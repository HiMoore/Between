# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import *
from orbit_km import *

import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['NSimSun', 'Times New Roman'] # 指定默认字体
mpl.rcParams['font.family']='sans-serif'
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
mpl.rcParams["figure.figsize"] = (3.2, 2.2); mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['legend.fontsize'] = 8; mpl.rcParams['axes.labelsize'] = 8;
mpl.rcParams['xtick.labelsize'] = 8; mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['savefig.dpi'] = 400; mpl.rcParams['figure.dpi'] = 400

NUMBER = 1440; orb = Orbit(); 
data = ( pd.read_csv("STK/Part_2/1_Inertial_HPOP_660.csv", nrows=NUMBER, usecols=range(1,4)) ).values
# 不同轨道高度的卫星数据
Low = ( pd.read_csv("STK/different_height/Sat_1_LowOrbit_120s.csv", nrows=NUMBER, usecols=range(1,4)) ).values
Mid = ( pd.read_csv("STK/different_height/Sat_1_MidOrbit_120s.csv", nrows=NUMBER, usecols=range(1,4)) ).values
High = ( pd.read_csv("STK/different_height/Sat_1_HighOrbit_120s.csv", nrows=NUMBER, usecols=range(1,4)) ).values
# # 基础卫星数据，轨道倾角分别为0和28.5°
basic_i0A = ( pd.read_csv("STK/basic/Sat_1_i0.csv", nrows=NUMBER, usecols=range(1,7)) ).values
# basic_i0B = ( pd.read_csv("STK/basic/Sat_1_i0.csv", nrows=NUMBER, usecols=range(1,7)) ).values
basic_i28A = ( pd.read_csv("STK/basic/Sat_1_i28.csv", nrows=NUMBER, usecols=range(1,7)) ).values
basic_i28B = ( pd.read_csv("STK/basic/Sat_1_i28.csv", nrows=NUMBER, usecols=range(1,7)) ).values
# # # 原来的卫星数据，偏心率相差0.01的两颗卫星
# old_A = ( pd.read_csv("STK/Part_2/1_Inertial_HPOP_660.csv", nrows=NUMBER*7, usecols=range(1,7)) ).values
# old_B = ( pd.read_csv("STK/Part_2/2_Inertial_HPOP_660.csv", nrows=NUMBER*7, usecols=range(1,7)) ).values
# # 60s 迭代步长数据
# A_60s = ( pd.read_csv("STK/60_30s/Sat_1_60s.csv", nrows=NUMBER, usecols=range(1,7)) ).values
# B_60s = ( pd.read_csv("STK/60_30s/Sat_2_60s.csv", nrows=NUMBER, usecols=range(1,7)) ).values
# A_30s = ( pd.read_csv("STK/60_30s/Sat_1_30s.csv", nrows=NUMBER, usecols=range(1,7)) ).values
# B_30s = ( pd.read_csv("STK/60_30s/Sat_2_30s.csv", nrows=NUMBER, usecols=range(1,7)) ).values
# 大偏心率(0.6)的数据
e06_A = ( pd.read_csv("STK/0.6e/Sat_1_e=0.6, i=28.5_120s.csv", nrows=NUMBER, usecols=range(1,7)) ).values
e06_B = ( pd.read_csv("STK/0.6e/Sat_2_e=0.6, i=28.5_120s.csv", nrows=NUMBER, usecols=range(1,7)) ).values
##################################################################################################

utc_array = (orb.generate_time(start_t="20180101", end_t="20180131"))[:NUMBER]
utcJD_list = [ time_utc.to_julian_date() for time_utc in utc_array ]
tdbJD_list = [ time_utc.to_julian_date() + 69.184/86400 for time_utc in utc_array ]

# 第二章中的图片
class Second_Chapter:

	def __init__(self):
		return
		
		
	def __del__(self):
		return
	
		
	def diff_order_acc(self, number=200, step=120):
		'''绘制10阶, 30阶, 80阶的摄动加速度量级对比'''
		a_10 = np.array([ np.linalg.norm(orb.nonspher_moon(r_sat, tdb_jd, lm=10)) for (r_sat, tdb_jd) in zip(data[:number], tdbJD_list[:number]) ]) * 1000
		a_20 = np.array([ np.linalg.norm(orb.nonspher_moon(r_sat, tdb_jd, lm=20)) for (r_sat, tdb_jd) in zip(data[:number], tdbJD_list[:number]) ]) * 1000
		a_30 = np.array([ np.linalg.norm(orb.nonspher_moon(r_sat, tdb_jd, lm=30)) for (r_sat, tdb_jd) in zip(data[:number], tdbJD_list[:number]) ]) * 1000
		a_80 = np.array([ np.linalg.norm(orb.nonspher_moon(r_sat, tdb_jd, lm=80)) for (r_sat, tdb_jd) in zip(data[:number], tdbJD_list[:number]) ]) * 1000
		
		time_range = np.arange(0, number*step/9000, step/9000)
		plt.figure(1)
		plt.xlabel("轨道数量 / 个"); plt.ylabel(r"加速度 / $(\mathrm{m/s^2})$")
		plt.plot(time_range, a_10, "r-", label="10阶次")
		plt.plot(time_range, a_20, "g--", label="20阶次")
		plt.plot(time_range, a_30, "b-.", label="30阶次")
		plt.plot(time_range, a_80, "m:", label="80阶次")
		plt.legend(); plt.tight_layout(); plt.savefig("Figure/不同阶次非球形引力摄动加速度.png", dpi=400, bbox_inches='tight')
		plt.show()

		
	def diff_height_orbit(self, number=1440, step=120):
		'''绘制低轨, 中轨, 高轨的月球卫星三维轨道图'''
		number = number // 5
		fig = plt.figure(1)
		ax = Axes3D(fig)
		ax.plot(Low[:number, 0], Low[:number, 1], Low[:number, 2], "c-", label='低轨卫星')
		# ax.plot(Mid[:number, 0], Mid[:number, 1], Mid[:number, 2], "g--", label='中轨卫星')
		# number = int(number * 3.2)
		# ax.plot(High[:number, 0], High[:number, 1], High[:number, 2], "b-.", label='高轨卫星')
		ax.scatter(0, 0, 0, marker="x", s=50, c='r', label='月心')
		ax.set_xlabel("X轴 / ($\mathrm{km}$)")
		ax.set_ylabel("Y轴 / ($\mathrm{km}$)")
		ax.set_zlabel("Z轴 / ($\mathrm{km}$)")
		plt.legend(); plt.tight_layout()
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
		plt.plot(time_low, Low_solar, "m:", label="太阳光压摄动加速度")
		plt.legend(); plt.tight_layout(); plt.savefig("Figure/低轨卫星受到的不同摄动加速度量级.png", dpi=400, bbox_inches='tight')
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
		plt.legend(); plt.tight_layout(); plt.savefig("Figure/中轨卫星受到的不同摄动加速度量级.png", dpi=400, bbox_inches='tight')
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
		plt.legend(); plt.tight_layout(); plt.savefig("Figure/高轨卫星受到的不同摄动加速度量级.png", dpi=400, bbox_inches='tight')

		# 不同轨道高度, 非球形摄动和地球引力摄动量级变化图
		# 非球形引力摄动加速度在不同高度的变化曲线
		plt.figure(4)
		plt.xlabel("时间 / 2 min"); plt.ylabel(r"加速度 / $(\mathrm{m/s^2})$")
		plt.plot(Low_non, "r-", label="a=2300 km")
		plt.plot(Mid_non, "g--", label="a=5300 km")
		plt.plot(High_non, "b-.", label="a=10300 km")
		plt.legend(); plt.tight_layout(); plt.savefig("Figure/非球形引力摄动加速度在不同高度的变化曲线.png", dpi=400, bbox_inches='tight')
		
		plt.figure(5)
		plt.xlabel("时间 / 2 min"); plt.ylabel(r"加速度 / $(\mathrm{m/s^2})$")
		plt.plot(Low_earth, "r-", label="a=2300 km")
		plt.plot(Mid_earth, "g--", label="a=5300 km")
		plt.plot(High_earth, "b-.", label="a=10300 km")
		plt.legend(); plt.tight_layout(); plt.savefig("Figure/地球引力摄动加速度在不同高度的变化曲线.png", dpi=400, bbox_inches='tight')
		
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
		
		
	def j2_orbit(self, number=720, step=120):
		'''绘制J2摄动加速度对不同轨道高度的轨道精度影响曲线'''
		time_range = np.arange(0, number*120/3600, 120/3600)
		low_hpop = Low[:number].T; mid_hpop = Mid[:number].T; high_hpop = High[:number].T
		low_0, mid_0, high_0 = low_hpop[:, 0], mid_hpop[:, 0], high_hpop[:, 0]
		low_two = orb.integrate_twobody(low_0, number)
		mid_two = orb.integrate_twobody(mid_0, number)
		high_two = orb.integrate_twobody(high_0, number)
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
		plt.xlabel("时间 / $\mathrm{h}$"); plt.ylabel("位置误差 / ($\mathrm{km}$)")
		plt.plot(time_range, delta_low, "r-", label="a = 2300km")
		plt.plot(time_range, delta_mid, "g--", label="a = 5300km")
		plt.plot(time_range, delta_high, "b-.", label="a = 10300km")
		plt.legend(); plt.tight_layout(); plt.savefig("Figure/只考虑中心引力的不同轨道高度的位置误差.png", dpi=400, bbox_inches='tight')

		
		plt.figure(2)
		plt.xlabel("时间 / $\mathrm{h}$"); plt.ylabel("位置误差 / ($\mathrm{km}$)")
		plt.plot(time_range, delta_low_non, "r-", label="a = 2300km")
		plt.plot(time_range, delta_mid_non, "g--", label="a = 5300km")
		plt.plot(time_range, delta_high_non, "b-.", label="a = 10300km")
		plt.legend(); plt.tight_layout(); plt.savefig("Figure/只考虑J2项的不同轨道高度的位置误差.png", dpi=400, bbox_inches='tight')

		
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
		plt.legend(); plt.tight_layout(); plt.savefig("Figure/SecChapter/只考虑非球形的不同轨道高度的位置误差.png", dpi=400, bbox_inches='tight')
		
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
		plt.legend(); plt.tight_layout(); plt.savefig("Figure/SecChapter/只考虑地球引力摄动的不同轨道高度的位置误差.png", dpi=400, bbox_inches='tight')
		
		plt.show()
		
	
	def complete_orbit(self, number=720, step=120):
		'''考虑完整摄动对不同轨道高度的卫星位置精度的影响曲线'''
		time_range = np.arange(0, number*STEP/3600, STEP/3600)
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
	
##################################################################################################	
		
		
	
# 第三章中的图片(部分)	
class Third_Chapter:

	def __init__(self):
		return
		
	def __del__(self):
		return
		
	def r1Crossr2(self, number=120):
		NonCross = np.array([ np.cross(r1, r2) for r1, r2 in zip(basic_i0A[:number, 0:3], basic_i28B[:number, 0:3]) ]) / 1e-5
		time_range = np.arange(0, number*STEP/3600, STEP/3600)
		fig = plt.figure(1, figsize=(5, 3))
		ax = Axes3D(fig)
		ax.plot(NonCross[:number, 0], NonCross[:number, 1], NonCross[:number, 2], "g--", linewidth=1)
		ax.set_xlabel("X轴 / ($\mathrm{km}$)")
		ax.set_ylabel("Y轴 / ($\mathrm{km}$)")
		ax.set_zlabel("Z轴 / ($\mathrm{km}$)")
		plt.savefig("Figure/ThirdChapter/不同轨道面的位置矢量叉乘.png", dpi=400, bbox_inches='tight')
		
		# CopCross = np.array([ np.cross(r1, r2) for r1, r2 in zip(basic_i28A[:number, 0:3], basic_i28B[:number, 0:3]) ])
		# fig = plt.figure(2, figsize=(5, 3))
		# ax = Axes3D(fig)
		# ax.plot(CopCross[:number, 0], CopCross[:number, 1], CopCross[:number, 2], "r-", linewidth=1)
		# ax.set_xlabel("X轴 / ($\mathrm{km}$)")
		# ax.set_ylabel("Y轴 / ($\mathrm{km}$)")
		# ax.set_zlabel("Z轴 / ($\mathrm{km}$)")
		# plt.savefig("Figure/ThirdChapter/同轨道面的位置矢量叉乘.png", dpi=400, bbox_inches='tight')
		
		plt.show()
		
		
		
##################################################################################################
	

	
# 第四章中的图片(部分)	
class FourChapter:

	def __init__(self):
		return
		
		
	def __del__(self):
		return
	
	def diff_incline_orbit(self, number=700, step=120):
		'''绘制不同轨道平面的月球卫星三维轨道图'''
		number = number // 2
		fig = plt.figure(1, figsize=(7, 4))
		ax = Axes3D(fig)
		basic_i0A, basic_i28B = e06_A, e06_B
		ax.plot(basic_i0A[:number, 0], basic_i0A[:number, 1], basic_i0A[:number, 2], "c-", label='A星, i=28.5', linewidth=1)
		ax.plot(basic_i28B[:number, 0], basic_i28B[:number, 1], basic_i28B[:number, 2], "g--", label='B星, i=28.5', linewidth=1)
		ax.scatter(0, 0, 0, marker="x", s=50, c='r', label='月心')
		ax.set_xlabel("X轴 / ($\mathrm{km}$)")
		ax.set_ylabel("Y轴 / ($\mathrm{km}$)")
		ax.set_zlabel("Z轴 / ($\mathrm{km}$)")
		ax.quiver3D(basic_i0A[0,0], basic_i0A[0,1], basic_i0A[0,2], basic_i28B[1,0]-basic_i0A[0,0], basic_i28B[1,1]-basic_i0A[0,1], basic_i28B[1,2]-basic_i0A[0,2])
		ax.quiver3D(basic_i0A[2,0], basic_i0A[2,1], basic_i0A[2,2], basic_i28B[3,0]-basic_i0A[2,0], basic_i28B[3,1]-basic_i0A[2,1], basic_i28B[3,2]-basic_i0A[2,2])
		ax.quiver3D(basic_i0A[4,0], basic_i0A[4,1], basic_i0A[4,2], basic_i28B[5,0]-basic_i0A[4,0], basic_i28B[5,1]-basic_i0A[4,1], basic_i28B[5,2]-basic_i0A[4,2])
		ax.quiver3D(basic_i0A[6,0], basic_i0A[6,1], basic_i0A[6,2], basic_i28B[7,0]-basic_i0A[6,0], basic_i28B[7,1]-basic_i0A[6,1], basic_i28B[7,2]-basic_i0A[6,2])
		ax.quiver3D(basic_i0A[8,0], basic_i0A[8,1], basic_i0A[8,2], basic_i28B[9,0]-basic_i0A[8,0], basic_i28B[9,1]-basic_i0A[8,1], basic_i28B[9,2]-basic_i0A[8,2])
		plt.legend(); plt.savefig("Figure/FourChapter/不同轨道平面的三维轨道图.png", dpi=400, bbox_inches='tight')
		plt.show()
		
		
	def vector_change(self, number=180, step=120):
		'''绘制星间矢量在惯性空间下的变化图'''
		fig = plt.figure(1)
		time_range = np.arange(0, number*120/3600, 120/3600)
		delta_r = old_A[:number, 0:3] - old_B[:number, 0:3]
		plt.xlabel("时间 / $\mathrm{h}$"); plt.ylabel("星间矢量 / ($\mathrm{km}$)")
		plt.figure(1)
		plt.plot(time_range, delta_r[:, 0], "r-", label="X轴")
		plt.plot(time_range, delta_r[:, 1], "g--", label="Y轴")
		plt.plot(time_range, delta_r[:, 2], "b-.", label="Z轴")
		plt.savefig("Figure/FourChapter/星间矢量在惯性空间下的2D图.png", dpi=400, bbox_inches='tight')
		
		fig = plt.figure(2)
		ax = Axes3D(fig)
		ax.plot(delta_r[:number, 0], delta_r[:number, 1], delta_r[:number, 2], "g--")
		ax.set_xlabel("X轴 / ($\mathrm{km}$)")
		ax.set_ylabel("Y轴 / ($\mathrm{km}$)")
		ax.set_zlabel("Z轴 / ($\mathrm{km}$)")
		plt.savefig("Figure/FourChapter/星间矢量在惯性空间下的3D图.png", dpi=400, bbox_inches='tight')
		plt.show()
		
		
	def statis_rms(self, number=720, step=120):
		'''统计不同Rk的滤波均方根误差结果'''
		Rk_1 = np.load("npy/i=28.5, P0=3e-2.npy")[:, 0:6]
		Rk_01 = np.load("npy/i=28, P0=3e-2, RkDivide10.npy")[:, 0:6]
		Rk_001 = np.load("npy/i=28, P0=3e-2, RkDivide100.npy")[:, 0:6]
		rms_1, rms_2, rms_3 = np.zeros(6), np.zeros(6), np.zeros(6)
		k = 360
		for i in range(6):
			rms_1[i] = sqrt( np.sum( np.power(Rk_1[k:number, i] - old_A[k:number, i], 2) ) / (number-k) ) * 1e3
			rms_2[i] = sqrt( np.sum( np.power(Rk_01[k:number, i] - old_A[k:number, i], 2)) / (number-k) ) * 1e3
			rms_3[i] = sqrt( np.sum( np.power(Rk_001[k:number, i] - old_A[k:number, i], 2)) / (number-k) ) * 1e3
		print(rms_1, "\n\n", rms_2, "\n\n", rms_3)
		
		
	def static_ure(self, X, HPOP_1, HPOP_2,number=720):
		'''统计用户伪距误差'''
		X1, X2 = X[:number, :6], X[:number, 6:]
		delta_x1 = X1[:number] - HPOP_1[:number]
		delta_x2 = X2[:number] - HPOP_2[:number]
		x1_rtn = np.array([ np.dot( orb.inertial2RTN(rv), X )[:3] for rv, X in zip(X1[:, 0:6], delta_x1[:, 0:3]) ])
		x2_rtn = np.array([ np.dot( orb.inertial2RTN(rv), X )[:3] for rv, X in zip(X2[:, 0:6], delta_x2[:, 0:3]) ])
		URE_A = np.sqrt([ pow(X[0], 2) + 1/49 * ( pow(X[1], 2) + pow(X[2], 2) ) for X in x1_rtn ])
		URE_B = np.sqrt([ pow(X[0], 2) + 1/49 * ( pow(X[1], 2) + pow(X[2], 2) ) for X in x2_rtn ])
		time_range = np.arange(0, number*120/3600, 120/3600)
		up_error = 15 + 1e2 * np.power(e, -time_range)

		# plt.figure(1)
		# plt.xlabel("时间 / $\mathrm{h}$"); plt.ylabel("位置误差 / ($\mathrm{m}$)")
		# plt.plot(time_range, x1_rtn[:, 0] * 1000, "r-", label="x")
		# plt.plot(time_range, x1_rtn[:, 1] * 1000, "b--", label="y")
		# plt.plot(time_range, x1_rtn[:, 2] * 1000, "g-.", label="z")
		# plt.plot(time_range, up_error, "k:")
		# plt.plot(time_range, low_error, "k:", label=" $\mathrm{\pm 50 m}$")
		# plt.legend(); plt.tight_layout(); plt.savefig("Figure/FourChapter/其它/URE_A_Three.png", dpi=400, bbox_inches='tight')
		
		# plt.figure(2)
		# plt.xlabel("时间 / $\mathrm{h}$"); plt.ylabel("位置误差 / ($\mathrm{m}$)")
		# plt.plot(time_range, x2_rtn[:, 0] * 1000, "r-", label="x")
		# plt.plot(time_range, x2_rtn[:, 1] * 1000, "b--", label="y")
		# plt.plot(time_range, x2_rtn[:, 2] * 1000, "g-.", label="z")
		# plt.plot(time_range, up_error, "k:")
		# plt.plot(time_range, low_error, "k:", label=" $\mathrm{\pm 50 m}$")
		# plt.legend(); plt.tight_layout(); plt.savefig("Figure/FourChapter/其它/URE_B_Three.png", dpi=400, bbox_inches='tight')
		# plt.show()
		
		plt.figure(3)
		plt.xlabel("时间 / $\mathrm{h}$"); plt.ylabel("位置误差 / ($\mathrm{m}$)")
		plt.plot(time_range, URE_A * 1000, "r-", label="x")
		plt.plot(time_range, URE_B * 1000, "b--", label="y")
		plt.plot(time_range, up_error, "k:", label=" $\mathrm{\pm 15 m}$")
		plt.legend(loc=2); plt.tight_layout(); plt.savefig("Figure/FourChapter/其它/URE_B_Three.png", dpi=400, bbox_inches='tight')
		plt.show()
		
		
if __name__ == "__main__":
	
	# X = np.load("npy/7days_i=28, P0=3e-2.npy")
	# HPOP_1, HPOP_2 = old_A, old_B
	# sec = Second_Chapter()
	# # sec.static_ure(X, HPOP_1, HPOP_2,number=720*7)
	# sec.diff_height_acc()
	# third = Third_Chapter()
	# third.r1Crossr2()
	four = FourChapter()
	four.diff_incline_orbit()
