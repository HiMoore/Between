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
mpl.rcParams["figure.figsize"] = (3.2, 2.2); mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['legend.fontsize'] = 8; mpl.rcParams['axes.labelsize'] = 8;
mpl.rcParams['xtick.labelsize'] = 8; mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['savefig.dpi'] = 400; mpl.rcParams['figure.dpi'] = 400

ob, nav = Orbit(), Navigation()
# 原来的卫星数据，偏心率相差0.01的两颗卫星
HPOP_1 = ( pd.read_csv("STK/Part_2/1_Inertial_HPOP_660.csv", nrows=NUMBER, usecols=range(1,7)) ).values
HPOP_2 = ( pd.read_csv("STK/Part_2/2_Inertial_HPOP_660.csv", nrows=NUMBER, usecols=range(1,7)) ).values


class Test_Orbit(Orbit):

	def __init__(self):
		return

	def singleDay_error(self):
		number, t0 = 720, 0
		HPOP = np.array( pd.read_csv("STK/Part_2/1_Inertial_HPOP_660.csv", nrows=number, usecols=range(1,7)) ).T	# 取前number个点进行试算
		# HPOP = (( pd.read_csv("STK/0.6e/Sat_1_e=0.6, i=28.5_120s.csv", nrows=number, usecols=range(1,7)) ).values).T
		TwoBody = np.array( pd.read_csv("STK/Part_2/Inertial_TwoBody_30d.csv", nrows=number, usecols=range(1,7)) ).T
		rv_0 = HPOP[:, 0]; time_range = np.arange(0, number*STEP/3600, STEP/3600)
		print(rv_0)
		# TwoBody = self.integrate_twobody(rv_0, number)
		J2 = self.integrate_J2(rv_0, number)
		orbit = self.integrate_orbit(rv_0, number)
		# orbit = (( pd.read_csv("STK/0.6e/Sat_1_e=0.6, i=28.5_120s_J2.csv", nrows=number, usecols=range(1,7)) ).values).T
		delta_1 = HPOP - TwoBody
		delta_2 = orbit - TwoBody
		delta_3 = HPOP - orbit
		delta_4 = HPOP - J2
		
		plt.figure(1)
		plt.xlabel("时间 / $\mathrm{h}$"); plt.ylabel("位置误差 / ($\mathrm{km}$)")
		plt.plot(time_range, delta_1[0], "r-", label="x")
		plt.plot(time_range, delta_1[1], "b--", label="y")
		plt.plot(time_range, delta_1[2], "g-.", label="z")
		plt.legend(); plt.tight_layout(); plt.savefig("Figure/中心引力单日误差图.png", dpi=400, bbox_inches='tight')
		
		plt.figure(2)
		plt.xlabel("时间 / $\mathrm{h}$"); plt.ylabel("位置误差 / ($\mathrm{km}$)")
		plt.plot(time_range, delta_2[0], "r-", label="x")
		plt.plot(time_range, delta_2[1], "b--", label="y")
		plt.plot(time_range, delta_2[2], "g-.", label="z")
		plt.legend(); plt.tight_layout(); plt.savefig("Figure/各种摄动加速度.png", dpi=400, bbox_inches='tight')
		
		plt.figure(3)
		plt.xlabel("时间 / $\mathrm{h}$"); plt.ylabel("位置误差 / ($\mathrm{km}$)")
		plt.plot(time_range, delta_3[0], "r-", label="x")
		plt.plot(time_range, delta_3[1], "b--", label="y")
		plt.plot(time_range, delta_3[2], "g-.", label="z")
		plt.legend(); plt.tight_layout(); plt.savefig("Figure/主要摄动单日误差图.png", dpi=400, bbox_inches='tight')
		
		plt.figure(4)
		plt.xlabel("时间 / $\mathrm{h}$"); plt.ylabel("位置误差 / ($\mathrm{km}$)")
		plt.plot(time_range, delta_4[0], "r-", label="x")
		plt.plot(time_range, delta_4[1], "b--", label="y")
		plt.plot(time_range, delta_4[2], "g-.", label="z")
		plt.legend(); plt.tight_layout(); plt.savefig("Figure/J2摄动单日误差图.png", dpi=400, bbox_inches='tight')
		
		# fig = plt.figure(1)
		# ax = Axes3D(fig)
		# ax.plot(HPOP[0], HPOP[1], HPOP[2], antialiased=True)
		# ax.scatter(0, 0, 0, marker="x", s=500, c='r')
		# ax.set_xlabel("x轴 / ($\mathrm{km}$)", fontsize=22)
		# ax.set_ylabel("y轴 / ($\mathrm{km}$)", fontsize=22)
		# ax.set_zlabel("z轴 / ($\mathrm{km}$)", fontsize=22)
		# ax.set_xticklabels([-2000, -1000, 0, 1000, 2000], fontdict={"fontsize": 18} )
		# ax.set_yticklabels([-2000, -1500, -1000, -500, 0, 500, 1000, 1500, 2000], fontdict={"fontsize": 18} )
		# ax.set_zticklabels([], fontdict={"fontsize": 20} )
		
		plt.show()
		
		
	def draw_accelerate(self, number=360, step=120):
		data = pd.read_csv("STK/Part_2/1_Inertial_HPOP_660.csv", nrows=number, usecols=range(1,4))
		r_array = data.values
		orb = Orbit()
		a0 = np.array([ np.linalg.norm(orb.centreGravity(r_sat=r_sat), 2) for r_sat in r_array ]) * 1000
		utc_array = (orb.generate_time(start_t="20180101", end_t="20180131"))[:number]
		utcJD_list = [ time_utc.to_julian_date() for time_utc in utc_array ]
		tdbJD_list = [ time_utc.to_julian_date() + 69.184/86400 for time_utc in utc_array ]
		a1 = np.array([ np.linalg.norm(orb.nonspher_moon(r_sat, tdb_jd)) for (r_sat, tdb_jd) in zip(r_array, tdbJD_list) ]) * 1000
		a_J2 = np.array([ np.linalg.norm(orb.nonspher_J2(r_sat, tdb_jd)) for (r_sat, tdb_jd) in zip(r_array, tdbJD_list) ])
		a_earth = np.array([ np.linalg.norm(orb.thirdEarth(r_sat, tdb_jd)) for (r_sat, tdb_jd) in zip(r_array, tdbJD_list) ]) * 1000
		a_sun = np.array([ np.linalg.norm(orb.thirdSun(r_sat, tdb_jd)) for (r_sat, tdb_jd) in zip(r_array, tdbJD_list) ]) * 1000
		a_solar = np.array([ np.linalg.norm(orb.solarPress(r_sat, tdb_jd)) for (r_sat, tdb_jd) in zip(r_array, tdbJD_list) ]) * 1000
		time_range = np.arange(0, number*step/9000, step/9000)
		
		# plt.figure(1)
		# plt.xlabel("轨道数量 / 个"); plt.ylabel(r"加速度 / $(\mathrm{m/s^2})$")
		# plt.plot(time_range, a0, "r-", label="中心天体引力加速度")
		# a = a0+a1+a_sun+a_earth+a_solar
		# plt.plot(time_range, a, "g-", label="完整摄动加速度")
		# plt.legend(); plt.tight_layout(); plt.savefig("Figure/中心天体加速度.png", dpi=400, bbox_inches='tight')
		
		plt.figure(2)
		plt.xlabel("轨道数量 / 个"); plt.ylabel(r"加速度 / $(\mathrm{m/s^2})$")
		plt.plot(time_range, a1, "r-", label="非球形引力摄动加速度")
		plt.plot(time_range, a_earth, "b--", label="地球引力摄动加速度")
		plt.plot(time_range, a_sun, "g-.", label="太阳引力摄动加速度")
		plt.plot(time_range, a_solar, "m:", label="太阳光压摄动加速度")
		plt.legend(); plt.tight_layout(); plt.savefig("Figure/各种摄动加速度.png", dpi=400, bbox_inches='tight')
		
		plt.figure(3)
		plt.xlabel("轨道数量 / 个"); plt.ylabel(r"加速度 / $(\mathrm{m/s^2})$")
		plt.plot(time_range, a1, "r-", label="非球形引力摄动加速度")
		plt.plot(time_range, a_earth, "b--", label="地球三体引力摄动加速度")
		plt.legend(); plt.tight_layout(); plt.savefig("Figure/主要摄动加速度.png", dpi=400, bbox_inches='tight')
		
		plt.figure(4)
		plt.xlabel("轨道数量 / 个"); plt.ylabel(r"加速度 / $(\mathrm{m/s^2})$")
		plt.plot(time_range, a_sun, "g-", label="太阳三体引力摄动加速度")
		plt.plot(time_range, a_solar, "m--", label="太阳光压摄动加速度")
		plt.legend(); plt.tight_layout(); plt.savefig("Figure/次要摄动加速度.png", dpi=400, bbox_inches='tight')
		
		plt.figure(5)
		plt.xlabel("轨道数量 / 个"); plt.ylabel(r"加速度 / $(\mathrm{m/s^2})$")
		plt.plot(time_range, a1, "r-", label="非球形引力摄动加速度")
		plt.plot(time_range, a_J2, "b--", label="J2项摄动加速度")
		plt.legend(); plt.tight_layout(); plt.savefig("Figure/J2项与球形对比.png", dpi=400, bbox_inches='tight')

		plt.show()
		
		
	def state_transMatrix(self, number=720):
		
		old_A = ( pd.read_csv("STK/Part_2/1_Inertial_HPOP_660.csv", nrows=number, usecols=range(1,7)) ).values
		old_B = ( pd.read_csv("STK/Part_2/2_Inertial_HPOP_660.csv", nrows=number, usecols=range(1,7)) ).values
		A_0, B_0 = old_A[0], old_B[0]
		A = ob.integrate_orbit(A_0, number)
		B = ob.integrate_orbit(B_0, number)
		X_A = [ A_0 ]
		for i in range(1, number):
			r1, r2 = A[i, 0:3], B[i, 0:3]
			Phi = nav.jacobian_double(t, r1, r2)
			New_A = np.dot(Phi, X_A[i-1])
			X_A.append(New_A)
			


if __name__ == "__main__":

	test = Test_Orbit()
	test.singleDay_error()
	# number = 720
	# time_range = range(0, number*STEP, STEP)
	# # orbit_1 = (ob.integrate_orbit(HPOP_1[0, :], number)).T
	# # orbit_2 = (ob.integrate_orbit(HPOP_2[0, :], number)).T
	# # np.save("npy/orbit_1.npy", orbit_1); np.save("npy/orbit_2.npy", orbit_2)
	# A, B = np.load("npy/orbit_1.npy"), np.load("npy/orbit_2.npy")
	# X_A = [ np.hstack((A[0], B[0])) ]
	# phi_1 = ob.jacobian_single(0, A[0, 0:3])
	# phi_2 = ob.jacobian_single(0, B[0, 0:3])
	

