# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from math import *
from scipy.integrate import ode,solve_ivp, RK45, LSODA
from scipy.sparse import coo_matrix, dia_matrix, bsr_matrix, block_diag, diags, identity
from functools import partial
from orbit_km import *
from filterpy.kalman import unscented_transform, MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import ExtendedKalmanFilter as EKF


Time, NUMBER = 0, 1440
Qk = np.diag([ 1e-10, 1e-10, 1e-10, 1e-14, 1e-14, 1e-14, 1e-10, 1e-10, 1e-10, 1e-14, 1e-14, 1e-14 ])	# 12*12
Rk = np.power( np.diag( [1e-3, radians(10/3600), radians(10/3600), radians(10/3600)] ), 2 )	# 4*4, sigma_r*I
# 基础卫星数据，轨道倾角分别为0和28.5°
basic_i0A = ( pd.read_csv("STK/basic/Sat_1_i0.csv", nrows=NUMBER, usecols=range(1,7)) ).values
basic_i0B = ( pd.read_csv("STK/basic/Sat_2_i0.csv", nrows=NUMBER, usecols=range(1,7)) ).values
basic_i28A = ( pd.read_csv("STK/basic/Sat_1_i28.csv", nrows=NUMBER, usecols=range(1,7)) ).values
basic_i28B = ( pd.read_csv("STK/basic/Sat_2_i28.csv", nrows=NUMBER, usecols=range(1,7)) ).values
# 赋值HPOP_1, HPOP_2
HPOP_1, HPOP_2 = basic_i0A, basic_i0B


import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['NSimSun', 'Times New Roman'] # 指定默认字体
mpl.rcParams['font.family']='sans-serif'
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
mpl.rcParams["figure.figsize"] = (15, 9); mpl.rcParams['lines.linewidth'] = 4
mpl.rcParams['legend.fontsize'] = 30; mpl.rcParams['axes.labelsize'] = 30;
mpl.rcParams['xtick.labelsize'] = 30;mpl.rcParams['ytick.labelsize'] = 30


class Navigation(Orbit):
	
	def __init__(self):
		return
		
	def __del__(self):
		return
			
	def coefMatrix_double(self, t, Phi, r1, r2):
		'''计算双星系统在t时刻的状态转移矩阵微分方程的右函数 np.array([ [A1, O], [O, A2] ]) * Phi
		输入：时刻 t,  s; 		状态转移矩阵(待求),  144*1; 	双星位置矢量,  km; 		平均计算时间0.012s'''
		Phi = dia_matrix( Phi.reshape((12, 12), order="F") ) 	# 12*12, sparse
		tdb_jd = (time_utc+int(t)).to_julian_date() + 69.184/86400
		A1 = coo_matrix( self.coefMatrix_single(r1, tdb_jd) )	 # 6*6
		A2 = coo_matrix( self.coefMatrix_single(r2, tdb_jd) )	 # 6*6
		Ft = block_diag((A1, A2), format="bsr")
		Phi_144 = (Ft * Phi).toarray().reshape(144, order="F")
		return Phi_144 	# 144*1, ndarray

		
	def jacobian_double(self, t, r1, r2):
		'''计算双星系统的状态转移矩阵(Jacobian矩阵), 使用RK45完成数值积分, 12*12'''
		Phi_0 = (np.identity(12)).reshape(144, order="F")		# 144*1, ndarray, 按列展开
		solution = solve_ivp( partial(self.coefMatrix_double, r1=r1, r2=r2), (t, t+STEP), Phi_0, method="RK45", \
					rtol=1e-9, atol=1e-9, t_eval=[t+STEP] )
		PHI = (solution.y).reshape(12, 12, order="F")
		return PHI	# 12*12, ndarray
		
		
	def jacobian_approx(self, t, r1, r2):
		'''计算双星系统的状态转移矩阵(Jacobian矩阵), 使用RK45完成数值积分, 12*12'''
		tdb_jd = (time_utc+int(t)).to_julian_date() + 69.184/86400
		F1 = coo_matrix( self.coefMatrix_single(r1, tdb_jd) )	 # 6*6
		F2 = coo_matrix( self.coefMatrix_single(r2, tdb_jd) )	 # 6*6
		Ft = block_diag((F1, F2), format="bsr")
		PHI = identity(12) + Ft*STEP# + 0.5*Ft.power(2)*pow(STEP, 2)
		return PHI	# 12*12, ndarray
		

	def jacobian_measure(self, r1, r2):
		'''计算双星测量模型的Jacobian矩阵, Hk = bsr([ [dh1_dr1, O, dh1_dr2, O], [dh2_dr1, O, dh2_dr2, O] ]),  4*12'''
		r1_r2, zeros = r1-r2, np.zeros((3,3))
		norm = np.linalg.norm(r1_r2, 2)
		h1_r1, h1_r2 = r1_r2/norm, -r1_r2/norm	# 1*3, 1*3
		dh1_dr1, dh1_dr2 = np.hstack((h1_r1, zeros[0])), np.hstack((h1_r2, zeros[0]))	# 1*6, 1*6
		up = np.hstack((dh1_dr1, dh1_dr2))	# 1*12
		h2_r1 = np.identity(3)/norm - np.outer(r1_r2, r1_r2)/norm**3	# 3*3
		h2_r2 = -np.identity(3) + np.outer(r1_r2, r1_r2)/norm**3	# 3*3
		dh2_dr1, dh2_dr2 = np.hstack((h2_r1, zeros)), np.hstack((h2_r2, zeros))	# 3*6, 3*6
		low = np.hstack((dh2_dr1, dh2_dr2))	 # 3*12
		H = np.vstack((up, low))	# 4*12
		return H	# 4*12, ndarray
		
	
	def extend_kf(self, number=240):
		'''扩展卡尔曼滤波算法, 初步看滤波效果'''
		global Time
		P0 = np.diag([ 3e-1, 3e-1, 3e-1, 1e-6, 1e-6, 1e-6, 3e-1, 3e-1, 3e-1, 1e-6, 1e-6, 1e-6 ])
		error = np.random.multivariate_normal(mean=np.zeros(12), cov=P0)
		X0 = np.hstack( (HPOP_1[0], HPOP_2[0]) ) + error
		X, P, I = [X0], [P0], identity(12)
		ekf = EKF(dix_x=12, dim_z=4, compute_log_likelihood=False)
		ekf.x = X0; ekf.F = self.jacobian_double(Time, )

#############################################################################################################
		
	def measure_stk(self, i):
		'''双星系统的测量方程的实际测量输出 Zk = h(Xk) + Vk, 由STK生成并加入噪声, ndarray'''
		delta_r = HPOP_1[i, :3] - HPOP_2[i, :3]
		r_norm = np.linalg.norm(delta_r, 2)
		v_rk = np.random.normal(loc=0, scale=Rk[0][0])	# 测距噪声, 0均值, 测量标准差为5e-3 km
		v_dk = np.random.multivariate_normal(mean=[0,0,0], cov=Rk[1][1]*np.identity(3))	# 测角噪声, 0均值, 协方差为 10角秒*I
		Z = [ r_norm + v_rk];   Z.extend(delta_r/r_norm + v_dk)
		return np.array(Z)	# np.array, (4, )
		
	
	def state_equation(self, X, dt=STEP):
		'''UKF的状态方程, X_k+1 = f(x_k) + w_k, ndarray;	 dt=120s, 为UKF的predict步长;	 计算时间0.20s'''
		global Time
		X1, X2 = X[0:6], X[6:12]
		ode_y1 = solve_ivp( self.complete_dynamic, (Time, Time+dt), X1, method="RK45", rtol=1e-9, atol=1e-12, t_eval=[Time+dt] ).y
		ode_y2 = solve_ivp( self.complete_dynamic, (Time, Time+dt), X2, method="RK45", rtol=1e-9, atol=1e-12, t_eval=[Time+dt] ).y
		y_list = [ x[0] for x in ode_y1 ]; y2 = [ x[0] for x in ode_y2 ]; y_list.extend(y2)
		return np.array(y_list)

	def measure_equation(self, X):
		'''双星系统的测量方程, Z_k = h(X_k) + v_k, ndarray'''
		delta_r = X[0:3] - X[6:9]
		r_norm = np.linalg.norm(delta_r, 2)
		Z = [r_norm];   Z.extend(delta_r/r_norm)
		return np.array(Z)	# np.array, (4, )
		
	
	def unscented_kf(self, number=number):
		global Time
		P0 = np.diag([ 9e-2, 9e-2, 9e-2, 9e-6, 9e-6, 9e-6, 9e-2, 9e-2, 9e-2, 9e-6, 9e-6, 9e-6 ])
		error = np.array([ 0.3, 0.3, 0.3, 3e-1, 3e-1, 3e-1, 0.3, 0.3, 0.3, 3e-1, 3e-1, 3e-1 ]) # np.random.multivariate_normal(mean=np.zeros(12), cov=P0)
		X0 = np.hstack( (HPOP_1[0], HPOP_2[0]) ) + error
		points = MerweScaledSigmaPoints(n=12, alpha=0.001, beta=2.0, kappa=-9)
		ukf = UKF(dim_x=12, dim_z=4, fx=self.state_equation, hx=self.measure_equation, dt=STEP, points=points)
		ukf.x = X0; ukf.P = P0; ukf.R = Rk; ukf.Q = Qk; XF, XP = [X0], [X0]
		print(error, "\n", Qk[0][0], "\n", Rk[0][0])
		for i in range(1, number+1):
			ukf.predict()
			Z = nav.measure_stk(i)
			ukf.update(Z)
			X_Up = ukf.x.copy(); XF.append(X_Up)
			Time = Time+STEP
		XF = np.array(XF)
		return XF
		

	def plot_filter(self, X, number):
		X1, X2 = X[:number, :6], X[:number, 6:]
		time_range = np.arange(0, number*120/3600, 120/3600)
		up_error = 50 + 1e3 * np.power(e, -time_range)
		low_error = -50 - 1e3 * np.power(e, -time_range)
		plt.figure(1)
		plt.xlabel("时间 / $\mathrm{h}$", fontsize=28); plt.ylabel("位置误差 / ($\mathrm{m}$)", fontsize=28)
		plt.xticks(fontsize=24); plt.yticks(fontsize=24)
		plt.plot(time_range, (X1[:number, 0] - HPOP_1[:number, 0]) * 1000, "r-", label="x")
		plt.plot(time_range, (X1[:number, 1] - HPOP_1[:number, 1]) * 1000, "b--", label="y")
		plt.plot(time_range, (X1[:number, 2] - HPOP_1[:number, 2]) * 1000, "g-.", label="z")
		plt.plot(time_range, up_error, "k:")
		plt.plot(time_range, low_error, "k:", label=" $\mathrm{\pm 50 m}$")
		plt.legend(fontsize=24)
		
		plt.figure(2)
		plt.xlabel("时间 / $\mathrm{h}$", fontsize=28); plt.ylabel("速度误差 / ($\mathrm{m/s}$)", fontsize=28)
		plt.xticks(fontsize=24); plt.yticks(fontsize=24)
		plt.plot(time_range, (X1[:number, 3] - HPOP_1[:number, 3]) * 1000, "r-", label="x")
		plt.plot(time_range, (X1[:number, 4] - HPOP_1[:number, 4]) * 1000, "b--", label="y")
		plt.plot(time_range, (X1[:number, 5] - HPOP_1[:number, 5]) * 1000, "g-.", label="z")
		plt.legend(fontsize=24)

		plt.figure(3)
		plt.xlabel("时间 / $\mathrm{h}$", fontsize=28); plt.ylabel("位置误差 / ($\mathrm{m}$)", fontsize=28)
		plt.xticks(fontsize=24); plt.yticks(fontsize=24)
		plt.plot(time_range, (X2[:number, 0] - HPOP_2[:number, 0]) * 1000, "r-", label="x")
		plt.plot(time_range, (X2[:number, 1] - HPOP_2[:number, 1]) * 1000, "b--", label="y")
		plt.plot(time_range, (X2[:number, 2] - HPOP_2[:number, 2]) * 1000, "g-.", label="z")
		plt.plot(time_range, up_error, "k:")
		plt.plot(time_range, low_error, "k:", label=" $\mathrm{\pm 50 m}$")
		plt.legend(fontsize=24)
		
		plt.figure(4)
		plt.xlabel("时间 / $\mathrm{h}$", fontsize=28); plt.ylabel("速度误差 / ($\mathrm{m/s}$)", fontsize=28)
		plt.xticks(fontsize=24); plt.yticks(fontsize=24)
		plt.plot(time_range, (X2[:number, 3] - HPOP_1[:number, 3]) * 1000, "r-", label="x")
		plt.plot(time_range, (X2[:number, 4] - HPOP_1[:number, 4]) * 1000, "b--", label="y")
		plt.plot(time_range, (X2[:number, 5] - HPOP_1[:number, 5]) * 1000, "g-.", label="z")
		plt.legend(fontsize=24)

		plt.show()
		
		
if __name__ == "__main__":
	import cProfile, pstats
	ob = Orbit()
	nav = Navigation()
	number = 720
	
	# # X = nav.extend_kf(number)
	X = nav.unscented_kf(number)
	np.save("npy/error_300m.npy", X)
	# # X = np.load("npy/basic_x.npy")
	nav.plot_filter(X, number)
	