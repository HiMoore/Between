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


HPOP_1 = np.load("STK/HPOP_1.npy")	# r, v
HPOP_2 = np.load("STK/HPOP_2.npy")	# r, v
Time = 0


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
		return Phi_144 	# 144*1, np.array
		
		
	def jacobian_double(self, t, r1, r2):
		'''计算双星系统的状态转移矩阵(Jacobian矩阵), 使用RK45完成数值积分, 12*12'''
		Phi_0 = (np.identity(12)).reshape(144, order="F")		# 144*1, ndarray, 按列展开
		solution = solve_ivp( partial(self.coefMatrix_double, r1=r1, r2=r2), (t, t+STEP), Phi_0, method="RK45", \
					rtol=1e-9, atol=1e-9, t_eval=[t+STEP] )
		ode_y = (solution.y).reshape(12, 12, order="F")
		return bsr_matrix(ode_y, blocksize=(6,6))	# 12*12, sparse (half zero)
		

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
		return bsr_matrix(H, blocksize=(1, 3))	# 4*12, sparse (half zero)
		
	
	def extend_kf(self, number=240):
		'''扩展卡尔曼滤波算法, 初步看滤波效果'''
		X0 = np.hstack( (HPOP_1[0], HPOP_2[0]) )
		P0 = identity(12)
		X, P, I = [X0], [P0], identity(12)
		Rk =  diags([1e-3, 10/3600, 10/3600, 10/3600], format="dia")		# 4*4, sparse
		Qk = diags(np.repeat(1, 12), format="dia")		# 12*12, sparse
		for i in range(1, number):
			r1, r2 = X[i-1][:3], X[i-1][6:9]
			Phi = nav.jacobian_double(STEP*(i-1), r1, r2)	# 12*12, sparse
			Zk = nav.measure_equation(i-1)	# (4, ), ndarray
			Hk = nav.jacobian_measure(r1, r2)	# 4*12, sparse
			Xk_pre = Phi * X[i-1]	# (12, )
			Pk_pre = Phi * P[i-1] * Phi.T  + Qk	# 12*12, sparse (half zero)
			print("Phi :", Phi, "\n\n", "HK :", Hk, "\n\n", "Pk_pre :", Pk_pre)
			Sk = (Hk*Pk_pre*Hk.T + Rk).toarray()	# 4*4, ndarray
			print("Sk :", Sk.shape, "\n", Sk)
			Kk = Pk_pre*Hk.T * np.linalg.inv( Sk )	# 12*4, ndarray
			Xk_upd = X[i-1] + Kk.dot( (Zk - Hk*Xk_pre) )
			print("Kk :", Kk, "\n\n", "Xk_upd: ", Xk_upd)
			Pk_upd = (I - Kk*Hk) * Pk_pre * (I - Kk*Hk).T + (Kk*Rk).dot(Kk.T)
			X.append(Xk_upd); P.append(Pk_upd)

		
	def plot_filter(self, X, number):
		r1, r2 = X[:number, :3], X[:number, 6:9]
		plt.figure(1)
		plt.plot(range(number), r1 - HPOP_1[:number, :3])
		plt.figure(2)
		plt.plot(range(number), r2 - HPOP_2[:number, :3])
		plt.show()
		
		
	def measure_stk(self, i):
		'''双星系统的测量方程的实际测量输出 Zk = h(Xk) + Vk, 由STK生成并加入噪声, ndarray'''
		delta_r = HPOP_1[i, :3] - HPOP_2[i, :3]
		r_norm = np.linalg.norm(delta_r, 2)
		v_rk = np.random.normal(loc=0, scale=1e-3)	# 测距噪声, 0均值, 测量标准差为1e-3 km
		v_dk = np.random.multivariate_normal(mean=[0,0,0], cov=10/3600*np.identity(3))	# 测角噪声, 0均值, 协方差为 10角秒*I
		Z = [ r_norm + v_rk];   Z.extend(delta_r/r_norm + v_dk)
		return np.array(Z)	# np.array, (4, )
		

	def complete_ukf(self, X, dt=STEP, Time=Time):
		'''UKF的状态方程, X_k+1 = f(x_k) + w_k, ndarray;	 dt=120s, 为UKF的predict步长'''
		X0, X1 = X[0:6], X[6:12]
		print(Time)
		y1 = solve_ivp( self.complete_dynamic, (Time, Time+dt), X0, method="RK45", rtol=1e-6, atol=1e-9, t_eval=[Time+dt] ).y
		y2 = solve_ivp( self.complete_dynamic, (Time, Time+dt), X1, method="RK45", rtol=1e-6, atol=1e-9, t_eval=[Time+dt] ).y
		y1L = [ x[0] for x in y1 ]; y2L = [ x[0] for x in y2 ]
		y1L.extend(y2L); ode_y = np.array(y1L)
		return ode_y

		
	def measure_equation(self, X):
		'''双星系统的测量方程, Z_k = h(X_k) + v_k, ndarray'''
		delta_r = X[0:3] - X[6:9]
		r_norm = np.linalg.norm(delta_r, 2)
		Z = [r_norm];   Z.extend(delta_r/r_norm)
		return np.array(Z)	# np.array, (4, )
		
	@fn_timer		
	def unscented_kf(self, number=240):
		X0 = np.hstack( (HPOP_1[0], HPOP_2[0]) )
		P0 = np.diag( np.repeat(0.1, 12) )
		Rk =  np.diag([1e-3, 10/3600, 10/3600, 10/3600])		# 4*4
		Qk = np.diag( np.repeat(0.1, 12) )		# 12*12
		points = MerweScaledSigmaPoints(n=12, alpha=0.1, beta=2, kappa=-9)
		ukf = UKF(dim_x=12, dim_z=4, fx=self.complete_ukf, hx=self.measure_equation, dt=STEP, points=points)
		ukf.x = X0
		ukf.P = P0
		ukf.R = Rk
		ukf.Q = Qk
		uxs = []
		global Time
		for i in range(number):	
			Z = self.measure_stk(i)
			ukf.predict()
			ukf.update(Z)
			uxs.append(ukf.x.copy())
			Time = Time+STEP
		uxs = np.array(uxs)
		return uxs

		
		
		
if __name__ == "__main__":
	import cProfile, pstats
	ob = Orbit()
	nav = Navigation()
	
	number = 720
	data = pd.read_csv("STK/Part_2/1_Inertial_HPOP_660.csv", nrows=number, usecols=range(1, 7))	# 取前number个点进行试算
	RV_array = data.values
	r_array = RV_array[:, :3]
	t_list = range(0, number*STEP, STEP)
	utc_array = (ob.generate_time(start_t="20180101", end_t="20180331"))[:number]
	utcJD_list = [ time_utc.to_julian_date() for time_utc in utc_array ]
	tdbJD_list = [ time_utc.to_julian_date() + 69.184/86400 for time_utc in utc_array ]
	I2F_list = [ ob.moon_Cbi(tdb_jd) for tdb_jd in tdbJD_list ]
	rFixed_list = [ np.dot(I2F, r_sat) for (I2F, r_sat) in zip(I2F_list, r_array) ]
	r_sat, r_fixed, RV, time_utc = r_array[0], rFixed_list[0], RV_array[0], utc_array[0]
	utc_jd, tdb_jd = time_utc.to_julian_date(), time_utc.to_julian_date() + 69.184/86400
	t, r1, r2 = 0, HPOP_1[0, :3], HPOP_2[0, :3]
	HPOP = np.hstack((HPOP_1[:number], HPOP_2[:number]))
	X = [ HPOP[0] ]
	t = 0
	
	