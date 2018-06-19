# -*- coding: utf-8 -*-

from scipy.sparse import coo_matrix, dia_matrix, bsr_matrix, block_diag, diags, identity
from orbit_km import *
from filterpy.kalman import unscented_transform, MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import ExtendedKalmanFilter as EKF


NUMBER = 720
Qk = np.diag([ 1e-14, 1e-14, 1e-14, 1e-14, 1e-14, 1e-14, 1e-14, 1e-14, 1e-14, 1e-14, 1e-14, 1e-14 ]) / 1e-1 # 12*12
Rk = np.power( np.diag( [1e-3, radians(10/3600), radians(10/3600), radians(10/3600)] ), 2 )	# 4*4, sigma_r*I
# # 基础卫星数据，轨道倾角分别为0和28.5°
# basic_i0A = ( pd.read_csv("STK/basic/Sat_1_i0.csv", nrows=NUMBER, usecols=range(1,7)) ).values
# basic_i0B = ( pd.read_csv("STK/basic/Sat_2_i0.csv", nrows=NUMBER, usecols=range(1,7)) ).values
# basic_i28A = ( pd.read_csv("STK/basic/Sat_1_i28.csv", nrows=NUMBER, usecols=range(1,7)) ).values
# basic_i28B = ( pd.read_csv("STK/basic/Sat_2_i28.csv", nrows=NUMBER, usecols=range(1,7)) ).values
# 原来的卫星数据，偏心率相差0.01的两颗卫星
old_A = ( pd.read_csv("STK/Part_2/1_Inertial_HPOP_660.csv", nrows=NUMBER, usecols=range(1,7)) ).values
old_B = ( pd.read_csv("STK/Part_2/2_Inertial_HPOP_660.csv", nrows=NUMBER, usecols=range(1,7)) ).values
# 大偏心率(0.6)的数据
# e06_A = ( pd.read_csv("STK/0.6e/Sat_1_e=0.6, i=28.5_120s.csv", nrows=NUMBER, usecols=range(1,7)) ).values
# e06_B = ( pd.read_csv("STK/0.6e/Sat_2_e=0.6, i=28.5_120s.csv", nrows=NUMBER, usecols=range(1,7)) ).values
# 赋值HPOP_1, HPOP_2
HPOP_1, HPOP_2 = old_A, old_B

import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['NSimSun', 'Times New Roman'] # 指定默认字体
mpl.rcParams['font.family']='sans-serif'
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
mpl.rcParams["figure.figsize"] = (3.2, 2.2); mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['legend.fontsize'] = 8; mpl.rcParams['axes.labelsize'] = 8;
mpl.rcParams['xtick.labelsize'] = 8; mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['savefig.dpi'] = 400; mpl.rcParams['figure.dpi'] = 400

class Navigation(Orbit):
	
	def __init__(self):
		return
		
	def __del__(self):
		return

	
	def jacobian_double(self, t, r1, r2):
		'''利用orbit.py中的函数, 进行两次数值积分计算双星系统的状态转移矩阵'''
		phi_1 = self.jacobian_single(t, r1)
		phi_2 = self.jacobian_single(t, r1)
		PHI = block_diag((phi_1, phi_2), format="bsr")
		return PHI


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
		P0 = np.diag([ 3e-2, 3e-2, 3e-2, 3e-6, 3e-6, 3e-6, 3e-2, 3e-2, 3e-2, 3e-6, 3e-6, 3e-6 ])
		error = np.random.multivariate_normal(mean=np.zeros(12), cov=P0)
		X0 = np.hstack( (HPOP_1[0], HPOP_2[0]) ) + error
		
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
		'''UKF的状态方程, X_k+1 = f(x_k) + w_k, ndarray;	 dt=STEPs, 为UKF的predict步长;	 计算时间0.20s'''
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
		
	
	def unscented_kf(self, number=NUMBER):
		global Time
		P0 = np.diag([ 3e-2, 3e-2, 3e-2, 3e-6, 3e-6, 3e-6, 3e-2, 3e-2, 3e-2, 3e-6, 3e-6, 3e-6 ])
		error = np.random.multivariate_normal(mean=np.zeros(12), cov=P0)
		X0 = np.hstack( (HPOP_1[0], HPOP_2[0]) ) + error
		points = MerweScaledSigmaPoints(n=12, alpha=0.01, beta=0.5, kappa=-9)
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
		delta_x1 = X1[:number] - HPOP_1[:number]
		delta_x2 = X2[:number] - HPOP_2[:number]
		delta_norm_r1 = np.array([ np.linalg.norm(X, 2) for X in delta_x1[:, 0:3] ])
		delta_norm_r2 = np.array([ np.linalg.norm(X, 2) for X in delta_x2[:, 0:3] ])
		time_range = np.arange(0, number*STEP/3600, STEP/3600)
		up_error = 100 + 1e3 * np.power(e, -time_range)
		low_error = -100 - 1e3 * np.power(e, -time_range)
		v_upError = 0.02 + np.power(e, -time_range)
		v_lowError = -0.02 - np.power(e, -time_range)
		plt.figure(1)
		plt.xlabel("时间 / $\mathrm{h}$"); plt.ylabel("位置误差 / ($\mathrm{m}$)")
		plt.plot(time_range, delta_x1[:, 0] * 1000, "r-", label="x")
		plt.plot(time_range, delta_x1[:, 1] * 1000, "b--", label="y")
		plt.plot(time_range, delta_x1[:, 2] * 1000, "g-.", label="z")
		plt.plot(time_range, up_error, "k:")
		plt.plot(time_range, low_error, "k:", label=" $\mathrm{\pm 100 m}$")
		plt.legend(); plt.tight_layout(); plt.savefig("Figure/FourChapter/e=0.6, i=28.5/A星位置收敛曲线.png", bbox_inches='tight')
		
		plt.figure(2)
		plt.xlabel("时间 / $\mathrm{h}$"); plt.ylabel("速度误差 / ($\mathrm{m/s}$)")
		plt.plot(time_range, delta_x1[:, 3] * 1000, "r-", label="x")
		plt.plot(time_range, delta_x1[:, 4] * 1000, "b--", label="y")
		plt.plot(time_range, delta_x1[:, 5] * 1000, "g-.", label="z")
		plt.plot(time_range, v_upError, "k:")
		plt.plot(time_range, v_lowError, "k:", label=" $\mathrm{\pm 0.02 m/s}$")
		plt.legend(); plt.tight_layout(); plt.savefig("Figure/FourChapter/e=0.6, i=28.5/A星速度收敛曲线.png", bbox_inches='tight')

		plt.figure(3)
		plt.xlabel("时间 / $\mathrm{h}$"); plt.ylabel("位置误差 / ($\mathrm{m}$)")
		plt.plot(time_range, delta_x2[:, 0] * 1000, "r-", label="x")
		plt.plot(time_range, delta_x2[:, 1] * 1000, "b--", label="y")
		plt.plot(time_range, delta_x2[:, 2] * 1000, "g-.", label="z")
		plt.plot(time_range, up_error, "k:")
		plt.plot(time_range, low_error, "k:", label=" $\mathrm{\pm 100 m}$")
		plt.legend(); plt.tight_layout(); plt.savefig("Figure/FourChapter/e=0.6, i=28.5/B星位置收敛曲线.png", bbox_inches='tight')
		
		plt.figure(4)
		plt.xlabel("时间 / $\mathrm{h}$"); plt.ylabel("速度误差 / ($\mathrm{m/s}$)")
		plt.plot(time_range, delta_x2[:, 3] * 1000, "r-", label="x")
		plt.plot(time_range, delta_x2[:, 4] * 1000, "b--", label="y")
		plt.plot(time_range, delta_x2[:, 5] * 1000, "g-.", label="z")
		plt.plot(time_range, v_upError, "k:")
		plt.plot(time_range, v_lowError, "k:", label=" $\mathrm{\pm 0.02 m/s}$")
		plt.legend(); plt.tight_layout(); plt.savefig("Figure/FourChapter/e=0.6, i=28.5/B星速度收敛曲线.png", bbox_inches='tight')
		
		abs_error = 100 + 1e3 * np.power(e, -time_range)
		plt.figure(5)
		plt.xlabel("时间 / $\mathrm{h}$"); plt.ylabel("位置误差 / ($\mathrm{m}$)")
		plt.plot(time_range, delta_norm_r1 * 1000, "r-", label="A星位置误差")
		plt.plot(time_range, abs_error, "k:", label="+100 m")
		plt.legend(); plt.tight_layout(); plt.savefig("Figure/FourChapter/e=0.6, i=28.5/A星绝对误差曲线.png", bbox_inches='tight')
		
		plt.figure(6)
		plt.xlabel("时间 / $\mathrm{h}$"); plt.ylabel("位置误差 / ($\mathrm{m}$)")
		plt.plot(time_range, delta_norm_r2 * 1000, "b--", label="B星位置误差")
		plt.plot(time_range, abs_error, "k:", label="+100 m")
		plt.legend(); plt.tight_layout(); plt.savefig("Figure/FourChapter/e=0.6, i=28.5/B星绝对误差曲线.png", bbox_inches='tight')

		plt.show()
		
		
if __name__ == "__main__":
	
	ob = Orbit()
	nav = Navigation()
	
	number = NUMBER-1
	X0 = np.array([ 1.84032000e+03,  0.00000000e+00,  0.00000000e+00, -0.00000000e+00, 1.57132000e+00,  8.53157000e-01])
	A, B = old_A, old_B
	for i in range(4):
		phi_1 = nav.jacobian_twice(0, A[0, 0:3], B[0, 0:3]).toarray()
		phi_2 = nav.jacobian_double(0, A[0, 0:3], B[0, 0:3])
		print(phi_1[:2], "\n\n", phi_2[:2], "\n\n")
	
	
	
	# # # X = nav.extend_kf(number)
	# X = nav.unscented_kf(number)
	# np.save("npy/Qk=1e-12.npy", X)
	# # X = np.load("npy/Qk=1e-12.npy")
	# nav.plot_filter(X, number)
	