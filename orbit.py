# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from math import *
from operator import mod
from datetime import datetime, timedelta
import de421
from jplephem import Ephemeris
from jplephem.spk import SPK
from scipy.integrate import solve_ivp, RK45, BDF, LSODA


# 恒定常值变量区
MIU_E, MIU_M, MIU_S = 398600.4415, 4902.80030555540, 132712422595.6590	# 引力系数，单位 km^3 / s^2
RE, RM, RS = 6378.1363, 1738.000, 69550.0		#天体半径，单位 km
CSD_MAX, CSD_EPS, number, CSD_LM = 1e+30, 1e-10, 495, 30
STEP, Beta = 120, 0	#全局积分步长, 太阳光压参数
C, S = np.load("Clm.npy"), np.load("Slm.npy")	# 序列化后代码速度提高4倍


# 计算常值变量与函数区, 加速运算过程
kernel = SPK.open(r"..\de421\de421.bsp")
eph = Ephemeris(de421)
time_utc = pd.Timestamp("2018-1-1 00:00:00", freq="1S")
B = []
for i in range(CSD_LM+1):
	A = [ sqrt( (2*i+1)*(2*i-1) / ((i+j)*(i-j)) ) for j in range(i) ]
	A.append(1); B.append(np.array(A))
B = np.array(B)	# B用于完全规格化勒让德函数计算
pi_dot = np.array([ np.array( [sqrt((i+1)*i/2)] + [ sqrt( (i+j+1)*(i-j) ) for j in range(1, i+1)] ) for i in range(CSD_LM+1) ])
M_Array = np.array([ j for j in range(0, CSD_LM+1) ])
Rx_bas = lambda x: np.array([[1, 0, 0], [0, cos(x), sin(x)], [0, -sin(x), cos(x)]])
Ry_bas = lambda y: np.array([[cos(y), 0, -sin(y)], [0, 1, 0], [sin(y),0, cos(x)]])
Rz_bas = lambda z: np.array([[cos(z), sin(z), 0], [-sin(z), cos(z), 0], [0, 0, 1]])
LCS2I = np.dot( Rx_bas(radians(24.358897)), Rz_bas(radians(-3.14227)) )	# 月心天球 -> 月心惯性系



import timeit
from functools import wraps
def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = timeit.default_timer()
        result = function(*args, **kwargs)
        t1 = timeit.default_timer()
        print("%s running: %s seconds" %(function.__name__, str(t1-t0)) )
        return result
    return function_timer
	


class Orbit:
	r, v = [], []
	sixEle = []
	
	def __init__(self, rv=None, sixEle=None, Time=0):
		if rv != None:
			self.r = rv[:3]
			self.v = rv[3:]
		else:
			self.sixEle = sixEle
		self.Time = Time
			
	
	def __del__(self):
		return
		
		
	def readCoffients(self, number=496, n=30):
		'''获取中心天体球谐引力系数，默认前30阶，包括0阶项, np.array'''
		df = pd.read_csv("STK/GL0660B.grv", sep="\s+", header=None, nrows=number)
		f = df[:number]
		f.columns = ["l", "m", "Clm", "Slm"]
		f = f.set_index([f["l"], f["m"]]); del f['l'], f['m']
		Clm, Slm = f["Clm"], f["Slm"]
		Clm = [ np.array(Clm.loc[i]) for i in range(0, n+1) ]
		Slm = [ np.array(Slm.loc[i]) for i in range(0, n+1) ]
		np.save("Clm.npy", Clm); np.save("Slm.npy", Slm)	
		
		
	def generate_time(self, start_t="20171231", end_t="20180101"):
		'''产生Timestamp列表，用于计算对应的儒略日, multi-time
		输入：起始和终止时间		str类型
		输出：返回产生的utc时间列表	Timestamp构成的list'''
		start_t = datetime.strptime(start_t, "%Y%m%d")
		end_t = datetime.strptime(end_t, "%Y%m%d")
		time_list = pd.date_range(start_t, end_t, freq=str(STEP)+"S")	#固定步长120s
		return time_list
		
		
	def moon2Earth(self, tdb_jd):
		'''计算地球相对于月球的位置矢量在地心J2000系下的表示, single-time
		输入：TDB的儒略日时间										Timestamp			
		输出：返回地球相对于月球的位置矢量(地球J2000下), 单位 km		np.array'''
		M2E_bc = -kernel[3, 301].compute(tdb_jd)	# 月球 -> 地月中心
		E_bc2E = kernel[3, 399].compute(tdb_jd)	# 地月中心 -> 地球
		return M2E_bc + E_bc2E	# 月球 -> 地球
		
		
	def moon2Sun(self, tdb_jd):		
		'''计算太阳相对于月球的位置矢量在地心J2000系下的表示, single-time
		输入：TDB的儒略日时间										Timestamp
		输出：返回太阳相对于月球的位置矢量(地球J2000下), 单位 km,		np.array'''
		E_bc2M = kernel[3, 301].compute(tdb_jd)	# 地月中心 -> 月球
		S_bc2E_bc = kernel[0, 3].compute(tdb_jd)	# 太阳系中心 -> 地月中心
		S2S_bc = -kernel[0, 10].compute(tdb_jd)	# 太阳 -> 太阳系中心
		return -(S2S_bc + S_bc2E_bc + E_bc2M)
		
		
	def moon_Cbi(self, tdb_jd):
		'''返回月惯系到月固系的方向余弦矩阵, single-time，张智飞, STK Help; 高玉东 - 月球探测器'''
		phi, theta, psi = eph.position("librations", tdb_jd)	# 物理天平动的三个欧拉角, np.array([0])
		CS2F = np.dot( np.dot(Rz_bas(psi), Rx_bas(theta)), Rz_bas(phi) )	# 月心天球 -> 月固系
		return np.dot(CS2F, LCS2I.T)	# 月惯系 -> 月固系, F = CS2F * I2CS * I
	

	def legendre_spher_alfs(self, phi, lm=CSD_LM):
		'''计算完全规格化缔合勒让德函数，球坐标形式，Brandon A. Jones - Efficient Models for the Evaluation and Estimation(eqs. 2.8)
		输入：地心纬度, rad;		阶次lm
		输出：可用于向量直接计算的勒让德函数，包含P_00项	np.array, B提前计算并存储以加快运算速度'''
		sin_phi, cos_phi = sin(phi), cos(phi)	# 提前计算提升效率
		P = [ np.array([1, 0]), np.array([ sqrt(3)*sin_phi, sqrt(3)*cos_phi, 0 ]) ]
		for i in range(2, lm+1):
			P_ij = [ B[i][j] * sin_phi * P[i-1][j] - B[i][j]/B[i-1][j] * P[i-2][j] for j in range(i) ]
			P_ii = sqrt((2*i+1)/(2*i)) * cos_phi * P[i-1][i-1]
			P_ij.extend([P_ii, 0])
			P.append(np.array(P_ij))
		return np.array(P)
		
		
	def legendre_diff(self, phi, P, lm=CSD_LM):
		'''计算完全规格化缔合勒让德函数的一阶导数，球坐标形式，Brandon A. Jones(eqs. 2.14)
		输入：地心纬度, rad;	完全规格化勒让德函数P;		阶次lm'''
		tan_phi = tan(phi)
		dP = np.array([ P[i][1:] * pi_dot[i] - M_Array[:i+1]*tan_phi * P[i][:-1] for i in range(0, lm+1) ])
		return dP
		
		
	def centreGravity(self, r_sat, miu=MIU_M):
		'''计算中心天体的引力加速度，single-time，输入 km, 输出 m/s^2
		输入：惯性系下卫星位置矢量r_sat，np.array, 单位 km;		miu默认为月球;
		输出：返回中心天体的引力加速度, np.array，单位 km/s^2'''
		r_norm = np.linalg.norm(r_sat, 2)
		g0 = -miu*r_sat / r_norm**3		# 中心引力 3*1
		return g0	# km/s^2


	def nonspher_moon(self, r_sat, tdb_jd, miu=MIU_M, Re=RM, lm=CSD_LM):
		'''计算月球的非球形引力加速度，single-time, Brandon A. Jones - Efficient Models for the Evaluation and Estimation(eq. 2.14)'''
		I2F = self.moon_Cbi(tdb_jd)	# 月惯系到月固系的方向余弦矩阵 3*3
		r_fixed = np.dot(I2F, r_sat)		# 应该在固连系下建立，王正涛
		r_norm = np.linalg.norm(r_fixed, 2)
		xy_norm = np.linalg.norm(r_fixed[:2], 2)
		phi, lamda = asin(r_fixed[2] / r_norm), atan2(r_fixed[1], r_fixed[0])	# Keric A. Hill-Autonomous Navigation in Libration Point Orbits
		tan_phi = r_fixed[2]/xy_norm
		P = self.legendre_spher_alfs(phi, lm)	# 勒让德函数
		rRatio = Re/r_norm
		# 球坐标对直角坐标的偏导数, Keric A. Hill(eq. 8.10), Brandon A. Jones(eq. 2.13)
		dR_dr = r_fixed / r_norm
		dphi_dr = 1/xy_norm * np.array([ -r_fixed[0]*r_fixed[2] / pow(r_norm, 2), -r_fixed[1]*r_fixed[2] / pow(r_norm, 2), \
									1 - pow(r_fixed[2], 2) / pow(r_norm, 2) ])
		dlamda_dr = 1/pow(xy_norm, 2) * np.array([ -r_fixed[1], r_fixed[0], 0 ])
		cos_m = np.array([ cos(j*lamda) for j in range(0, lm+1) ])
		sin_m = np.array([ sin(j*lamda) for j in range(0, lm+1) ])
		dU_dr = -miu/pow(r_norm, 2) * ( sum([ pow(rRatio, i)*(i+1) * np.dot(P[i][:-1], C[i]*cos_m[:i+1] + S[i]*sin_m[:i+1]) for i in range(2, lm+1) ]) )
		dU_dphi = miu/r_norm * sum([ pow(rRatio, i) * np.dot(P[i][1:] * pi_dot[i] -  tan_phi*M_Array[:i+1] * P[i][:-1], \
							C[i]*cos_m[:i+1] + S[i]*sin_m[:i+1]) for i in range(2, lm+1) ])
		dU_dlamda = miu/r_norm * sum( [ pow(rRatio, i) * np.dot(M_Array[:i+1]*P[i][:-1], S[i]*cos_m[:i+1] - C[i]*sin_m[:i+1]) for i in range(2, lm+1) ] )
		a_fixed = dR_dr * dU_dr + dphi_dr * dU_dphi + dlamda_dr * dU_dlamda
		a_inertial = np.dot(I2F.T, a_fixed)
		return a_inertial	# km/s^2

		
	def thirdSun(self, r_sat, tdb_jd):
		'''计算太阳对卫星产生的第三天体摄动加速度, single-time, 输入km, 输出m/s^2
		输入：UTC时间time_utc，卫星相对中心天体的矢量，km, 均为np.array；
		输出：返回第三体太阳摄动加速度, m/s^2, np.array'''
		r_sun = np.dot( LCS2I, self.moon2Sun(tdb_jd) )	# 转换为月心惯性系下的位置矢量
		norm_st = np.linalg.norm(r_sat-r_sun, 2)	# 太阳 -> 卫星
		norm_et = np.linalg.norm(r_sun, 2)	# 月球 -> 太阳
		a_sun = -MIU_S * ( (r_sat-r_sun)/pow(norm_st, 3) + r_sun/pow(norm_et, 3) )
		return a_sun	# km/s^2
		
		
	def thirdEarth(self, r_sat, tdb_jd):
		'''计算地球对卫星产生的第三天体摄动加速度, single-time, 输入km, 输出m/s^2
		输入：UTC时间time_utc，卫星相对中心天体的矢量，m, 均为np.array;
		输出：返回第三体地球摄动加速度, km/s^2, np.array'''
		r_earth = np.dot( LCS2I, self.moon2Earth(tdb_jd) )	# 转换为月心惯性系下的位置矢量
		norm_st = np.linalg.norm(r_sat-r_earth, 2)	# l(vector), 地球 -> 卫星
		norm_et = np.linalg.norm(r_earth, 2)	# 月球 -> 地球
		a_earth = -MIU_E * ( (r_sat-r_earth)/pow(norm_st, 3) + r_earth/pow(norm_et, 3) )
		return a_earth	# km/s^2
			

	def solarPress(self, r_sat, tdb_jd):
		'''计算太阳光压摄动, single-time, 输入 km, 输出 m/s^2
		输入：卫星位置矢量，m, np.array;	儒略日时间(float);
		输出：返回太阳光压在r_sat处造成的摄动, m / s^2, np.array'''
		# 计算太阳光压摄动中的系数，使用随机漫步过程模拟
		v = np.random.normal(0, 1)	# 0均值，协方差为1的高斯白噪声序列
		sigma = 9.18e-5	# 给定的仿真参数
		global Beta
		Beta = Beta + sigma*sqrt(STEP)*v
		r_sun = self.moon2Sun(tdb_jd)		#月球->太阳矢量, km
		ro = 4.5605e-6	#太阳常数
		delta_0 = 1.495978707e+8	#日地距离 km
		delta = r_sat - r_sun
		norm_delta = np.linalg.norm(delta, 2)
		F = Beta * ro*(delta_0**2/norm_delta**2) * delta/norm_delta
		return F/1000	# km/s^2
			
	
	def twobody_dynamic(self, t, RV,  miu=MIU_M):
		'''二体动力学方程，仅考虑中心引力u
		输入：	时间t，从0开始（好像不太重要）; 	惯性系下位置速度	km, km/s, np.array
		输出：	返回d(RV)/dt，动力学增量, m/s, m/s^2, np.array'''
		R_Norm = np.linalg.norm(RV[:3], 2)
		dr_dv = []
		dr_dv.extend(RV[3:])
		dr_dv.extend(-miu*(RV[:3]) / pow(R_Norm, 3))
		return np.array(dr_dv)
		
	
	def complete_dynamic(self, t, RV, miu=MIU_M, Re=RM, lm=CSD_LM):
		'''系统完整动力学模型, 暂考虑中心引力，非球形引力，第三天体引力
		输入：	时间t，从0开始（好像不太重要）; 	惯性系下位置速度	km, km/s, np.array
		输出：	返回d(RV)/dt，动力学增量, np.array, 1*6, m/s, m/s^2'''
		tdb_jd = (time_utc+int(t)).to_julian_date() + 69.184/86400
		R, V = RV[:3], (RV[3:]).tolist()	# km, km/s
		F0 = self.centreGravity(R, miu)
		F1 = self.nonspher_moon(R, tdb_jd, miu, Re, lm)
		F2 = self.thirdEarth(R, tdb_jd)
		F3 = self.thirdSun(R, tdb_jd)
		F4 = self.solarPress(R, tdb_jd)	# 加入太阳光压摄动
		F = F0 + F1 + F2 + F3 + F4	# km/s^2
		V.extend(F)
		return np.array(V)
		
	@fn_timer	
	def integrate_orbit(self, rv_0, num):
		'''数值积分器，使用RK45获得卫星递推轨道'''
		ode_y = solve_ivp( self.complete_dynamic, (0,STEP*num), rv_0, method="RK45", rtol=1e-10, t_eval=range(0,STEP*num, STEP) ).y
		return ode_y
	

	def partial_centre(self, r_sat, miu=MIU_M):
		'''中心引力加速度对 (x, y, z) 的偏导数在月惯系下的表达'''
		r_norm = np.linalg.norm(r_sat, 2)
		gradient = -miu * (3*np.outer(r_sat, r_sat) - np.identity(3)) / pow(r_norm, 5)
		return gradient
		
		
	def partial_nonspher(self, r_sat, tdb_jd, miu=MIU_M, Re=RM, lm=CSD_LM):
		'''非球形引力摄动加速度 对 (x, y, z) 的偏导数, 在月心惯性系下的表达(转换已完成), Keric A. Hill(eq. 8.14)
		输入: 月心惯性系下的位置矢量; 	动力学时对应的儒略时间'''
		I2F = self.moon_Cbi(tdb_jd)	# 月惯系到月固系的方向余弦矩阵 3*3
		r_fixed = np.dot(I2F, r_sat)		# 应该在固连系下建立，王正涛
		x, y, z, r_norm = r_fixed[0], r_fixed[1], r_fixed[2], np.linalg.norm(r_fixed, 2)
		xy_norm, pow_r2, pow_r3 = np.linalg.norm(r_fixed[:2], 2), pow(r_norm, 2), pow(r_norm, 3)
		phi, lamda = asin(z / r_norm), atan2(y, x)	# Keric A. Hill-Autonomous Navigation in Libration Point Orbits
		cos_phi, tan_phi = xy_norm / r_norm, z / xy_norm
		P = self.legendre_spher_alfs(phi, lm)	# 勒让德函数
		dP = self.legendre_diff(phi, P, lm)
		rRatio, lm_range = Re/r_norm, range(2, lm+1)
		cos_m = np.array([ cos(j*lamda) for j in range(0, lm+1) ])
		sin_m = np.array([ sin(j*lamda) for j in range(0, lm+1) ])
		# 计算a对r(vector)的偏导需要用到 球坐标对直角坐标的偏导数(坐标变换), 应为(1*3)矩阵
		dR_dr = np.array([ r_fixed / r_norm ]) 	# (1*3)
		dphi_dr = 1/xy_norm * np.array([ [-x*z / pow_r2, -y*z/pow_r2, 1 - pow(z,2)/pow_r2] ])		# (1*3)
		dlamda_dr = 1/pow(xy_norm, 2) * np.array([ [-y, x, 0] ])	# (1*3)
		# U对r(vector) 的一阶偏导数, 均为 const
		dU_dR = -miu/pow_r2 * ( sum([ pow(rRatio, i)*(i+1) * np.dot(P[i][:-1], C[i]*cos_m[:i+1] + S[i]*sin_m[:i+1]) for i in lm_range ]) )
		dU_dphi = miu/r_norm * sum([ pow(rRatio, i) * np.dot(P[i][1:] * pi_dot[i] -  tan_phi*M_Array[:i+1] * P[i][:-1], \
							C[i]*cos_m[:i+1] + S[i]*sin_m[:i+1]) for i in lm_range ])
		dU_dlamda = miu/r_norm * sum( [ pow(rRatio, i) * np.dot(M_Array[:i+1]*P[i][:-1], S[i]*cos_m[:i+1] - C[i]*sin_m[:i+1]) for i in lm_range ] )
		# U对R(scalar), phi, lamda 的二阶偏导数计算, Hill与王庆宾一致, 均为 const
		dR_2 = miu/pow_r3 * sum([ pow(rRatio, i)*(i+2)*(i+1) * np.dot(P[i][:-1], C[i]*cos_m[:i+1] + S[i]*sin_m[:i+1]) for i in lm_range ])
		dR_dphi = -miu/pow_r2 * sum([ pow(rRatio, i)*(i+1) * np.dot(dP[i], C[i]*cos_m[:i+1] + S[i]*sin_m[:i+1]) for i in lm_range ])
		dR_dlamda = -miu/pow_r2 * sum([ pow(rRatio, i)*(i+1) * np.dot(M_Array[:i+1]*P[i][:-1], \
					S[i]*cos_m[:i+1] - C[i]*sin_m[:i+1]) for i in lm_range ])
		dphi_2 = miu/r_norm * sum([ pow(rRatio, i) * np.dot( (np.power(M_Array[:i+1]/cos_phi, 2) - i*(i+1)) * P[i][:-1] \
					+ tan_phi * dP[i], C[i]*cos_m[:i+1] + S[i]*sin_m[:i+1]) for i in lm_range ])
		dphi_dlamda = miu/r_norm * sum([ pow(rRatio, i) * np.dot(M_Array[:i+1]*dP[i], S[i]*cos_m[:i+1] - C[i]*sin_m[:i+1]) for i in lm_range ])
		dlamda_2 = -miu/r_norm * sum([ pow(rRatio, i) * np.dot( np.power(M_Array[:i+1], 2) * P[i][:-1], \
					C[i]*cos_m[:i+1] + S[i]*sin_m[:i+1]) for i in lm_range ])
		# U先对r(vector), 再对R(scalar), phi, lamda的二阶偏导数		
		dr_dr = np.array([ dR_dr[0], dphi_dr[0], dlamda_dr[0] ])	# 3*3 Matrix
		dr_dR = np.dot( np.array([ [dR_2, dR_dphi, dR_dlamda] ]), dr_dr )	# (1*3) * (3*3)
		dr_dphi = np.dot( np.array([ [dR_dphi, dphi_2, dphi_dlamda] ]), dr_dr )	# (1*3) * (3*3)
		dr_dlamda = np.dot( np.array([ [dR_dlamda, dphi_dlamda, dlamda_2] ]), dr_dr )	# (1*3) * (3*3)
		# 后三项的计算, 包括R(scalar), phi, lamda对r(vector)的二阶偏导数
		d2R_dr2 = np.array([ [ 1/r_norm - pow(x,2)/pow_r3,		-x*y/pow_r3,					-x*z/pow_r3 ],
							 [ -y*x/pow_r3,						1/r_norm - pow(y,2)/pow_r3, 	- y*z/pow_r3 ],
							 [ -z*x/pow_r3,  					-z*y/pow_r3,  					1/r_norm - pow(z,2)/pow_r3] ]) 
		d2phi_dr2 = 1/pow(xy_norm, 3) * np.array([ [ z*pow(x,2)/pow_r2,  		z*x*y/pow_r2,  			0 ], 
												   [ z*x*y/pow_r2,  			z*pow(y,2)/pow_r2,  	0 ],
												   [ pow(z,2)*x/pow_r2 - x,  	pow(z,2)*y/pow_r2 - y, 	0 ] ]) \
				+ 1/(pow_r2*xy_norm) * np.array([ [ 2*z*pow(x,2)/pow_r2 - z,  	2*x*y*z/pow_r2,  			2*x*pow(z,2)/pow_r2 - x ],
												  [ 2*x*y*z/pow_r2,  			2*z*pow(y,2)/pow_r2 - z,  	2*y*pow(z,2)/pow_r2 - y ],
												  [ 2*pow(z,2)*x/pow_r2, 		2*pow(z,2)*y/pow_r2, 		2*pow(z,3)/pow_r2 - 2*z ] ])
													  
		d2lamda_dr2 = 1/pow(xy_norm, 4) * np.array([ [ 2*x*y,				pow(y,2)-pow(x,2), 	0 ],
													 [ pow(y,2)-pow(x,2), 	-2*x*y, 			0 ],
													 [ 0, 					0, 					0 ] ])
		# 3 * [ (3*1) * (1*3) ] + 3 * [ (3*3) * const]
		da_dr = np.dot(dR_dr.T, dr_dR) + np.dot(dphi_dr.T, dr_dphi) + np.dot(dlamda_dr.T, dr_dlamda) \
				+ d2R_dr2 * dU_dR + d2phi_dr2 * dU_dphi + d2lamda_dr2 * dU_dlamda
		da_dr = np.dot( np.dot(I2F.T, da_dr) ,  I2F ) 	# Hill(eq 8.28) and 苏勇-利用GOCE和GRACE卫星(eq 2-67)
		return da_dr
		
		
	def partial_nonspher_wang(self, r_sat, tdb_jd, miu=MIU_M, Re=RM, lm=CSD_LM):
		'''非球形引力摄动加速度 对 (x, y, z) 的偏导数, 在月心惯性系下的表达(转换已完成), Keric A. Hill(eq. 8.14)
		输入: 月心惯性系下的位置矢量; 	动力学时对应的儒略时间'''
		I2F = self.moon_Cbi(tdb_jd)	# 月惯系到月固系的方向余弦矩阵 3*3
		r_fixed = np.dot(I2F, r_sat)		# 应该在固连系下建立，王正涛
		x, y, z = r_fixed
		r_norm = np.linalg.norm(r_fixed, 2)
		xy_norm, pow_r2, pow_r3 = np.linalg.norm(r_fixed[:2], 2), pow(r_norm, 2), pow(r_norm, 3)
		phi, lamda = asin(z / r_norm), atan2(y, x)	# Keric A. Hill-Autonomous Navigation in Libration Point Orbits
		cos_phi, tan_phi = xy_norm / r_norm, z / xy_norm
		P = self.legendre_spher_alfs(phi, lm)	# 勒让德函数
		dP = self.legendre_diff(phi, P, lm)
		rRatio, lm_range = Re/r_norm, range(2, lm+1)
		cos_m = np.array([ cos(j*lamda) for j in range(0, lm+1) ])
		sin_m = np.array([ sin(j*lamda) for j in range(0, lm+1) ])
		B = np.array([ [cos(phi)*cos(lamda), 				cos(phi)*sin(lamda), 				sin(phi)], \
					[(-1/r_norm)*sin(phi)*cos(lamda), 		(-1/r_norm)*sin(phi)*sin(lamda),  	(1/r_norm)*cos(phi)], \
					[(-1/r_norm)*sin(lamda)/cos(phi), 		(1/r_norm)*cos(lamda)/cos(phi), 	0] ])	#球坐标到直角坐标 3*3
		dB_dr = np.array([ [0, 											0, 										0],
							[sin(phi)*cos(lamda)/pow(r_norm, 2), 		sin(phi)*sin(lamda)/pow(r_norm, 2), 	-cos(phi)/pow(r_norm, 2)],
							[sin(lamda)/(pow(r_norm, 2)*cos(phi)), 		-cos(lamda)/(pow(r_norm, 2)*cos(phi)), 	0] ])				
		dB_dphi = np.array([ [-sin(phi)*cos(lamda), 							-sin(phi)*sin(lamda), 							cos(phi)],
							 [-cos(phi)*cos(lamda)/r_norm, 						-cos(phi)*sin(lamda)/r_norm, 					-sin(phi)/r_norm],
							 [-sin(phi)*sin(lamda)/(r_norm*pow(cos(phi), 2)), 	sin(phi)*cos(lamda)/(r_norm*pow(cos(phi), 2)), 0] ])
		dB_dlamda = np.array([ [-cos(phi)*sin(lamda), 				cos(phi)*cos(lamda), 			0],
								[sin(phi)*sin(lamda)/r_norm, 		-sin(phi)*cos(lamda)/r_norm, 	0],
								[-cos(lamda)/(r_norm*cos(phi)),		-sin(lamda)/(r_norm*cos(phi)), 	0] ])
		# 计算a对r(vector)的偏导需要用到 U对r(vector) 的一阶偏导数
		dU_dR = -miu/pow_r2 * ( sum([ pow(rRatio, i)*(i+1) * np.dot(P[i][:-1], C[i]*cos_m[:i+1] + S[i]*sin_m[:i+1]) for i in lm_range ]) )
		dU_dphi = miu/r_norm * sum([ pow(rRatio, i) * np.dot(P[i][1:] * pi_dot[i] -  tan_phi*M_Array[:i+1] * P[i][:-1], \
							C[i]*cos_m[:i+1] + S[i]*sin_m[:i+1]) for i in lm_range ])
		dU_dlamda = miu/r_norm * sum( [ pow(rRatio, i) * np.dot(M_Array[:i+1]*P[i][:-1], S[i]*cos_m[:i+1] - C[i]*sin_m[:i+1]) for i in lm_range ] )
		# U对R(scalar), phi, lamda 的二阶偏导数计算, Hill与王庆宾一致
		dR_2 = miu/pow_r3 * sum([ pow(rRatio, i)*(i+2)*(i+1) * np.dot(P[i][:-1], C[i]*cos_m[:i+1] + S[i]*sin_m[:i+1]) for i in lm_range ])
		dR_dphi = -miu/pow(r_norm, 2) * sum([ pow(rRatio, i)*(i+1) * np.dot(dP[i], C[i]*cos_m[:i+1] + S[i]*sin_m[:i+1]) \
					for i in lm_range ])
		dR_dlamda = -miu/pow(r_norm, 2) * sum([ pow(rRatio, i)*(i+1) * np.dot(M_Array[:i+1]*P[i][:-1], \
					S[i]*cos_m[:i+1] - C[i]*sin_m[:i+1]) for i in lm_range ])
		dphi_2 = miu/r_norm * sum([ pow(rRatio, i) * np.dot( (np.power(M_Array[:i+1]/cos_phi, 2) - i*(i+1)) * P[i][:-1] \
					+ tan_phi * dP[i], C[i]*cos_m[:i+1] + S[i]*sin_m[:i+1]) for i in lm_range ])
		dphi_dlamda = miu/r_norm * sum([ pow(rRatio, i) * np.dot(M_Array[:i+1]*dP[i], S[i]*cos_m[:i+1] - C[i]*sin_m[:i+1]) \
					for i in lm_range ])
		dlamda_2 = -miu/r_norm * sum([ pow(rRatio, i) * np.dot( np.power(M_Array[:i+1], 2) * P[i][:-1], \
					C[i]*cos_m[:i+1] + S[i]*sin_m[:i+1]) for i in lm_range ])
		# 
		df1_dr = np.array([ dB_dr[0][0]*dU_dR + dB_dr[1][0]*dU_dphi + dB_dr[2][0]*dU_dlamda + B[0][0]*dR_2 + B[1][0]*dR_dphi + B[2][0]*dR_dlamda, 
					dB_dphi[0][0]*dU_dR + dB_dphi[1][0]*dU_dphi + dB_dphi[2][0]*dU_dlamda + B[0][0]*dR_dphi + B[1][0]*dphi_2 + B[2][0]*dphi_dlamda,
					dB_dlamda[0][0]*dU_dR + dB_dlamda[1][0]*dU_dphi + dB_dlamda[2][0]*dU_dlamda + B[0][0]*dR_dlamda + B[1][0]*dphi_dlamda + B[2][0]*dlamda_2 ])
		df2_dr = np.array([ dB_dr[0][1]*dU_dR + dB_dr[1][1]*dU_dphi + dB_dr[2][1]*dU_dlamda + B[0][1]*dR_2 + B[1][1]*dR_dphi + B[2][1]*dR_dlamda, 
					dB_dphi[0][1]*dU_dR + dB_dphi[1][1]*dU_dphi + dB_dphi[2][1]*dU_dlamda + B[0][1]*dR_dphi + B[1][1]*dphi_2 + B[2][1]*dphi_dlamda,
					dB_dlamda[0][1]*dU_dR + dB_dlamda[1][1]*dU_dphi + dB_dlamda[2][1]*dU_dlamda + B[0][1]*dR_dlamda + B[1][1]*dphi_dlamda + B[2][1]*dlamda_2 ])
		df3_dr = np.array([ dB_dr[0][2]*dU_dR + dB_dr[1][2]*dU_dphi + dB_dr[2][2]*dU_dlamda + B[0][2]*dR_2 + B[1][2]*dR_dphi + B[2][2]*dR_dlamda, 
					dB_dphi[0][2]*dU_dR + dB_dphi[1][2]*dU_dphi + dB_dphi[2][2]*dU_dlamda + B[0][2]*dR_dphi + B[1][2]*dphi_2 + B[2][2]*dphi_dlamda,
					dB_dlamda[0][2]*dU_dR + dB_dlamda[1][2]*dU_dphi + dB_dlamda[2][2]*dU_dlamda + B[0][2]*dR_dlamda + B[1][2]*dphi_dlamda + B[2][2]*dlamda_2 ])
		df_dr = np.array([df1_dr, df2_dr, df3_dr])	# 3*3
		df_dr = np.dot(df_dr, B)
		da_dr = np.dot( np.dot(I2F.T, df_dr),  I2F )
		return da_dr
		
		
	def partial_third(self, r_sat, time_tdb):
		'''第三天体引力摄动加速度对 (x, y, z) 的偏导数在月惯系下的表达'''
		earth = self.moon2Earth(time_tdb)	# moon -> earth
		sun = self.moon2Sun(time_tdb)	# moon -> sun
		l_earth, l_sun = r_sat-earth, r_sat-sun
		norm_earth, norm_sun = np.linalg.norm(l_earth, 2), np.linalg.norm(l_sun, 2)
		earth_dadr = - MIU_E / pow(norm_earth, 3) * ( np.identity(3) - 3/pow(norm_earth, 2) * np.outer(l_earth, l_earth) )
		sun_dadr = -MIU_S / pow(norm_sun, 3) * ( np.identity(3) - 3/pow(norm_earth, 2) * np.outer(l_sun, l_sun) )
		return earth_dadr + sun_dadr

		
		
		
if __name__ == "__main__":
	
	ob = Orbit()
	number = 20
	data = pd.read_csv("STK/Inertial_HPOP_30d.csv", nrows=number, usecols=range(1, 7))	# 取前number个点进行试算
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
	# da_dr = [ ob.partial_centre(r_sat) for r_sat in r_array ]
	nonsper_1 = [ ob.partial_nonspher(r_sat, tdb_jd) for (r_sat, tdb_jd) in zip(r_array, tdbJD_list) ]
	nonsper_2 = [ ob.partial_nonspher_wang(r_sat, tdb_jd) for (r_sat, tdb_jd) in zip(r_array, tdbJD_list) ]
	third = [ ob.partial_third(r_sat, tdb_jd) for (r_sat, tdb_jd) in zip(r_array, tdbJD_list) ]
	print(nonsper_1[:5], "\n\n", nonsper_2[:5])
