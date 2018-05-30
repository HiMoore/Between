# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from math import *
from datetime import datetime, timedelta
import de421
from jplephem import Ephemeris
from jplephem.spk import SPK
from scipy.integrate import solve_ivp



# 恒定常值变量区
MIU_E, MIU_M, MIU_S = 398600.4415, 4902.80030555540, 132712422595.6590		# 引力系数，单位 km^3 / s^2
RE, RM, RS = 6378.1363, 1738.000, 69550.0		#天体半径，单位 km
number, CSD_LM, STEP = 495, 30, 120	#全局积分步长
Beta = 0
C, S = np.load("STK/Clm.npy"), np.load("STK/Slm.npy")	# 序列化后代码速度提高4倍


# 计算常值变量与函数区, 加速运算过程
kernel = SPK.open(r"..\de421\de421.bsp")
eph = Ephemeris(de421)
time_utc = pd.Timestamp("2018-1-1 00:00:00", freq="1S")
B = []
for i in range(CSD_LM+1):
	A = [ sqrt( (2*i+1)*(2*i-1) / ((i+j)*(i-j)) ) for j in range(i) ]
	A.append(1); B.append(np.array(A))
B = np.array(B)		# B用于完全规格化勒让德函数计算
pi_dot = np.array([ np.array( [sqrt((i+1)*i/2)] + [ sqrt( (i+j+1)*(i-j) ) for j in range(1, i+1)] ) for i in range(CSD_LM+1) ])
M_Array = np.array([ j for j in range(0, CSD_LM+1) ])
Rx_bas = lambda x: np.array([[1, 0, 0], [0, cos(x), sin(x)], [0, -sin(x), cos(x)]])
Ry_bas = lambda y: np.array([[cos(y), 0, -sin(y)], [0, 1, 0], [sin(y),0, cos(x)]])
Rz_bas = lambda z: np.array([[cos(z), sin(z), 0], [-sin(z), cos(z), 0], [0, 0, 1]])
LCS2I = np.dot( Rx_bas(radians(24.358897)), Rz_bas(radians(-3.14227)) )		# 月心天球 -> 月心惯性系



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
	
	def __init__(self):
		self.rv = np.zeros(6)
		self.time = 0
			
	def __del__(self):
		return	
		
	def readCoffients(self, number=3321, n=80):
		'''获取中心天体球谐引力系数，默认前30阶，包括0阶项, np.array'''
		df = pd.read_csv("STK/GL0660B.grv", sep="\s+", header=None, nrows=number)
		f = df[:number]
		f.columns = ["l", "m", "Clm", "Slm"]
		f = f.set_index([f["l"], f["m"]]); del f['l'], f['m']
		Clm, Slm = f["Clm"], f["Slm"]
		Clm = [ np.array(Clm.loc[i]) for i in range(0, n+1) ]
		Slm = [ np.array(Slm.loc[i]) for i in range(0, n+1) ]
		np.save("STK/Clm.npy", Clm); np.save("STK/Slm.npy", Slm)	
		
		
	def generate_time(self, start_t="20171231", end_t="20180101"):
		'''产生Timestamp列表，用于计算对应的儒略日, multi-time
		输入：起始和终止时间		str类型
		输出：返回产生的utc时间列表	Timestamp构成的list'''
		start_t = datetime.strptime(start_t, "%Y%m%d")
		end_t = datetime.strptime(end_t, "%Y%m%d")
		time_list = pd.date_range(start_t, end_t, freq=str(STEP)+"S")	#固定步长120s
		return time_list
		
		
	def moon2Earth(self, tdb_jd):
		'''计算地球相对于月球的位置矢量在地心J2000系下的表示, single-time, km
		输入：TDB的儒略日时间										Timestamp			
		输出：返回地球相对于月球的位置矢量(地球J2000下), 单位 km		np.array'''
		M2E_bc = -kernel[3, 301].compute(tdb_jd)	# 月球 -> 地月中心
		E_bc2E = kernel[3, 399].compute(tdb_jd)	# 地月中心 -> 地球
		return M2E_bc + E_bc2E	# 月球 -> 地球
		
		
	def moon2Sun(self, tdb_jd):		
		'''计算太阳相对于月球的位置矢量在地心J2000系下的表示, single-time, km
		输入：TDB的儒略日时间										Timestamp
		输出：返回太阳相对于月球的位置矢量(地球J2000下), 单位 km,		np.array'''
		E_bc2M = kernel[3, 301].compute(tdb_jd)	# 地月中心 -> 月球
		S_bc2E_bc = kernel[0, 3].compute(tdb_jd)	# 太阳系中心 -> 地月中心
		S2S_bc = -kernel[0, 10].compute(tdb_jd)	# 太阳 -> 太阳系中心
		return -(S2S_bc + S_bc2E_bc + E_bc2M)	# 月球 -> 太阳
		
		
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
			P_ij.extend([P_ii, 0]); P.append(np.array(P_ij))
		return np.array(P)
		
		
	def legendre_diff(self, phi, P, lm=CSD_LM):
		'''计算完全规格化缔合勒让德函数的一阶导数，球坐标形式，Brandon A. Jones(eqs. 2.14)
		输入：地心纬度, rad;	完全规格化勒让德函数P;		阶次lm'''
		tan_phi = tan(phi)
		dP = np.array([ P[i][1:] * pi_dot[i] - M_Array[:i+1]*tan_phi * P[i][:-1] for i in range(0, lm+1) ])
		return dP
		
		
	def centreGravity(self, r_sat, miu=MIU_M):
		'''计算中心天体的引力加速度，single-time，输入 km, 输出 km/s^2
		输入：惯性系下卫星位置矢量r_sat，np.array, 单位 km;		miu默认为月球;
		输出：返回中心天体的引力加速度, np.array，单位 km/s^2'''
		r_norm = np.linalg.norm(r_sat, 2)
		g0 = -miu*r_sat / r_norm**3		# 中心引力 3*1
		return g0	# km/s^2
		
		
	def nonspher_J2(self, r_sat, miu=MIU_M, Re=RM):
		'''计算J2项的摄动引力加速度'''
		J2 = C[2][0]; x, y, z = r_sat
		r_norm = np.linalg.norm(r_sat, 2)
		pow_r2, pow_r3 = pow(r_norm, 2), pow(r_norm, 3)
		fx = -miu*x/pow_r3 * J2 * pow((Re/r_norm), 2) * (7.5*pow(z,2)/pow_r2 - 1.5)
		fy = -miu*y/pow_r3 * J2 * pow((Re/r_norm), 2) * (7.5*pow(z,2)/pow_r2 - 1.5)
		fz = -miu*z/pow_r3 * J2 * pow((Re/r_norm), 2) * (7.5*pow(z,2)/pow_r2 - 4.5)
		J2_inertial = np.array([ fx, fy, fz ])
		return J2_inertial
		

	def nonspher_moon(self, r_sat, tdb_jd, miu=MIU_M, Re=RM, lm=CSD_LM):
		'''计算月球的非球形引力加速度，single-time, Brandon A. Jones - Efficient Models for the Evaluation and Estimation(eq. 2.14)
		输入：惯性系下卫星位置矢量r_sat，np.array, 单位 km;		TDB的儒略日时间;	miu默认为月球;
		输出：返回非球形引力摄动加速度, ndarray，单位 km/s^2'''
		I2F = self.moon_Cbi(tdb_jd)	# 月惯系到月固系的方向余弦矩阵 3*3
		r_fixed = np.dot(I2F, r_sat)		# 应该在固连系下建立，王正涛
		r_norm, xy_norm = np.linalg.norm(r_fixed, 2), np.linalg.norm(r_fixed[:2], 2)
		phi, lamda = asin(r_fixed[2] / r_norm), atan2(r_fixed[1], r_fixed[0])
		rRatio, pow_r2, tan_phi = Re/r_norm, pow(r_norm, 2), r_fixed[2]/xy_norm
		cos_m = np.array([ cos(j*lamda) for j in range(0, lm+1) ])
		sin_m = np.array([ sin(j*lamda) for j in range(0, lm+1) ])
		P = self.legendre_spher_alfs(phi, lm)	# 勒让德函数
		dU_dr = -miu/pow_r2 * sum([ pow(rRatio, i)*(i+1) * np.dot(P[i][:-1], C[i]*cos_m[:i+1] + S[i]*sin_m[:i+1]) for i in range(2, lm+1) ]) 
		dU_dphi = miu/r_norm * sum([ pow(rRatio, i) * np.dot(P[i][1:] * pi_dot[i] -  tan_phi*M_Array[:i+1] * P[i][:-1], \
							C[i]*cos_m[:i+1] + S[i]*sin_m[:i+1]) for i in range(2, lm+1) ])
		dU_dlamda = miu/r_norm * sum([ pow(rRatio, i) * np.dot(M_Array[:i+1]*P[i][:-1], S[i]*cos_m[:i+1] - C[i]*sin_m[:i+1]) for i in range(2, lm+1) ])
		dR_dr = r_fixed / r_norm	# 球坐标对直角坐标的偏导数, Keric Hill(eq. 8.10), Brandon Jones(eq. 2.13)
		dphi_dr = 1/xy_norm * np.array([ -r_fixed[0]*r_fixed[2]/pow_r2,  -r_fixed[1]*r_fixed[2]/pow_r2,  1 - pow(r_fixed[2], 2)/pow_r2 ])
		dlamda_dr = 1/pow(xy_norm, 2) * np.array([ -r_fixed[1], r_fixed[0], 0 ])
		a_fixed = dR_dr * dU_dr + dphi_dr * dU_dphi + dlamda_dr * dU_dlamda
		a_inertial = np.dot(I2F.T, a_fixed)	# 将月固系下加速度转换到月惯系下
		return a_inertial	# km/s^2, 3*1
		

	def thirdSun(self, r_sat, tdb_jd):
		'''计算太阳对卫星产生的第三天体摄动加速度, single-time, 输入 km, 输出 km/s^2
		输入：UTC时间time_utc，卫星相对中心天体的矢量，km, 均为np.array；
		输出：返回第三体太阳摄动加速度, km/s^2, np.array'''
		r_sun = np.dot( LCS2I, self.moon2Sun(tdb_jd) )	# 转换为月心惯性系下的位置矢量
		norm_st = np.linalg.norm(r_sat-r_sun, 2)	# 太阳 -> 卫星
		norm_et = np.linalg.norm(r_sun, 2)	# 月球 -> 太阳
		a_sun = -MIU_S * ( (r_sat-r_sun)/pow(norm_st, 3) + r_sun/pow(norm_et, 3) )
		return a_sun	# km/s^2
		
		
	def thirdEarth(self, r_sat, tdb_jd):
		'''计算地球对卫星产生的第三天体摄动加速度, single-time, 输入 km, 输出 km/s^2
		输入：UTC时间time_utc，卫星相对中心天体的矢量，km, 均为np.array;
		输出：返回第三体地球摄动加速度, km/s^2, np.array'''
		r_earth = np.dot( LCS2I, self.moon2Earth(tdb_jd) )	# 转换为月心惯性系下的位置矢量
		norm_st = np.linalg.norm(r_sat-r_earth, 2)	# l(vector), 地球 -> 卫星
		norm_et = np.linalg.norm(r_earth, 2)	# 月球 -> 地球
		a_earth = -MIU_E * ( (r_sat-r_earth)/pow(norm_st, 3) + r_earth/pow(norm_et, 3) )
		return a_earth	# km/s^2
		
		
	def solarPress(self, r_sat, tdb_jd, step=120):
		'''计算太阳光压摄动, single-time
		输入：beta = (1+ita)*S(t);  utc时间(datetime);	卫星位置矢量，m, np.array
		输出：返回太阳光压在r_sat处造成的摄动, np.array'''
		r_sun = self.moon2Sun(tdb_jd)		#月球->太阳矢量
		ro = 4.5605e-6	#太阳常数
		delta_0 = 1.495978707e+8	#日地距离
		delta = r_sat - r_sun
		norm_delta = np.linalg.norm(delta, 2)
		# 计算太阳光压摄动中的系数，使用随机漫步过程模拟
		v = np.random.normal(0, 1)	# 0均值，协方差为1的高斯白噪声序列
		sigma_beta = 3e-5	# 9.18e-6
		global Beta
		Beta += sigma_beta*sqrt(step)*v
		F = Beta * ro*(delta_0**2/norm_delta**2) * delta/norm_delta
		return F / 1000

	
	def twobody_dynamic(self, t, RV,  miu=MIU_M):
		'''二体动力学方程，仅考虑中心引力u
		输入：	时间t，从0开始（好像不太重要）; 	惯性系下位置速度	km, km/s, np.array
		输出：	返回d(RV)/dt，动力学增量, km/s, km/s^2, np.array'''
		R_Norm = np.linalg.norm(RV[:3], 2)
		dr_dv = []
		dr_dv.extend(RV[3:])
		dr_dv.extend(-miu*(RV[:3]) / pow(R_Norm, 3))
		return np.array(dr_dv)	# km/s, km/s^2
		
		
	def J2_dynamic(self, t, RV, miu=MIU_M, Re=RM, lm=CSD_LM):
		'''系统完整动力学模型, 暂考虑中心引力，非球形引力，第三天体引力
		输入：	时间t，从0开始（好像不太重要）; 	惯性系下位置速度	km, km/s, np.array
		输出：	返回d(RV)/dt，动力学增量, np.array, 1*6, m/s, m/s^2'''
		tdb_jd = (time_utc+int(t)).to_julian_date() + 69.184/86400
		R, V = RV[:3], (RV[3:]).tolist()	# km, km/s
		F0 = self.centreGravity(R, miu)
		F1 = self.nonspher_J2(R, miu, Re) 
		F = F0 + F1		# km/s^2
		V.extend(F)
		return np.array(V)		# km/s, km/s^2
		
	
	def complete_dynamic(self, t, RV, miu=MIU_M, Re=RM, lm=CSD_LM):
		'''系统完整动力学模型, 暂考虑中心引力，非球形引力，第三天体引力
		输入：	时间t，从0开始（好像不太重要）; 	惯性系下位置速度	km, km/s, np.array
		输出：	返回d(RV)/dt，动力学增量, np.array, 1*6, m/s, m/s^2'''
		tdb_jd = (time_utc+int(t)).to_julian_date() + 69.184/86400
		# print("utc_jd:", time_utc.to_julian_date(), '\t', "t:", t, '\t', "tdb_jd:", tdb_jd)
		R, V = RV[:3], (RV[3:]).tolist()	# km, km/s
		F0 = self.centreGravity(R, miu)
		F1 = self.nonspher_moon(R, tdb_jd, miu, Re) 
		F2 = self.thirdEarth(R, tdb_jd)
		F3 = self.thirdSun(R, tdb_jd)
		F = F0 + F1 + F2 + F3		# km/s^2
		V.extend(F)
		return np.array(V)		# km/s, km/s^2	
		
		
	def integrate_orbit(self, rv_0, num):
		'''数值积分器，使用RK45获得卫星递推轨道'''
		ode_y = solve_ivp( self.complete_dynamic, (0,STEP*num), rv_0, method="RK45", rtol=1e-9, atol=1e-12, \
							t_eval=range(0, STEP*num, STEP) ).y
		return ode_y
		

	def integrate_twobody(self, rv_0, num):
		'''数值积分器，使用RK45获得卫星递推轨道'''
		ode_y = solve_ivp( self.twobody_dynamic, (0,STEP*num), rv_0, method="RK45", rtol=1e-9, atol=1e-12, \
							t_eval=range(0, STEP*num, STEP) ).y
		return ode_y
		
		
	def integrate_J2(self, rv_0, num):
		'''数值积分器，使用RK45获得卫星递推轨道'''
		ode_y = solve_ivp( self.J2_dynamic, (0,STEP*num), rv_0, method="RK45", rtol=1e-9, atol=1e-12, \
							t_eval=range(0, STEP*num, STEP) ).y
		return ode_y


	def partial_centre(self, r_sat, miu=MIU_M):
		'''中心引力加速度对 (x, y, z) 的偏导数在月惯系下的表达, 单位 1/s^2'''
		r_norm = np.linalg.norm(r_sat, 2)
		gradient = miu * (3*np.outer(r_sat, r_sat) - pow(r_norm, 2)*np.identity(3)) / pow(r_norm, 5)
		return gradient
		
	
	def partial_nonspher(self, r_sat, tdb_jd, miu=MIU_M, Re=RM, lm=CSD_LM):
		'''非球形引力摄动加速度 对 (x, y, z) 的偏导数, 在月心惯性系下的表达(转换已完成), Keric A. Hill(eq. 8.14)
		输入: 月心惯性系下的位置矢量; 	动力学时对应的儒略时间;  输出: 单位 1/s^2; 		列表推导调用下单次计算0.005s左右'''
		I2F = self.moon_Cbi(tdb_jd)	# 月惯系到月固系的方向余弦矩阵 3*3
		r_fixed = np.dot(I2F, r_sat)		# 应该在固连系下建立，王正涛
		x, y, z, r_norm = r_fixed[0], r_fixed[1], r_fixed[2], np.linalg.norm(r_fixed, 2)
		xy_norm, pow_r2, pow_r3 = np.linalg.norm(r_fixed[:2], 2), pow(r_norm, 2), pow(r_norm, 3)
		phi, lamda, cos_phi, tan_phi = asin(z / r_norm), atan2(y, x), xy_norm / r_norm, z / xy_norm	# Keric A. Hill
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
		# 3 * [ (3*1) * (1*3) ]   +   3 * [ (3*3) * const]
		da_dr = np.dot(dR_dr.T, dr_dR) + np.dot(dphi_dr.T, dr_dphi) + np.dot(dlamda_dr.T, dr_dlamda) \
				+ d2R_dr2 * dU_dR + d2phi_dr2 * dU_dphi + d2lamda_dr2 * dU_dlamda
		da_dr = np.dot( np.dot(I2F.T, da_dr) ,  I2F ) 	# Hill(eq 8.28) and 苏勇-利用GOCE和GRACE卫星(eq 2-67)
		return da_dr
	
		
	def partial_third(self, r_sat, time_tdb):
		'''第三天体引力摄动加速度对 (x, y, z) 的偏导数在月惯系下的表达, 单位 1/s^2'''
		earth = self.moon2Earth(time_tdb)	# moon -> earth
		sun = self.moon2Sun(time_tdb)	# moon -> sun
		l_earth, l_sun = r_sat-earth, r_sat-sun
		norm_earth, norm_sun = np.linalg.norm(l_earth, 2), np.linalg.norm(l_sun, 2)
		earth_dadr = - MIU_E / pow(norm_earth, 3) * ( np.identity(3) - 3/pow(norm_earth, 2) * np.outer(l_earth, l_earth) )
		sun_dadr = -MIU_S / pow(norm_sun, 3) * ( np.identity(3) - 3/pow(norm_sun, 2) * np.outer(l_sun, l_sun) )
		return earth_dadr + sun_dadr

		
	def coefMatrix_single(self, r_sat, tdb_jd):
		'''单颗卫星构成的状态方程的Jaccobian矩阵计算公式中的系数矩阵项,  王正涛(eq 3-3-5)
		输出: 状态转移矩阵一阶微分方程的系数矩阵, np.array([ [0, I], [da_dr, da_dv] ]);  	平均计算时间0.004s'''
		da_dr = self.partial_centre(r_sat) + self.partial_nonspher(r_sat, tdb_jd) + self.partial_third(r_sat, tdb_jd)
		da_dv, O, I = np.zeros((3,3)), np.zeros((3,3)), np.identity(3)
		up, low = np.hstack((O, I)), np.hstack((da_dr, da_dv))
		M_st = np.vstack((up, low))		# 6*6
		return M_st		# dP = M_st * P

		
		
		
if __name__ == "__main__":
	
	ob = Orbit()
	# ob.readCoffients()
	number = 10
	data = pd.read_csv("STK/Part_2/1_Inertial_HPOP_660.csv", nrows=number, usecols=range(1, 7))	# 取前number个点进行试算
	RV_array = data.values
	r_array = RV_array[:, :3]
	t_list = range(0, number*STEP, STEP)
	utc_array = (ob.generate_time(start_t="20180101", end_t="20180131"))[:number]
	utcJD_list = [ time_utc.to_julian_date() for time_utc in utc_array ]
	tdbJD_list = [ time_utc.to_julian_date() + 69.184/86400 for time_utc in utc_array ]
	r_sat, RV, time_utc = r_array[0], RV_array[0], utc_array[0]
	utc_jd, tdb_jd = time_utc.to_julian_date(), time_utc.to_julian_date() + 69.184/86400
	X0 = np.array([ 1.84032000e+03,  0.00000000e+00,  0.00000000e+00, -0.00000000e+00, 1.57132000e+00,  8.53157000e-01])
	# print(ob.coefMatrix_single(r_sat, tdb_jd))
	
	centre_1 = np.array([ ob.partial_third(r_sat, tdb_jd) for r_sat, tdb_jd in zip(r_array[:10], tdbJD_list[:10]) ])
	centre_2 = np.array([ ob.partial_third_1(r_sat, tdb_jd) for r_sat, tdb_jd in zip(r_array[:10], tdbJD_list[:10]) ])
	centre_3 = np.array([ ob.partial_third_2(r_sat, tdb_jd) for r_sat, tdb_jd in zip(r_array[:10], tdbJD_list[:10]) ])
	
