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
from orbit_predictor.keplerian import rv2coe
from orbit_predictor.angles import ta_to_M, M_to_ta

import sys
if sys.argv[1] == "matlab":
	import matlab.engine
	eng = matlab.engine.start_matlab()

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



import time
from functools import wraps
def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
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
		'''返回月惯系到月固系的方向余弦矩阵, single-time，月球轨道根数由平均轨道根数计算公式提供'''
		phi, theta, psi = eph.position("librations", tdb_jd)	# 物理天平动的三个欧拉角, np.array([0])
		CS2F = np.dot( np.dot(Rz_bas(psi), Rx_bas(theta)), Rz_bas(phi) )	# 月心天球 -> 月固系
		return np.dot(CS2F, LCS2I.T)	# 月惯系 -> 月固系, F = CS2F * I2CS * I
	
	
	def legendre_spher_alfs(self, phi, lm=CSD_LM):
		'''计算完全规格化缔合勒让德函数，球坐标形式，Brandon A. Jones - Efficient Models for the Evaluation and Estimation(eqs. 2.8)
		输入：地心纬度, rad;		阶次lm
		输出：可用于向量直接计算的勒让德函数，包含P_00项	np.array, B提前计算并存储以加快运算速度'''
		P = [ np.array([1, 0]), np.array([ sqrt(3)*sin(phi), sqrt(3)*cos(phi), 0 ]) ]
		for i in range(2, lm+1):
			P_ij = [ B[i][j] * sin(phi) * P[i-1][j] - B[i][j]/B[i-1][j] * P[i-2][j] for j in range(i) ]
			P_ii = sqrt((2*i+1)/(2*i)) * cos(phi) * P[i-1][i-1]
			P_ij.extend([P_ii, 0])
			P.append(np.array(P_ij))
		return np.array(P)
		
		
	def centreGravity(self, r_sat, miu=MIU_M):
		'''计算中心天体的引力加速度，single-time，输入 km, 输出 m/s^2
		输入：惯性系下卫星位置矢量r_sat，np.array, 单位 km;		miu默认为月球;
		输出：返回中心天体的引力加速度, np.array，单位 km/s^2'''
		r_norm = np.linalg.norm(r_sat, 2)
		g0 = -miu*r_sat / r_norm**3		# 中心引力 3*1
		return g0	# km/s^2

	@fn_timer	
	def nonspherGravity(self, r_sat, tdb_jd, miu=MIU_M, Re=RM, lm=CSD_LM):
		'''计算中心天体的非球形引力加速度，single-time, 输入 km, 输出 m/s^2, 王正涛-卫星跟踪卫星测量(公式2-4-7)
		输入：惯性系下卫星位置矢量r_sat，均为np.array, 单位 km;	tdb的儒略时间, float;	miu默认为月球;
		输出：返回中心天体的引力加速度, np.array，单位 km/s^2'''
		I2F = self.moon_Cbi(tdb_jd)	# 月惯系到月固系的方向余弦矩阵 3*3
		r_fixed = np.dot(I2F, r_sat)		# 应该在固连系下建立，王正涛
		r_norm = np.linalg.norm(r_fixed, 2)
		xy_norm = 1/np.linalg.norm(r_fixed[:2], 2)
		phi, lamda = atan2(r_fixed[2], 1/xy_norm), atan2(r_fixed[1], r_fixed[0])
		spher2rect = np.array([ [cos(phi)*cos(lamda), cos(phi)*sin(lamda), sin(phi)], \
					[(-1/r_norm)*sin(phi)*cos(lamda), (-1/r_norm)*sin(phi)*sin(lamda),  (1/r_norm)*cos(phi)], \
					[(-1/r_norm)*sin(lamda)/cos(phi), (1/r_norm)*cos(lamda)/cos(phi), 0] ])	#球坐标到直角坐标 3*3
		P = self.legendre_spher_alfs(phi, lm)	# 勒让德函数
		Vr, Vphi, Vlamda, const = 0, 0, 0, Re/r_norm
		for i in range(2, lm+1):	# 王正涛i从2开始
			temp_r, temp = (i+1)*pow(const, i), pow(const, i)
			for j in range(0, i+1):
				Vr += ( C[i][j]*cos(j*lamda) + S[i][j]*sin(j*lamda) ) * P[i][j] * temp_r
				Vphi += ( C[i][j]*cos(j*lamda) + S[i][j]*sin(j*lamda) ) * \
						( pi_dot[i][j] * P[i][j+1] - j * r_fixed[2]*xy_norm * P[i][j] ) * temp 	# check dP_lm!
				Vlamda += ( -C[i][j]*sin(j*lamda) + S[i][j]*cos(j*lamda) ) * P[i][j] * temp * j
		g1 = np.array([-miu/r_norm**2*Vr, miu/r_norm*Vphi, miu/r_norm*Vlamda])	# 非球形引力 3*1
		g1 = np.dot(g1, spher2rect)	 # 球坐标到直角坐标，乘积顺序不要反了
		g1 = np.dot(I2F.T, g1)	# 将月固系下加速度转换到月惯系下
		return g1*1000	# km/s^2
		
	
	@fn_timer	
	def nonspher_moon(self, r_sat, tdb_jd, miu=MIU_M, Re=RM, lm=CSD_LM):
		'''计算月球的非球形引力加速度，single-time, Brandon A. Jones - Efficient Models for the Evaluation and Estimation(eq. 2.14)'''
		I2F = self.moon_Cbi(tdb_jd)	# 月惯系到月固系的方向余弦矩阵 3*3
		r_fixed = np.dot(I2F, r_sat)		# 应该在固连系下建立，王正涛
		r_norm = np.linalg.norm(r_fixed, 2)
		xy_norm = 1/np.linalg.norm(r_fixed[:2], 2)
		phi, lamda = asin(r_fixed[2] / r_norm), atan2(r_fixed[1], r_fixed[0])	# Keric A. Hill-Autonomous Navigation in Libration Point Orbits
		P = self.legendre_spher_alfs(phi, lm)	# 勒让德函数
		rRatio = Re/r_norm
		# 球坐标对直角坐标的偏导数, Keric A. Hill(eq. 8.10), Brandon A. Jones(eq. 2.13)
		dR_dr = r_fixed / r_norm
		dphi_dr = xy_norm * np.array([ -r_fixed[0]*r_fixed[2] / pow(r_norm, 2), -r_fixed[1]*r_fixed[2] / pow(r_norm, 2), \
									1 - pow(r_fixed[2], 2) / pow(r_norm, 2) ])
		dlamda_dr = pow(xy_norm, 2) * np.array([ -r_fixed[1], r_fixed[0], 0 ])
		spher2rect = np.array([dR_dr, dphi_dr, dlamda_dr])
		cos_m = np.array([ cos(j*lamda) for j in range(0, lm+1) ])
		sin_m = np.array([ sin(j*lamda) for j in range(0, lm+1) ])
		dU_dr = -miu/pow(r_norm, 2) * ( sum([ pow(rRatio, i)*(i+1) * np.dot(P[i][:-1], C[i]*cos_m[:i+1] + S[i]*sin_m[:i+1]) for i in range(2, lm+1) ]) )
		dU_dphi = miu/r_norm * sum([ pow(rRatio, i) * np.dot(P[i][1:] * pi_dot[i] -  r_fixed[2]*xy_norm*M_Array[:i+1] * P[i][:-1], \
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
		norm_st = np.linalg.norm(r_sat-r_sun, 2)
		norm_et = np.linalg.norm(r_sun, 2)
		a_sun = -MIU_S * ((r_sat-r_sun)/norm_st**3 + r_sun/norm_et**3)
		return a_sun	# km/s^2
		
		
	def thirdEarth(self, r_sat, tdb_jd):
		'''计算地球对卫星产生的第三天体摄动加速度, single-time, 输入km, 输出m/s^2
		输入：UTC时间time_utc，卫星相对中心天体的矢量，m, 均为np.array;
		输出：返回第三体地球摄动加速度, km/s^2, np.array'''
		r_earth = np.dot( LCS2I, self.moon2Earth(tdb_jd) )	# 转换为月心惯性系下的位置矢量
		norm_st = np.linalg.norm(r_sat-r_earth, 2)
		norm_et = np.linalg.norm(r_earth, 2)
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
		print(tdb_jd, "\t\t", t)
		R, V = RV[:3], (RV[3:]).tolist()	# km, km/s
		F0 = self.centreGravity(R, miu)
		F1 = self.nonspher_moon(R, tdb_jd, miu, Re, lm)
		F2 = self.thirdEarth(R, tdb_jd)
		F3 = self.thirdSun(R, tdb_jd)
		F4 = self.solarPress(R, tdb_jd)	# 加入太阳光压摄动
		F = F0 + F1 + F2 + F3 #+ F4	# km/s^2
		return F
		# V.extend(F)
		# return np.array(V)
		
	@fn_timer	
	def integrate_orbit(self, rv_0, num):
		'''数值积分器，使用RK45获得卫星递推轨道'''
		ode_y = solve_ivp( self.twobody_dynamic, (0,STEP*num), rv_0, method="RK45", rtol=1e-10, t_eval=range(0,STEP*num, STEP) ).y
		return ode_y


		
		
		
if __name__ == "__main__":
	
	ob = Orbit()
	number = 20
	data = pd.read_csv("STK/Inertial HPOP_30d.csv", nrows=number, usecols=range(1, 7))	# 取前number个点进行试算
	data = pd.read_csv("STK/Inertial TwoBody_30d.csv", nrows=number, usecols=range(1, 7))
	RV_array = data.values
	r_array = RV_array[:, :3]
	t_list = range(0, number*STEP, STEP)
	utc_array = (ob.generate_time(start_t="20180101", end_t="20180331"))[:number]
	utcJD_list = [ time_utc.to_julian_date() for time_utc in utc_array ]
	tdbJD_list = [ time_utc.to_julian_date() + 69.184/86400 for time_utc in utc_array ]
	I2F_list = [ ob.moon_Cbi(tdb_jd) for tdb_jd in tdbJD_list ]
	rFixed_list = [ np.dot(I2F, r_sat) for (I2F, r_sat) in zip(I2F_list, r_array) ]
	r_sat, r_fixed, RV, time_utc = r_array[10], rFixed_list[10], RV_array[10], utc_array[10]
	utc_jd, tdb_jd = time_utc.to_julian_date(), time_utc.to_julian_date() + 69.184/86400
	a0 = np.array([ ob.centreGravity(RV) for t, RV in zip(t_list, RV_array) ])
	a_inertial = np.array([ ob.complete_dynamic(t, RV) for t, RV in zip(t_list, RV_array) ])

