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

# import matlab.engine
# eng = matlab.engine.start_matlab()


MIU_E, MIU_M, MIU_S = 398600.4415, 4902.801056, 132712422595.6590	# 引力系数，单位 km^3 / s^2
RE, RM, RS = 6378.1363, 1738.000, 69550.0		#天体半径，单位 km
CSD_MAX, CSD_EPS, number = 1e+30, 1e-10, 495
STEP = 120		#全局积分步长
time_utc = pd.Timestamp("2018-1-1 00:00:00", freq=str(STEP)+"S")
beta = 0 	# 太阳光压参数
kernel = SPK.open(r"..\de421\de421.bsp")
eph = Ephemeris(de421)
C, S = np.load("Clm.npy"), np.load("Slm.npy")	# 序列化后代码速度提高4倍



import time
from functools import wraps
import random 
 
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
		
		
	def generate_time(self, start_t="20171231", end_t="20180101"):
		'''产生Timestamp列表，用于计算对应的儒略日, multi-time
		输入：起始和终止时间		str类型
		输出：返回产生的utc时间列表	Timestamp构成的list'''
		start_t = datetime.strptime(start_t, "%Y%m%d")
		end_t = datetime.strptime(end_t, "%Y%m%d")
		global STEP
		time_list = pd.date_range(start_t, end_t, freq=str(STEP)+"S")	#固定步长120s
		return time_list
		
		
	def moon2Earth(self, time_jd):
		'''计算地球相对于月球的位置矢量, single-time
		输入：儒略日时间											Timestamp			
		输出：返回地球相对于月球的位置矢量(J2000下), 单位 km		np.array'''
		M2E_bc = -kernel[3, 301].compute(time_jd)	# 月球 -> 地月中心
		E_bc2E = kernel[3, 399].compute(time_jd)	# 地月中心 -> 地球
		return M2E_bc + E_bc2E	# 月球 -> 地球
		
		
	def moon2Sun(self, time_jd):		
		'''计算太阳相对于月球的位置矢量, single-time
		输入：儒略日时间											Timestamp
		输出：返回太阳相对于月球的位置矢量(J2000下), 单位 km,		np.array'''
		E_bc2M = kernel[3, 301].compute(time_jd)	# 地月中心 -> 月球
		S_bc2E_bc = kernel[0, 3].compute(time_jd)	# 太阳系中心 -> 地月中心
		S2S_bc = -kernel[0, 10].compute(time_jd)	# 太阳 -> 太阳系中心
		return -(S2S_bc + S_bc2E_bc + E_bc2M)
		
		
	def moon_Cbi(self, time_jd):
		'''返回月惯系到月固系的方向余弦矩阵, single-time，月球轨道根数由平均轨道根数计算公式提供'''
		jd_t = (time_jd - 2451545.0) / 36525.0
		Omega = mod( radians(125.044555556 - 1934.1361850*jd_t + 0.0020767*jd_t**2), 2*pi )
		I, epsilon = 0.02691686, 0.40909063	# 平赤道倾角，黄赤交角
		omega, Is, lamda = eph.position("librations", time_jd)	# 物理天平动的三个欧拉角, np.array([0])
		Rx = lambda x: np.array([[1, 0, 0], [0, cos(x), sin(x)], [0, -sin(x), cos(x)]])
		Ry = lambda y: np.array([[cos(y), 0, -sin(y)], [0, 1, 0], [sin(y),0, cos(x)]])
		Rz = lambda z: np.array([[cos(z), sin(z), 0], [-sin(z), cos(z), 0], [0, 0, 1]])
		N_T = (np.dot( np.dot( np.dot(Rz(-Omega), Rx(-I)), Rz(Omega) ), Rx(epsilon) )).T
		M1 = np.dot( np.dot(Rz(lamda), Rx(Is)), Rz(omega) )
		return np.dot(M1, N_T)	# 月惯系 -> 月固系
		
		
	def legendre_spher_col(self, phi, lm=30):
		'''计算完全规格化缔合勒让德函数，球坐标形式，张飞-利用函数计算重力场元(公式2.14)，标准向前列递推算法, 
		输入：地心纬度phi, rad;		阶次lm
		输出：可用于向量直接计算的勒让德函数，包含P_00项	np.array'''
		P = [ np.array([1, 0]), np.array([sqrt(3)*sin(phi), sqrt(3)*cos(phi), 0]) ]
		for i in range(2, lm+1):	# P[0][0] -- P[30][30]存在，P[i][i+1]均为0
			p_ij = [ sqrt( (4*i**2-1) / (i**2-j**2) ) * sin(phi) * P[i-1][j] - \
					sqrt( (2*i+1)*(i+j-1)*(i-j-1) / ((i**2-j**2)*(2*i-3)) ) * P[i-2][j] for j in range(i) ]
			p_ii = sqrt( (2*i+1)/(2*i) ) * cos(phi) * P[i-1][i-1]
			p_ij.extend([p_ii, 0])
			P.append(np.array(p_ij))
		return np.array(P)
		
		
	def legendre_cart(self, r_fixed, Re=RM, lm=30):
		'''计算缔合勒让德函数，直角坐标形式，钟波-基于GOCE卫星(公式2.2.12)
		输入：月固系下的卫星位置矢量, r_fixed, 		np.array
		输出：直角坐标下的勒让德函数，包含0阶项, 	np.array'''
		X, Y, Z = r_fixed[0], r_fixed[1], r_fixed[2]
		r = np.linalg.norm(r_fixed, 2)
		V = [ np.array([Re/r, 0]), np.array([ sqrt(3)*Z*Re**2/r**3, sqrt(3)*X*Re**2/r**3, 0]) ]	
		W = [ np.array([0, 0]), np.array([ 0, sqrt(3)*Y*Re**2/r**3, 0]) ]
		cons_x, cons_y, cons_z, const = X*Re/r**2, Y*Re/r**2, Z*Re/r**2, (Re/r)**2
		for i in range(2, lm+2):
			Vij = [ sqrt( (4*i**2-1) / (i**2-j**2) ) * cons_z * V[i-1][j] - \
					sqrt( (2*i+1)*(i-j-1)*(i+j-1) / ((i**2-j**2)*(2*i-3)) ) * const * V[i-2][j] for j in range(i) ]
			Vii = sqrt( (2*i+1) / (2*i) ) * ( cons_x * V[i-1][i-1] - cons_y * W[i-1][i-1] )
			Vij.extend([Vii, 0]); V.append(np.array(Vij))
			
			Wij = [ sqrt( (4*i**2-1) / (i**2-j**2) ) * cons_z * W[i-1][j] - \
					sqrt( (2*i+1)*(i-j-1)*(i+j-1) / ((i**2-j**2)*(2*i-3)) ) * const * W[i-2][j] for j in range(i) ]
			Wii = sqrt( (2*i+1) / (2*i) ) * ( cons_x * W[i-1][i-1] + cons_y * V[i-1][i-1] )	#钟波/王庆宾此处为加号
			Wij.extend([Wii, 0]); W.append(np.array(Wij))
		return (np.array(V), np.array(W))
		
		
	def centreGravity(self, r_sat, miu=MIU_M):
		'''计算中心天体的引力加速度，single-time，输入 km, 输出 m/s^2
		输入：惯性系下卫星位置矢量r_sat，np.array, 单位 km;		miu默认为月球;
		输出：返回中心天体的引力加速度, np.array，单位 m/s^2'''
		r_norm = np.linalg.norm(r_sat, 2)
		g0 = -miu*r_sat / r_norm**3		# 中心引力 3*1
		return g0 * 1000

		
	def nonspherGravity(self, r_sat, time_jd, miu=MIU_M, Re=RM, lm=30):
		'''计算中心天体的非球形引力加速度，single-time, 输入 km, 输出 m/s^2, 王正涛-卫星跟踪卫星测量(公式2-4-7)
		输入：惯性系下卫星位置矢量r_sat，均为np.array, 单位 km;	儒略时间, float;	miu默认为月球;
		输出：返回中心天体的引力加速度, np.array，单位 m/s^2'''
		HL = self.moon_Cbi(time_jd)	# 月惯系到月固系的方向余弦矩阵 3*3
		r_fixed = np.dot(HL, r_sat)		# 应该在固连系下建立，王正涛
		r_norm = np.linalg.norm(r_fixed, 2)
		phi, lamda = atan(r_fixed[2] / sqrt(r_fixed[0]**2+r_fixed[1]**2)), atan(r_fixed[1]/r_fixed[0])
		spher2rect = np.array([ [cos(phi)*cos(lamda), cos(phi)*sin(lamda), sin(phi)], \
					[(-1/r_norm)*sin(phi)*cos(lamda), (-1/r_norm)*sin(phi)*sin(lamda),  (1/r_norm)*cos(phi)], \
					[(-1/r_norm)*sin(lamda)/cos(phi), (1/r_norm)*cos(lamda)/cos(phi), 0] ])	#球坐标到直角坐标 3*3
		P = self.legendre_spher_col(phi, lm)	# 勒让德函数
		Vr, Vphi, Vlamda, const = 0, 0, 0, Re/r_norm
		for i in range(2, lm):	# 王正涛i从2开始
			temp_r, temp = (i+1)*const**i, const**i
			Vr +=  C[i][0] * P[i][0] * temp_r		# j=0时dP不同，需要单独计算
			Vphi += C[i][0] * ( sqrt(i*(i+1)/2) * P[i][1] ) * temp
			for j in range(1, i+1):
				Vr += ( C[i][j]*cos(j*lamda) + S[i][j]*sin(j*lamda) ) * P[i][j] * temp_r
				Vphi += ( C[i][j]*cos(j*lamda) + S[i][j]*sin(j*lamda) ) * \
						( sqrt((i-j)*(i+j+1)) * P[i][j+1] - j * tan(phi) * P[i][j] ) * temp 	# check dP_lm!
				Vlamda += ( -C[i][j]*sin(j*lamda) + S[i][j]*cos(j*lamda) ) * P[i][j] * temp * j
		g1 = np.array([-miu/r_norm**2*Vr, miu/r_norm*Vphi, miu/r_norm*Vlamda])	# 非球形引力 3*1
		g1 = np.dot(g1, spher2rect)	 # 球坐标到直角坐标，乘积顺序不要反了
		g1 = np.dot(HL.T, g1)	# 将月固系下加速度转换到月惯系下
		return g1 * 1000
		
		
	def diff_legendre_spher(self, phi, P, lm=30):
		'''计算完全规格化缔合勒让德函数的一阶导数，球坐标形式，王正涛-卫星跟踪卫星测量(公式2-4-6)
		输入：地心纬度phi， 勒让德函数P
		输出：一阶导数，包含dP_00项		np.array'''
		deri_P, tan_phi = [], tan(phi)
		for i in range(0, lm+1):
			dp_i0 = [ sqrt(i*(i+1) / 2) * P[i][1] ]  # j=0
			dp_ij = [ sqrt((i-j)*(i+j+1)) * P[i][j+1] - j*tan_phi * P[i][j] for j in range(1, i+1) ]
			dp_i0.extend(dp_ij); deri_P.append(np.array(dp_i0))
		return np.array(deri_P)
		
		
	def nonspher_Gvec(self, r_sat, time_jd, miu=MIU_M, Re=RM, lm=30):
		'''计算中心天体的非球形引力加速度，single-time，矢量化版本，输入 km, 输出 m/s^2, 大约快20%'''
		HL = self.moon_Cbi(time_jd)	# 月惯系到月固系的方向余弦矩阵 3*3
		r_fixed = np.dot(HL, r_sat)		# 应该在固连系下建立，王正涛
		r_norm = np.linalg.norm(r_fixed, 2)
		phi, lamda = atan(r_fixed[2] / sqrt(r_fixed[0]**2+r_fixed[1]**2)), atan(r_fixed[1]/r_fixed[0])
		spher2rect = np.array([ [cos(phi)*cos(lamda), cos(phi)*sin(lamda), sin(phi)], \
					[(-1/r_norm)*sin(phi)*cos(lamda), (-1/r_norm)*sin(phi)*sin(lamda),  (1/r_norm)*cos(phi)], \
					[(-1/r_norm)*sin(lamda)/cos(phi), (1/r_norm)*cos(lamda)/cos(phi), 0] ])	#球坐标到直角坐标 3*3
		P = self.legendre_spher_col(phi, lm)	# 勒让德函数
		dP = self.diff_legendre_spher(phi, P, lm)	# 勒让德函数的一阶导数
		const = Re/r_norm
		cos_m = np.array([ np.array([cos(j*lamda) for j in range(0, i+1)]) for i in range(0, lm+1) ])
		sin_m = np.array([ np.array([sin(j*lamda) for j in range(0, i+1)]) for i in range(0, lm+1) ])
		m_array = np.array([ np.array([j for j in range(0, i+1)]) for i in range(0, lm+1) ])
		Vr = np.sum([ (i+1)*const**i * np.dot(P[i][:-1], C[i] * cos_m[i] + S[i] * sin_m[i])  for i in range(2, lm+1) ])
		Vphi = np.sum([ const**i * np.dot(dP[i], C[i] * cos_m[i] + S[i] * sin_m[i]) for i in range(2, lm+1) ])
		Vlamda = np.sum([ const**i * np.dot(m_array[i] * P[i][:-1], -C[i] * sin_m[i] + S[i] * cos_m[i])  for i in  range(2, lm+1) ])
		g1 = np.array([-miu/r_norm**2*Vr, miu/r_norm*Vphi, miu/r_norm*Vlamda])	# 非球形引力 3*1
		g1 = np.dot(g1, spher2rect)	 # 球坐标到直角坐标，乘积顺序不要反了
		g1 = np.dot(HL.T, g1)	# 将月固系下加速度转换到月惯系下
		return g1 * 1000
		
		
	def nonspher_matlab(self, r_sat, time_jd, miu=MIU_M, Re=RM, lm=30):
		HL = self.moon_Cbi(time_jd)	# 月惯系到月固系的方向余弦矩阵 3*3
		r_fixed = np.dot(HL, r_sat)	* 1000	# 转化单位 km 为 m 
		g1 = np.array( eng.gravitysphericalharmonic( matlab.double(r_fixed.tolist()), 'LP165P', 30.0, nargout=3 ) )
		g1 = np.dot(HL.T, g1)	
		return g1	# m/s^2
		
		
	def thirdSun(self, r_sat, time_jd):
		'''计算太阳对卫星产生的第三天体摄动加速度, single-time, 输入km, 输出m/s^2
		输入：UTC时间time_utc，卫星相对中心天体的矢量，km, 均为np.array；
		输出：返回第三体太阳摄动加速度, m/s^2, np.array'''
		miu = MIU_S
		r_sun = self.moon2Sun(time_jd)
		norm_st = np.linalg.norm(r_sat-r_sun, 2)
		norm_et = np.linalg.norm(r_sun, 2)
		a_sun = -miu * ((r_sat-r_sun)/norm_st**3 + r_sun/norm_et**3)
		return a_sun * 1000
		
		
	def thirdEarth(self, r_sat, time_jd):
		'''计算地球对卫星产生的第三天体摄动加速度, single-time, 输入km, 输出m/s^2
		输入：UTC时间time_utc，卫星相对中心天体的矢量，m, 均为np.array;
		输出：返回第三体地球摄动加速度, km/s^2, np.array'''
		miu = MIU_E
		r_earth = self.moon2Earth(time_jd)
		norm_st = np.linalg.norm(r_sat-r_earth, 2)
		norm_et = np.linalg.norm(r_earth, 2)
		a_earth = -miu * ((r_sat-r_earth)/norm_st**3 + r_earth/norm_et**3)
		return a_earth * 1000
			

	def solarPress(self, r_sat, time_jd):
		'''计算太阳光压摄动, single-time, 输入 km, 输出 m/s^2
		输入：卫星位置矢量，m, np.array;	儒略日时间(float);
		输出：返回太阳光压在r_sat处造成的摄动, m / s^2, np.array'''
		global STEP, beta
		# 计算太阳光压摄动中的系数，使用随机漫步过程模拟
		v = np.random.normal(0, 1)	# 0均值，协方差为1的高斯白噪声序列
		sigma = 9.18e-5	# 给定的仿真参数
		beta = beta + sigma*sqrt(STEP)*v
		r_sun = self.moon2Sun(time_jd)		#月球->太阳矢量, km
		ro = 4.5605e-6	#太阳常数
		delta_0 = 1.495978707e+8	#日地距离 km
		delta = r_sat - r_sun
		norm_delta = np.linalg.norm(delta, 2)
		F = beta * ro*(delta_0**2/norm_delta**2) * delta/norm_delta
		return F	# m/s^2
			
	
	def twobody_dynamic(self, t, RV,  miu=MIU_M):
		'''二体动力学方程，仅考虑中心引力u
		输入：	时间t，从0开始（好像不太重要）; 	惯性系下位置速度	km, km/s, np.array
		输出：	返回d(RV)/dt，动力学增量, np.array'''
		R_Norm = sqrt(np.dot(RV[:3], RV[:3]))
		drdv = []
		drdv.extend(RV[3:])
		drdv.extend(-miu*(RV[:3]) / R_Norm**3)
		return np.array(drdv)
		
	@fn_timer	
	def complete_dynamic(self, t, RV, miu=MIU_M, Re=RM, lm=30):
		'''系统完整动力学模型, 暂考虑中心引力，非球形引力，第三天体引力
		输入：	时间t，从0开始（好像不太重要）; 	惯性系下位置速度	km, km/s, np.array
		输出：	返回d(RV)/dt，动力学增量, np.array, 1*6, m/s, m/s^2'''
		global time_utc
		time_jd = time_utc.to_julian_date()
		R, V = RV[:3], (RV[3:]*1000).tolist()	# R保持km, V转换为m/s
		F0 = self.centreGravity(R, miu)
		F1 = self.nonspher_Gvec(R, time_jd, miu, Re, lm)
		F2 = self.thirdEarth(R, time_jd)
		F3 = self.thirdSun(R, time_jd)
		F4 = self.solarPress(R, time_jd)	# 加入太阳光压摄动
		# print("F0: ", F0, "\n", "F1: ", F1, "\n", "F2: ", F2, "\n", "F3: ", F3, "\n", "F4: ", F4, "\n\n")
		F = F0 + F1 + F2 + F3 + F4	# m/s^2
		V.extend(F)
		return np.array(V)
		
	@fn_timer	
	def integrate_orbit(self, rv_0, num):
		'''数值积分器，使用RK45获得卫星递推轨道'''
		global STEP
		ode_y = solve_ivp( self.complete_dynamic, (0,STEP*num), rv_0, method="RK45", rtol=1e-3, atol=1e-6, t_eval=range(0,STEP*num, STEP) ).y
		return ode_y


		
		
		
if __name__ == "__main__":
	# import matlab.engine
	# eng = matlab.engine.start_matlab()
	
	ob = Orbit()
	number = 10
	data = pd.read_csv("STK/Moon_HPOP.csv", nrows=number)	# 取前number个点进行试算
	del data["Time (UTCG)"]
	RV_array = data.values
	r_array = data[['x (m)', 'y (m)', 'z (m)']].values
	utc_list = (ob.generate_time(start_t="20180101", end_t="20180331"))[:number]
	r_sat, RV, time_utc = r_array[0], RV_array[0], utc_list[0]
	time_jd = time_utc.to_julian_date()
	
	[ ob.complete_dynamic(RV/1000) for RV in RV_array[:4] ]
	# r_sat, time_utc = r_array[1], utc_list[1]
	# r_sat = matlab.double(r_sat.tolist())
	# gravity = np.array( eng.gravitysphericalharmonic(r_sat, 'LP165P', 30.0, nargout=3) )
	# HL = ob.moon_Cbi(time_utc)
	# print(np.dot(HL.T, gravity))

	# eng.quit()