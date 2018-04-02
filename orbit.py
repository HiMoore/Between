# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import re
from math import *
from datetime import datetime, timedelta
from jplephem.spk import SPK
from scipy.integrate import solve_ivp, RK45, BDF, LSODA
from pprint import pprint

import matlab.engine
eng = matlab.engine.start_matlab()


MIU_E, MIU_M, MIU_S = 3.986004415e+14, 4.902801056e+12, 1.327122e+20	#引力系数
RE, RM, RS = 6378136.3, 1.738000e+06, 695508000.0		#天体半径
CSD_MAX, CSD_EPS, number = 1e+30, 1e-10, 495
STEP = 120		#全局积分步长
time_utc = pd.Timestamp("2018-1-1 00:00:00", freq=str(STEP)+"S")
beta = 0 	# 太阳光压参数
kernel = SPK.open(r"..\de421\de421.bsp")
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
		输入：儒略日时间															Timestamp
		输出：返回太阳相对于月球的位置矢量(J2000下), 单位 km,	[[x], [y], [z]]		np.array'''
		E_bc2M = kernel[3, 301].compute(time_jd)	# 地月中心 -> 月球
		S_bc2E_bc = kernel[0, 3].compute(time_jd)	# 太阳系中心 -> 地月中心
		S2S_bc = -kernel[0, 10].compute(time_jd)	# 太阳 -> 太阳系中心
		return -(S2S_bc + S_bc2E_bc + E_bc2M)
		
		
	def moon_Cbi(self, time_jd):
		'''返回月惯系到月固系的方向余弦矩阵, single-time，张巍-月球物理天平动对环月轨道(公式12)'''
		jd_day = time_jd - 2451545.0	# 自历元J2000起的儒略日数
		jd_t = jd_day / 36525.0		# 自历元J2000起的儒略世纪数
		l_moon = ((134.963413889 + 13.06499315537*jd_day + 0.0089939*jd_day**2) * (pi/180)) % (2*pi)
		l_sun = ((357.52910 + 35999.05030*jd_t - 0.0001559*jd_t**2 - 0.0000048*jd_t**3) * (pi/180)) % (2*pi)
		w_moon = ((318.308686110 - 6003.1498961*jd_t + 0.0124003*jd_t**2) * (pi/180)) % (2*pi) #需要化为弧度
		omega_moon = ((125.044555556 - 1934.1361850*jd_t + 0.0020767*jd_t**2) * (pi/180)) % (2*pi) #需要化为弧度
		kesai = ((l_moon + w_moon + omega_moon)) % (2*pi)
		tao_1 , tao_2, tao_3 = 2.9e-4, -0.58e-4, -0.87e-4	#张巍(公式3)
		tao = tao_1*sin(l_sun) + tao_2*sin(l_moon) + tao_3*sin(2*w_moon)
		I = 0.026917;
		phi_1, phi_2, phi_3 = -5.18e-4, 1.8e-4, -0.53e-4	#张巍(公式4)
		phi = asin(phi_1*sin(l_moon) + phi_2*sin(l_moon+2*w_moon) + phi_3*sin(2*l_moon+2*w_moon)) 
		theta_1, theta_2, theta_3= -5.1875e-4,  phi_2,  phi_3;
		theta = theta_1*cos(l_moon) + theta_2*cos(l_moon+2*w_moon) + theta_3*cos(2*l_moon+2*w_moon)
		Rx = lambda x: np.array([[1, 0, 0], [0, cos(x), sin(x)], [0, -sin(x), cos(x)]])
		Ry = lambda y: np.array([[cos(y), 0, -sin(y)], [0, 1, 0], [sin(y),0, cos(x)]])
		Rz = lambda z: np.array([[cos(z), sin(z), 0], [-sin(z), cos(z), 0], [0, 0, 1]])
		HL = np.dot( ( ( Rz(kesai+tao-phi).dot(Rx(-I-theta)) ).dot(Rz(phi)) ).dot(Rx(I)), Rz(omega_moon) )
		return HL
		
		
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
		'''计算中心天体的引力加速度，single-time, 单位 km/(s^2)np.array'''
		r_norm = np.linalg.norm(r_sat, 2)
		g0 = -miu*r_sat / r_norm**3		# 中心引力 3*1
		return g0

		
	def nonspherGravity(self, r_sat, time_utc, miu=MIU_M, Re=RM, lm=30):
		'''计算中心天体的非球形引力加速度，single-time, 王正涛-卫星跟踪卫星测量(公式2-4-7)
		输入：惯性系下卫星位置矢量r_sat，均为np.array;	utc时间(Timestamp);	miu默认为月球;
		输出：返回中心天体的引力加速度, np.array，与matlab比精度较差'''
		HL = self.moon_Cbi(time_utc)	# 月惯系到月固系的方向余弦矩阵 3*3
		r_fixed = np.dot(HL, r_sat)		# 应该在固连系下建立，王正涛
		r_norm = np.linalg.norm(r_fixed, 2)
		phi, lamda = atan(r_fixed[2] / sqrt(r_fixed[0]**2+r_fixed[1]**2)), atan(r_fixed[1]/r_fixed[0])
		spher2rect = np.array([ [cos(phi)*cos(lamda), cos(phi)*sin(lamda), sin(phi)], \
					[(-1/r_norm)*sin(phi)*cos(lamda), (-1/r_norm)*sin(phi)*sin(lamda),  (1/r_norm)*cos(phi)], \
					[(-1/r_norm)*sin(lamda)/cos(phi), (1/r_norm)*cos(lamda)/cos(phi), 0] ])	#球坐标到直角坐标 3*3
		P = self.legendre_spher_col(phi, lm)	# 勒让德函数
		tan_phi = tan(phi)
		global C, S
		Vr, Vphi, Vlamda, const = 0, 0, 0, Re/r_norm
		for i in range(2, lm):	# 王正涛i从2开始
			temp_r, temp = (i+1)*const**i, const**i
			Vr +=  C[i][0] * P[i][0] * temp_r		# j=0时dP不同，需要单独计算
			Vphi += C[i][0] * ( sqrt(i*(i+1)/2) * P[i][1] ) * temp
			for j in range(1, i+1):
				Vr += ( C[i][j]*cos(j*lamda) + S[i][j]*sin(j*lamda) ) * P[i][j] * temp_r
				Vphi += ( C[i][j]*cos(j*lamda) + S[i][j]*sin(j*lamda) ) * \
						( sqrt((i-j)*(i+j+1)) * P[i][j+1] - j * tan_phi * P[i][j] ) * temp 	# check dP_lm!
				Vlamda += ( -C[i][j]*sin(j*lamda) + S[i][j]*cos(j*lamda) ) * P[i][j] * temp * j
		g1 = np.array([-miu/r_norm**2*Vr, miu/r_norm*Vphi, miu/r_norm*Vlamda])	# 非球形引力 3*1
		g1 = np.dot(g1, spher2rect)	 # 球坐标到直角坐标，乘积顺序不要反了
		g1 = np.dot(HL.T, g1)	# 将月固系下加速度转换到月惯系下
		return g1
		
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
		
		
	def nonspher_Gvec(self, r_sat, time_utc, miu=MIU_M, Re=RM, lm=30):
		'''计算中心天体的非球形引力加速度，single-time，矢量化版本，大约快20%，与matlab比精度较高'''
		HL = self.moon_Cbi(time_utc)	# 月惯系到月固系的方向余弦矩阵 3*3
		r_fixed = np.dot(HL, r_sat)		# 应该在固连系下建立，王正涛
		print(np.linalg.norm(r_fixed), '\t\t', time_utc)
		r_norm = np.linalg.norm(r_fixed, 2)
		phi, lamda = atan(r_fixed[2] / sqrt(r_fixed[0]**2+r_fixed[1]**2)), atan(r_fixed[1]/r_fixed[0])
		spher2rect = np.array([ [cos(phi)*cos(lamda), cos(phi)*sin(lamda), sin(phi)], \
					[(-1/r_norm)*sin(phi)*cos(lamda), (-1/r_norm)*sin(phi)*sin(lamda),  (1/r_norm)*cos(phi)], \
					[(-1/r_norm)*sin(lamda)/cos(phi), (1/r_norm)*cos(lamda)/cos(phi), 0] ])	#球坐标到直角坐标 3*3
		P = self.legendre_spher_col(phi, lm)	# 勒让德函数
		dP = self.diff_legendre_spher(phi, P, lm)	# 勒让德函数的一阶导数
		tan_phi, const = tan(phi), Re/r_norm
		global C, S
		cos_m = np.array([ np.array([cos(j*lamda) for j in range(0, i+1)]) for i in range(0, lm+1) ])
		sin_m = np.array([ np.array([sin(j*lamda) for j in range(0, i+1)]) for i in range(0, lm+1) ])
		m_array = np.array([ np.array([j for j in range(0, i+1)]) for i in range(0, lm+1) ])
		Vr = np.sum([ (i+1)*const**i * np.dot(P[i][:-1], C[i] * cos_m[i] + S[i] * sin_m[i])  for i in range(2, lm+1) ])
		Vphi = np.sum([ const**i * np.dot(dP[i], C[i] * cos_m[i] + S[i] * sin_m[i]) for i in range(2, lm+1) ])
		Vlamda = np.sum([ const**i * np.dot(m_array[i] * P[i][:-1], -C[i] * sin_m[i] + S[i] * cos_m[i])  for i in  range(2, lm+1) ])
		g1 = np.array([-miu/r_norm**2*Vr, miu/r_norm*Vphi, miu/r_norm*Vlamda])	# 非球形引力 3*1
		g1 = np.dot(g1, spher2rect)	 # 球坐标到直角坐标，乘积顺序不要反了
		g1 = np.dot(HL.T, g1)	# 将月固系下加速度转换到月惯系下
		return g1
		
		
	def nonspher_matlab(self, r_sat, time_utc, miu=MIU_M, Re=RM, lm=30):
		HL = self.moon_Cbi(time_utc)	# 月惯系到月固系的方向余弦矩阵 3*3
		r_fixed = np.dot(HL, r_sat)		# 应该在固连系下建立，王正涛
		print(np.linalg.norm(r_fixed), '\t\t', time_utc)
		g1 = np.array( eng.gravitysphericalharmonic( matlab.double(r_fixed.tolist()), 'LP165P', 30.0, nargout=3 ) )
		g1 = np.dot(HL.T, g1)
		return g1
		
		
	def thirdSun(self, r_sat, time_utc, miu=MIU_S):
		'''计算太阳对卫星产生的第三天体摄动加速度, single-time
		输入：UTC时间time_utc，卫星相对中心天体的矢量，m, 均为np.array；
		输出：返回第三体太阳摄动加速度, m/s^2, np.array'''
		r_sun = self.moon2Sun(time_utc)
		norm_st = np.linalg.norm(r_sat-r_sun, 2)
		norm_et = np.linalg.norm(r_sun, 2)
		a_sun = -miu * ((r_sat-r_sun)/norm_st**3 + r_sun/norm_et**3)
		return a_sun
		
		
	def thirdEarth(self, r_sat, time_utc, miu=MIU_E):
		'''计算地球对卫星产生的第三天体摄动加速度, single-time
		输入：UTC时间time_utc，卫星相对中心天体的矢量，m, 均为np.array;
		输出：返回第三体地球摄动加速度, m/s^2, np.array'''
		r_earth = self.moon2Earth(time_utc)
		norm_st = np.linalg.norm(r_sat-r_earth, 2)
		norm_et = np.linalg.norm(r_earth, 2)
		a_earth = -miu * ((r_sat-r_earth)/norm_st**3 + r_earth/norm_et**3)
		return a_earth
			

	def solarPress(self, r_sat, time_utc):
		'''计算太阳光压摄动, single-time
		输入：卫星位置矢量，m, np.array;	utc时间(Timestamp);
		输出：返回太阳光压在r_sat处造成的摄动, np.array'''
		global STEP, beta
		# 计算太阳光压摄动中的系数，使用随机漫步过程模拟
		v = np.random.normal(0, 1)	# 0均值，协方差为1的高斯白噪声序列
		sigma = 9.18e-5	# 给定的仿真参数
		beta = beta + sigma*sqrt(STEP)*v
		r_sun = self.moon2Sun(time_utc)		#月球->太阳矢量
		ro = 4.5605e-6	#太阳常数
		delta_0 = 1.495978707e+11	#日地距离
		delta = r_sat - r_sun
		norm_delta = np.linalg.norm(delta, 2)
		F = beta * ro*(delta_0**2/norm_delta**2) * delta/norm_delta
		return F
			
	
	def twobody_dynamic(self, t, RV,  MIU=MIU_M):
		'''二体动力学方程，仅考虑中心引力u
		输入：	时间t，从0开始（好像不太重要）; 	惯性系下位置速度	m, m/s, np.array
		输出：	返回d(RV)/dt，动力学增量, np.array'''
		R_Norm = sqrt(np.dot(RV[:3], RV[:3]))
		drdv = []
		drdv.extend(RV[3:])
		drdv.extend(-MIU*(RV[:3]) / R_Norm**3)
		return np.array(drdv)
		
		
	def complete_dynamic(self, t, RV, miu=MIU_M, Re=RM, lm=30):
		'''系统完整动力学模型, 暂考虑中心引力，非球形引力，第三天体引力
		输入：	时间t，从0开始（好像不太重要）; 	惯性系下位置速度	m, m/s, np.array
		输出：	返回d(RV)/dt，动力学增量, np.array,  1*6'''
		global time_utc
		utc_local = time_utc
		R, V = RV[:3], RV[3:].tolist()
		F0 = self.centreGravity(R, miu)
		F1 = self.nonspher_Gvec(R, utc_local, miu, Re, lm)
		F2 = self.thirdSun(R, utc_local) + self.thirdEarth(R, utc_local)
		F3 = self.solarPress(R, utc_local)	# 加入太阳光压摄动
		F = F0 + F1 + F2 + F3
		V.extend(F)
		return np.array(V)
		
	@fn_timer	
	def integrate_orbit(self, rv_0, num):
		'''数值积分器，使用RK45获得卫星递推轨道'''
		global STEP
		ode_y = solve_ivp( self.complete_dynamic, (0,STEP*num), rv_0, method="RK45", rtol=1e-10, atol=1e-13, t_eval=range(0,STEP*num, STEP) ).y
		return ode_y

			
	def partial_nonspher(self, r_sat, time_utc, miu=MIU_M, Re=RM, lm=30):
		'''计算非球形引力加速度 对 位置 的偏导数矩阵，王正涛-卫星跟踪卫星(4-3-12), 对速度为0阵'''
		HL = self.moon_Cbi(time_utc)	# 月惯系到月固系的方向余弦矩阵 3*3
		r_fixed = np.dot(HL, r_sat)		# 转换为月固系下的位置矢量
		E, F = self.legendre_cart(r_fixed, Re, lm+1)
		C, S = self.readCoffients(number=495, n=lm)
		const = miu / Re**3
		da_xx, da_xy, da_xz, da_yz = 0, 0, 0, 0
		for i in range(2, lm+1):
			temp = (2*i+1) / (2*i+5)
			d1 = sqrt( (i+1)*(i+2)*(i+3)*(i+4) * temp/2 )
			d2 = sqrt( (i+1)**2 * (i+2)**2 * temp )
			d3 = sqrt( (i+2)*(i+3)*(i+4)*(i+5) * temp/2 )
			d4 = sqrt( i*(i+1)*(i+2)*(i+3) * temp )
			d8 = sqrt( (i+1)**2 * (i+2)*(i+3) * temp/2 )
			# j=0和j=1时, ax对X的偏导
			da_xx += const/2 * ( d1 * (E[i+2][2] * C[i][0]) - d2 * (E[i+2][0] * C[i][0]) ) 
			da_xx += const/4 * ( d3 * (E[i+2][3] * C[i][1] + F[i+2][3] * S[i][1]) + \
								d4 * (-3*E[i+2][1] * C[i][1] - F[i+2][1] * S[i][1]) )
			# j=0和j=1时, ax对Y的偏导, 也是ay对X的偏导		
			da_xy += const/2 * ( d1 * (F[i+2][2] * C[i][0]) )
			da_xy += const/4 * ( d3 * (F[i+2][3] * C[i][1] - E[i+2][3] * S[i][1]) + \
								d4 * (-F[i+2][3] * C[i][1] - E[i+2][1] * S[i][1]) )
			# j=0和j=1时, ax对Z的偏导, 也是az对X的偏导	
			da_xz += const * ( d8 * (E[i+2][1] * C[i][0]) )		# j=0时, ax对Z的偏导
			d9_1 = sqrt( i*(i+2)*(i+3)*(i+4) * temp/4 )
			d10_1 = sqrt( i*(i+1)*(i+2)**2 / (2*temp) )
			da_xz += const * ( d9_1 * (E[i+2][2] * C[i][1] + F[i+2][2] * S[i][1]) + \
							d10_1 * (-E[i+2][0] * C[i][1] - F[i+2][0] * S[i][1]) )		# j=1时, ax对Z的偏导
			# j=0和j=1时, ay对Z的偏导, 也是az对Y的偏导	
			da_yz += const * ( d8 * (F[i+2][1] * C[i][0]) )
			da_yz += const * ( d9_1 * (F[i+2][1] * C[i][1] - E[i+2][2] * S[i][1]) + 
							d10_1 * (F[i+2][0] * C[i][0] - E[i+2][0] * S[i][1]) )
			for j in range(2, i):
				d5 = sqrt( (i+j+1)*(i+j+2)*(i+j+3)*(i+j+4) * temp )
				d6 = sqrt( (i-j+1)*(i-j+2)*(i+j+1)*(i+j+2) * temp )
				d7 = sqrt( (i-j+1)*(i-j+2)*(i-j+3)*(i-j+4)*(i+j+1)*(i+j+2) * temp ) if j != 2 \
						else sqrt( (i+3)*(i+4) / ((i+1)*(i+2)) * temp )
				d9 = sqrt( (i-j+1)*(i+j+1)*(i+j+2)*(i+j+3) * temp/4 )
				d10 = sqrt( (i-j+1)*(i-j+2)*(i-j+3)*(i-j+4)*(i+j+1) * temp / 4 )
				da_xx += const/4 * ( d5 * (E[i+2][j+2] * C[i][j] + F[i+2][j+2] * S[i][j]) + \
								   2*d6 * (-E[i+2][j] * C[i][j] - F[i+2][j] * S[i][j]) + \
								   2*d7 * (E[i+2][j-2] * C[i][j] - F[i+2][j-2] * S[i][j]) )
				da_xy += const/4 * ( d5 * (F[i+2][j+2] * C[i][j] - E[i+2][j+2] * S[i][j]) + \
								   2*d7 * (-F[i+2][j-2] * C[i][j] + F[i+2][j-2] * S[i][j]) )
				da_xz += const * ( d9 * (E[i+2][j+1] * C[i][j] + F[i+2][j+1] * S[i][j]) + \
								d10 * (-E[i+2][j-1] * C[i][j] - F[i+2][j-1] * S[i][j]) )
				da_yz += const * ( d9 * (F[i+2][j+1] * C[i][j] - E[i+2][j+1] * S[i][j]) + \
								d10 * (F[i+2][j-1] * C[i][j] - E[i+2][j-1] * S[i][j]) )
			da_zz = np.sum( np.array([ const * (sqrt( (i-j+1)*(i-j+2)*(i+j+1)*(i+j+2) * temp ) * \
							(E[i+2][j] * C[i][j] + F[i+2][j] * S[i][j])) for j in range(0, i) ]) )
		A = np.array([ [da_xx, da_xy, da_xz], [da_xy, 0, da_yz], [da_xz, da_yz, da_zz] ])
		A = np.dot( np.dot(HL.T, A), HL )	# 变换到惯性系下
		return A	# 3*3
			

	def partial_third(self, r_sat, time_utc):
		'''计算第三体引力摄动加速度 对 位置 的偏导数矩阵, 王正涛(3-3-18), 对速度为0阵'''
		m2e = self.moon2Earth(time_utc)
		m2s = self.moon2Sun(time_utc)
		l1, l2 = m2e - r_sat, m2s - r_sat
		l1_norm, l2_norm = np.linalg.norm(l1, 2), np.linalg.norm(l2, 2)
		global MIU_E, MIU_S
		A = - ( MIU_E / l1_norm**3 * (np.identity(3) - 3/l1_norm**2 * np.outer(l1, l1)) + \
				MIU_S / l2_norm**3 * (np.identity(3) - 3/l2_norm**2 * np.outer(l2, l2)) )
		return A	# 3*3
		
	
	def jacobian_single(self, r_sat, time_utc):
		'''计算惯性系下，单颗卫星状态的Jacobian矩阵'''
		partial = self.partial_nonspher(r_sat, time_utc) + self.partial_third(r_sat, time_utc) 	# partial 3*3
		low = np.hstack( (partial, np.zeros((3,3))) )	# low 3*6
		up = np.hstack( (np.zeros((3,3)), np.identity(3)) )	# up 3*6
		J = np.vstack( (up, low) )
		return np.array(J)

		
		
		
if __name__ == "__main__":
	import matlab.engine
	eng = matlab.engine.start_matlab()
	
	ob = Orbit()
	number = 720
	data = pd.read_csv("STK/Moon_HPOP.csv")[:number]	# 取前number个点进行试算
	del data["Time (UTCG)"]
	r_array = data[['x (m)', 'y (m)', 'z (m)']].values
	utc_list = (ob.generate_time(start_t="20180101", end_t="20180331"))[:number]
	r_sat, time_utc = r_array[1], utc_list[1]
	r_sat = matlab.double(r_sat.tolist())
	gravity = np.array( eng.gravitysphericalharmonic(r_sat, 'LP165P', 30.0, nargout=3) )
	HL = ob.moon_Cbi(time_utc)
	print(np.dot(HL.T, gravity))

	eng.quit()