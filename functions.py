# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import re
from math import *
from datetime import datetime
from jplephem.spk import SPK
from pprint import pprint


MIU_E, MIU_M, MIU_S = 3.986004415e+14, 4.902801056e+12, 1.327122e+20	#引力系数
RE, RM, RS = 6378136.3, 1.738e+06, 695508000.0		#天体半径
CSD_MAX, CSD_EPS, number = 1e+30, 1e-10, 495
STEP = 600		#全局积分步长
kernel = SPK.open(r"..\de421\de421.bsp")
df = pd.read_csv("STK/LP165P.grv", sep="\t", header=None)


import time
from functools import wraps
import random 
 
def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print ("%s running: %s seconds" %
               (function.__name__, str(t1-t0))
               )
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
		
		
	def legendre_spher_row(self, phi, l=30, m=30):
		'''计算完全规格化缔合勒让德函数，球坐标形式，张飞-利用函数计算重力场元(公式2.17)，标准向前行递推算法
		输入：地心纬度phi, rad;		阶次l和m
		输出：可用于向量直接计算的勒让德函数，包含P_00项	list'''
		P = [ np.array([1, 0]), np.array([sqrt(3)*sin(phi), sqrt(3)*cos(phi), 0]) ]
		tan_phi = tan(phi)
		for i in range(2, l+1):
			p_ij = list(np.zeros(i))
			p_ii = [ sqrt( (2*i+1)/(2*i) ) * cos(phi) * P[i-1][i-1], 0 ]
			p_ij.extend(p_ii)
			for j in range(i-1, 0, -1):
				g, h = 2*(j+1) / sqrt((i+j+1)*(i-j)), sqrt( (i+j+2)*(i-j-1)/((i+j+1)*(i-j)) )
				p_ij[j] =  g * tan_phi * p_ij[j+1] -  h * p_ij[j+2]
			p_ij[0] = 1/sqrt(2) * ( 2/sqrt((i+1)*i) * tan_phi * p_ij[1] - sqrt((i+2)*(i-1)/((i+1)*i)) * p_ij[2] )
			P.append(np.array(p_ij))
		return P
		
		
	def legendre_spher_col_1(self, theta, lm=30):
		'''计算完全规格化缔合勒让德函数，球坐标形式，张飞-利用函数计算重力场元(公式2.14)，标准向前列递推算法
		输入：地心余纬theta, rad;		阶次lm
		输出：可用于向量直接计算的勒让德函数，包含P_00项	np.array'''
		P = [ np.array([1, 0]), np.array([sqrt(3)*cos(theta), sqrt(3)*sin(theta), 0]) ]
		for i in range(2, lm+1):	# P[0][0] -- P[30][30]存在，P[i][i+1]均为0
			p_ij = [ sqrt( (4*i**2-1) / (i**2-j**2) ) * cos(theta) * P[i-1][j] - \
					sqrt( (2*i+1)*(i+j-1)*(i-j-1) / ((i**2-j**2)*(2*i-3)) ) * P[i-2][j] for j in range(i) ]
			p_ii = sqrt( (2*i+1)/(2*i) ) * sin(theta) * P[i-1][i-1]
			p_ij.extend([p_ii, 0])
			P.append(np.array(p_ij))
		return np.array(P)
		
		
	def legendre_cart_1(self, r_fixed, Re=RM, l=30, m=30):
		'''计算缔合勒让德函数，直角坐标形式，王正涛-卫星跟踪卫星测量确定地球重力场(公式4-2-5)
		输入：月固系下的卫星位置矢量, r_fixed, 		np.array
		输出：直角坐标下的勒让德函数，包含0阶项, 	list'''
		X, Y, Z = r_fixed[0], r_fixed[1], r_fixed[2]
		r = np.linalg.norm(r_fixed, 2)
		E = [ np.array([Re/r, 0]), np.array([ sqrt(3)*Z*Re**2/r**3, sqrt(3)*X*Re**2/r**3, 0]) ]
		F = [ np.array([0, 0]), np.array([ 0, sqrt(3)*Y*Re**2/r**3, 0]) ]
		cons_x, cons_y, cons_z, const = X*Re/r**2, Y*Re/r**2, Z*Re/r**2, (Re/r)**2
		for i in range(2, l+2):
			Eij = [ sqrt( (4*i**2-1) / (i**2-j**2) ) * cons_z * E[i-1][j] - \
					sqrt( (2*i+1)*(i-j-1)*(i+j-1) / ((i**2-j**2)*(2*i-3)) ) * const * E[i-2][j] for j in range(i) ]
			Eii = sqrt((2*i+1) / (2*i)) * ( cons_x * E[i-1][i-1] - cons_y * F[i-1][i-1] )
			Eij.extend([Eii, 0]); E.append(np.array(Eij))
			
			Fij = [ sqrt( (4*i**2-1) / (i**2-j**2) ) * cons_z * F[i-1][j] - \
					sqrt( (2*i+1)*(i-j-1)*(i+j-1) / ((i**2-j**2)*(2*i-3)) ) * const * F[i-2][j] for j in range(i) ]
			Fii = sqrt((2*i+1) / (2*i)) * ( cons_x * F[i-1][i-1] + cons_y * E[i-1][i-1] ) # 钟波/王庆宾此处为加号，王正涛此处为减号
			Fij.extend([Fii, 0]); F.append(np.array(Fij))
		return ( E, F )
		
		
	def legendre_cart_1(self, r_fixed, Re=RM, lm=30):
		'''计算缔合勒让德函数，直角坐标形式，钟波-基于GOCE卫星(公式2.2.12)
		输入：月固系下的卫星位置矢量, r_fixed, 		np.array
		输出：直角坐标下的勒让德函数，包含0阶项, 	np.array'''
		X, Y, Z = r_fixed[0], r_fixed[1], r_fixed[2]
		r = np.linalg.norm(r_fixed, 2)
		V = [ np.array([Re/r, 0]), np.array([ sqrt(3)*Z*Re**2/r**3, -sqrt(3)*X*Re**2/r**3, 0]) ]	# 增加负号
		W = [ np.array([0, 0]), np.array([ 0, -sqrt(3)*Y*Re**2/r**3, 0]) ]		# 增加负号
		cons_x, cons_y, cons_z, const = X*Re/r**2, Y*Re/r**2, Z*Re/r**2, (Re/r)**2
		for i in range(2, lm+2):
			Vij = [ sqrt( (4*i**2-1) / (i**2-j**2) ) * cons_z * V[i-1][j] - \
					sqrt( (2*i+1)*(i-j-1)*(i+j-1) / ((i**2-j**2)*(2*i-3)) ) * const * V[i-2][j] for j in range(i) ]
			Vii = -sqrt( (2*i+1) / (2*i) ) * ( cons_x * V[i-1][i-1] - cons_y * W[i-1][i-1] )	# 增加负号
			Vij.extend([Vii, 0]); V.append(np.array(Vij))
			
			Wij = [ sqrt( (4*i**2-1) / (i**2-j**2) ) * cons_z * W[i-1][j] - \
					sqrt( (2*i+1)*(i-j-1)*(i+j-1) / ((i**2-j**2)*(2*i-3)) ) * const * W[i-2][j] for j in range(i) ]
			Wii = -sqrt( (2*i+1) / (2*i) ) * ( cons_x * W[i-1][i-1] + cons_y * V[i-1][i-1] )	#钟波/王庆宾此处为加号 # 增加负号
			Wij.extend([Wii, 0]); W.append(np.array(Wij))
		return (np.array(V), np.array(W))
		
		
		# test
	def cart_define(self, r_fixed, Re=RM, lm=30):
		'''按定义计算E, F, 测试球坐标Plm和直角坐标E, F是否等价'''
		r = np.linalg.norm(r_fixed, 2)
		phi, lamda = atan(r_fixed[2] / sqrt(r_fixed[0]**2+r_fixed[1]**2)), atan(r_fixed[1]/r_fixed[0])
		P = self.legendre_spher_col(phi, lm=31)
		V, W = [], []
		for i in range(0, lm+2):
			const = (Re/r)**(i+1)
			Vij = np.array([ P[i][j] * cos(j*lamda) for j in range(i+1) ] + [0]) * const
			Wij = np.array([ P[i][j] * sin(j*lamda) for j in range(i+1) ] + [0]) * const
			V.append(Vij); W.append(Wij)
		V, W = np.array(V), np.array(W)
		return V, W	
	# test
		
		
	def diff_legendre_row(self, phi, P, l=30, m=30):
		'''计算完全规格化缔合勒让德函数的一阶导数，标准向前 行递推(雷伟伟_完全规格化_2016), Plm'(cos(theta))
		输入：地心纬度phi， 勒让德函数P
		输出：一阶导数，包含dP_00项		list'''
		dP, tan_phi = [], tan(phi)
		for i in range(0, l+1):
			dP_i0 = [ -sqrt((i+1)*i / 2) * P[i][1] ]	# j=0时公式不同，单独计算
			dP_ij = [ j*tan_phi * P[i][j] - sqrt((i+j+1)*(i-j)) * P[i][j+1] for j in range(1, i+1) ]
			dP_i0.extend(dP_ij); dP.append(np.array(dP_i0))
		return dP
		
		
	def diff_legendre_col(self, phi, P, l=30, m=30):
		'''计算完全规格化缔合勒让德函数的一阶导数，标准向前 列递推(雷伟伟_完全规格化_2016), Plm'(cos(theta))
		输入：地心纬度phi， 勒让德函数P
		输出：一阶导数，包含dP_00项		list'''
		dP, tan_phi, cos_phi = [], tan(phi), cos(phi)
		for i in range(0, l+1):
			temp = (2*i+1)/(2*i-1)
			dP_ij = [ i*tan_phi * P[i][j] - sqrt((i-j)*(i+j)*temp) / cos_phi * P[i-1][j] for j in range(0, i+1) ]
			dP.append(np.array(dP_ij))
		return dP
		
		
	def nonspherGravity_1(self, r_sat, time_utc, miu=MIU_M, Re=RM, l=30, m=30):
		'''计算中心天体的非球形引力加速度，single-time, 刘晓刚-GOCE卫星(公式2.3.6)'''
		HL = self.moon_Cbi(time_utc)	# 月惯系到月固系的方向余弦矩阵 3*3
		r_fixed = np.dot(HL, r_sat)		# 应该在固连系下建立，王正涛
		r_norm = np.linalg.norm(r_fixed, 2)
		phi, lamda = atan(r_fixed[2] / sqrt(r_fixed[0]**2+r_fixed[1]**2)), atan(r_fixed[1]/r_fixed[0])
		spher2rect = np.array([ [cos(phi)*cos(lamda), cos(phi)*sin(lamda), sin(phi)], \
					[(-1/r_norm)*sin(phi)*cos(lamda), (-1/r_norm)*sin(phi)*sin(lamda),  (1/r_norm)*cos(phi)], \
					[(-1/r_norm)*sin(lamda)/cos(phi), (1/r_norm)*cos(lamda)/cos(phi), 0] ])	#球坐标到直角坐标 3*3
		P = self.legendre_spher_col(phi, l, m)	# 勒让德函数
		dP = self.diff_legendre_spher(phi, P, l, m)
		tan_phi, cos_phi = tan(phi), cos(phi); 
		C, S = self.readCoffients(number=495, n=l)	# 包括0阶项
		Vr, Vphi, Vlamda, const = 0, 0, 0, Re/r_norm
		for i in range(0, l):
			temp_r, temp = i+1, const**(i+2)
			for j in range(0, i+1):
				Vr += temp_r * temp * ( C[i][j]*cos(j*lamda) + S[i][j]*sin(j*lamda) ) * P[i][j]
				Vphi += temp * ( C[i][j]*cos(j*lamda) + S[i][j]*sin(j*lamda) ) * dP[i][j] * r_norm
				Vlamda += temp * j * ( -C[i][j]*sin(j*lamda) + S[i][j]*cos(j*lamda) ) * P[i][j] * r_norm
		g1 = miu/Re**2 * np.array([-Vr, Vphi, Vlamda])
		g1 = np.dot(g1, spher2rect)	 # 球坐标到直角坐标，乘积顺序不要反了
		g1 = np.dot(HL.T, g1)	# 将月固系下加速度转换到月惯系下
		return g1
		
		
	def nonspherGravity_2(self, r_sat, time_utc, miu=MIU_M, Re=RM, l=30, m=30):
		'''计算中心天体的非球形引力加速度，single-time, 钟波-基于GOCE卫星(公式2.2.4)
		输入：惯性系下卫星位置矢量r_sat，均为np.array;	utc时间(datetime);	miu默认为月球;
		输出：返回中心天体的引力加速度, np.array'''
		HL = self.moon_Cbi(time_utc)	# 月惯系到月固系的方向余弦矩阵 3*3
		r_fixed = np.dot(HL, r_sat)		# 应该在固连系下建立，王正涛
		r_norm = np.linalg.norm(r_fixed, 2)
		phi, lamda = atan(r_fixed[2] / sqrt(r_fixed[0]**2+r_fixed[1]**2)), atan(r_fixed[1]/r_fixed[0])
		spher2rect = np.array([ [cos(phi)*cos(lamda), cos(phi)*sin(lamda), sin(phi)], \
					[(-1/r_norm)*sin(phi)*cos(lamda), (-1/r_norm)*sin(phi)*sin(lamda),  (-1/r_norm)*cos(phi)], \
					[(-1/r_norm)*sin(lamda)/cos(phi), (1/r_norm)*cos(lamda)/cos(phi), 0] ])	#球坐标到直角坐标 3*3
		P = self.legendre_spher_col(phi, l, m)	# 勒让德函数包括0阶项
		dP = self.diff_legendre_spher(phi, P, l, m)
		C, S = self.readCoffients(number=495, n=l)	# 包括0阶项
		Vr, Vphi, Vlamda, const = 0, 0, 0, Re/r_norm
		for i in range(0, l):	# C, S, P均从0阶项开始
			temp_r, temp = -(i+1)/r_norm, const**(i+1)
			for j in range(0, i+1):
				Vr += temp_r * temp * ( C[i][j]*cos(j*lamda) + S[i][j]*sin(j*lamda) ) * P[i][j]
				Vphi += temp * ( C[i][j]*cos(j*lamda) + S[i][j]*sin(j*lamda) ) * dP[i][j] 	# check dP_lm!
				Vlamda += temp * ( S[i][j]*cos(j*lamda) - C[i][j]*sin(j*lamda) ) * j * P[i][j] 
		g1 = miu/Re * np.array([Vr, Vphi, Vlamda])	# 非球形引力 3*1
		g1 = np.dot(g1, spher2rect)	 # 球坐标到直角坐标，乘积顺序不要反了
		g1 = np.dot(HL.T, g1)	# 将月固系下加速度转换到月惯系下
		return g1

		
	def nonspherG_cart(self, r_sat, time_utc, miu=MIU_M, Re=RM, l=30, m=30):
		'''计算中心天体的非球形引力加速度，使用直角坐标形式，single-time，王正涛(公式4.3.9)'''
		HL = self.moon_Cbi(time_utc)	# 月惯系到月固系的方向余弦矩阵 3*3
		r_fixed = np.dot(HL, r_sat)		# 转换为月固系下的位置矢量
		E, F = self.legendre_cart_check(r_fixed, Re, l, m); E.pop(0); F.pop(0)	# 去除0阶项？？
		C, S = self.readCoffients(number=495, n=l)
		ax, ay, az = 0, 0, 0
		const = miu/Re**2
		for i in range(0, l):	# E,F均存在0阶项，直接从1阶项开始
			temp = (2*i+1) / (2*i+3)
			b1 = sqrt( (i+1)*(i+2)*temp/2 )
			ax += const * (-b1*E[i+1][1] * C[i][0])		# j=0时公式不同，单独计算
			ay += const * (-b1*F[i+1][1] * C[i][0])
			az += const * ( sqrt( (i+1)**2 * temp) * (-E[i+1][0]*C[i][0] - F[i+1][0]*S[i][0]) )	 # az需要j从0开始
			for j in range(1, i):
				b2 = sqrt( (i+j+1)*(i+j+2) * temp )
				b3 = sqrt( (i-j+1)*(i-j+2) * temp )
				b4 = sqrt( (i-j+1)*(i+j+1) * temp )
				ax += const/2 * ( b2 * (-E[i+1][j+1]*C[i][j] - F[i+1][j+1]*S[i][j]) + \
								  b3 * ( E[i+1][j-1]*C[i][j] + F[i+1][j-1]*S[i][j]) )
				ay += const/2 * ( b2 * (-F[i+1][j+1]*C[i][j] + E[i+1][j+1]*S[i][j] ) + \
								  b3 * (-F[i+1][j-1]*C[i][j] + E[i+1][j-1]*S[i][j] ) )
				az += const * ( b4 * (-E[i+1][j]*C[i][j] - F[i+1][j]*S[i][j]) )
		g1 = np.array([ax, ay, az])
		g1 = np.dot(HL.T, g1)	# 将月固系下加速度转换到月惯系下
		return g1
		
		
	def nonspherG_cart_1(self, r_sat, time_utc, miu=MIU_M, Re=RM, l=30, m=30):
		'''计算中心天体的非球形引力加速度，使用直角坐标形式，single-time，王正涛(公式4.3.9)'''
		HL = self.moon_Cbi(time_utc)	# 月惯系到月固系的方向余弦矩阵 3*3
		r_fixed = np.dot(HL, r_sat)		# 转换为月固系下的位置矢量
		E, F = self.legendre_cart(r_fixed, Re, l)	
		C, S = self.readCoffients(number=495, n=l)
		ax, ay, az = 0, 0, 0
		const = miu/Re**2
		for i in range(2, l):	# E,F均存在0阶项，直接从2阶项开始
			temp = (2*i+1) / (2*i+3)
			b1 = sqrt( (i+1)*(i+2)*temp/2 )
			ax += const * (-b1*E[i+1][1] * C[i][0])		# j=0时公式不同，单独计算
			ay += const * (-b1*F[i+1][1] * C[i][0])
			az += const * ( sqrt( (i+1)**2 * temp) * (-E[i+1][0]*C[i][0] - F[i+1][0]*S[i][0]) )	 # az需要j从0开始
			for j in range(1, i):
				b2 = sqrt( (i+j+1)*(i+j+2) * temp )
				b3 = sqrt( (i-j+1)*(i-j+2) * temp )
				b4 = sqrt( (i-j+1)*(i+j+1) * temp )
				ax += const/2 * ( b2 * (-E[i+1][j+1]*C[i][j] - F[i+1][j+1]*S[i][j]) + \
								  b3 * ( E[i+1][j-1]*C[i][j] + F[i+1][j-1]*S[i][j]) )
				ay += const/2 * ( b2 * (-F[i+1][j+1]*C[i][j] + E[i+1][j+1]*S[i][j] ) + \
								  b3 * (-F[i+1][j-1]*C[i][j] + E[i+1][j-1]*S[i][j] ) )
				az += const * ( b4 * (-E[i+1][j]*C[i][j] - F[i+1][j]*S[i][j]) )
		g1 = np.array([ax, ay, az])
		g1 = np.dot(HL.T, g1)	# 将月固系下加速度转换到月惯系下
		return g1
		
		
	def acceler_disturb(self, time_utc, r_sat, da_last, step=120):
		'''对高阶非球形引力做保守计算，使用一阶Gauss/Markov过程模拟，返回产生的加速度
		输入：time_utc(datetime)；卫星位置矢量r_sat； 步长 step  s；
		输出：返回当前时刻的扰动加速度 da  m/s**2	np.array'''
		r_norm = np.linalg.norm(r_sat, 2)
		tao, sigma = r_norm ** 1.5, r_norm ** (-3)	#不知道为什么缩放
		va = np.random.normal(0, 1, 3)# 0均值，协方差为1的高斯白噪声序列
		da = exp(-step/tao) * da_last + sigma*sqrt(1-exp(-2*step/tao)) * va
		return da
		
		
	def dynamic_model(self, RV, time_utc, da_last, beta_last):
		'''系统完整动力学模型，包含月球引力，太阳和地球引力，太阳光压摄动，以及忽略的扰动加速度
		输入：位置和速度RV  m m/s； time_utc(datetime)
		输出：返回包括速度和加速度动力学模型'''
		DX = RV[:3]
		g0 = self.centreGravity(r_sat=RV[:3], miu=MIU_M, Re=1.738e+06)
		g1 = self.nonspherGravity(r_sat=RV[:3], time_utc=time_utc, miu=MIU_M, Re=1.738e+06, l=30, m=30)
		dirta_a = self.acceler_disturb(time_utc=time_utc, r_sat=RV[:3], da_last=da_last, step=120)
		a_sun = self.thirdSun(time_utc=time_utc, r_sat=RV[:3], miu=MIU_S)
		a_earth = self.thirdEarth(time_utc=time_utc, r_sat=RV[:3], miu=MIU_E)
		a_solar = self.solarPress(beta_last, time_utc, r_sat=RV[:3])
		a = g0 + g1 + dirta_a + a_sun + a_earth + a_solar
		DX.extend(a)
		return DX
		
	
		
		
		
if __name__ == "__main__":
	initial_rv = [6832.842724e3, 801.612273e3, 435.239952e3, -1.003679e3, 6.641038e3, 3.605789e3]
	initial_six = [6928.14e3, 0.00505, 28.5, 0, 0, 7.527*pi/180]
	ob = Orbit(rv = initial_rv)
	sixGeng = ob.rv2sixEle_Geng(initial_rv)
	sixZhang = ob.rv2sixEle_Zhang(initial_rv)
	time_list = ob.generate_time(start_t="20171231", end_t="20180131")
	Clm, Slm = ob.readCoffients()
	data = pd.read_csv("STK/Moon.csv")[:10]
	del data["Time (UTCG)"]
	data *= 1000
	data.columns = ['x (m)', 'y (m)', 'z (m)', 'vx (m/sec)', 'vy (m/sec)', 'vz (m/sec)']
	r_array = data[['x (m)', 'y (m)', 'z (m)']].values
	utc_list = ob.generate_time(start_t="20171231", end_t="20180101")
	r_sat, time_utc = r_array[1], utc_list[1]
	HL = ob.moon_Cbi(time_utc)
	r_fixed = np.dot(HL, r_sat)
	r = np.linalg.norm(r_fixed, 2)
	phi, lamda = atan(r_fixed[2] / sqrt(r_fixed[0]**2+r_fixed[1]**2)), atan(r_fixed[1]/r_fixed[0])
	P = ob.legendre_spher_col(phi, l=30, m=30)
	#print(dp1 - dp2)
	# a1 = [ np.linalg.norm(ob.nonspherGravity(r_sat, time_utc), 2) for (r_sat, time_utc) in zip(r_array, utc_list) ]
	# plt.plot(a1, label="a1 spher")
	a2 = [ np.linalg.norm(ob.nonspherGravity_check(r_sat, time_utc), 2) for (r_sat, time_utc) in zip(r_array, utc_list) ]
	plt.plot(a2, label="a2 spher")
	a3 = [ np.linalg.norm(ob.nonspherG_cart(r_sat, time_utc)) for (r_sat, time_utc) in zip(r_array, utc_list) ]
	plt.plot(a3, "--", label="a3 wang")
	a4 = [ np.linalg.norm(ob.nonspherG_cart_check(r_sat, time_utc), 2) for (r_sat, time_utc) in zip(r_array, utc_list) ]
	plt.plot(a4, "--", label="a4 zhong")
	plt.legend(); plt.show()
	