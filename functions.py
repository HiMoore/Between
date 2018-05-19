# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import re
from math import *
from datetime import datetime
from jplephem.spk import SPK
from pprint import pprint
from orbit_predictor.keplerian import rv2coe
from orbit_predictor.angles import ta_to_M, M_to_ta


MIU_E, MIU_M, MIU_S = 3.986004415e+14, 4.902801056e+12, 1.327122e+20	#引力系数
RE, RM, RS = 6378136.3, 1.738e+06, 695508000.0		#天体半径
CSD_MAX, CSD_EPS, number = 1e+30, 1e-10, 495
STEP = 600		#全局积分步长
kernel = SPK.open(r"..\de421\de421.bsp")
df = pd.read_csv("STK/LP165P.grv", sep="\t", header=None)
for i in range(CSD_LM+1):
	A = [ sqrt( (2*i+1)*(2*i-1) / ((i+j)*(i-j)) ) for j in range(i) ]
	A.append(1); B.append(np.array(A))
	# PI_i0 = [ sqrt( factorial(i) / (factorial(i)*(2*i+1)) ) ]
	# PI_ij = [ sqrt( factorial(i+j) / (factorial(i-j)*2*(2*i+1)) ) for j in range(1, i+1) ]
	# PI_ij.append(1); PI_i0.extend(PI_ij); PI.append(np.array(PI_i0))


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
		
		
	def LimitUpLow(self, x, x_up, x_low):
		'''功能：变量限幅
		输入：变量, 变量上界, 变量下界
		输出：返回限幅后的变量'''
		if x > x_up:
			return x_up
		elif x < x_low:
			return x_low
		else:
			return x
		
	
	def sixEle2rv_Zhang(self, sixEle, MIU=MIU_E):
		'''功能：轨道六根数到惯性位置速度转换, single-time
		输入：六根数 sixEle[6]	single time
		输出：返回位置向量 r[3]		m;  速度向量 v[3]		m/s'''
		a, e, i, omiga, w, M = sixEle
		a = self.LimitUpLow(a, CSD_MAX, 1.0);
		e = self.LimitUpLow(e, 1-CSD_EPS, 0.0);
		f = self.M2f(M, e)
		u = w + f
		r = self.LimitUpLow(a*(1.0-e*e)/(1.0+e*cos(f)), CSD_MAX, 1.0)
		v = self.LimitUpLow(sqrt(MIU*abs(2/r - 1/a)), 3e8, 1.0)		#标量
		P = np.array([cos(w)*cos(omiga) - sin(w)*sin(omiga)*cos(i), cos(w)*sin(omiga) + sin(w)*cos(omiga)*cos(i), sin(w)*sin(i)])
		Q = np.array([-sin(w)*cos(omiga) - cos(w)*sin(omiga)*cos(i), -sin(w)*sin(omiga) + cos(w)*cos(omiga)*cos(i), cos(w)*sin(i)])
		r = list(r*cos(f)*P + r*sin(f)*Q)
		v = list(sqrt(MIU/a/(1-e**2)) * (-sin(f)*P + (e+cos(f))*Q))
		return [r, v]
		
		
	def rv2sixEle_Zhang(self, rv, MIU=MIU_E):
		'''功能：位置速度到轨道六根数惯性转换, single-time
		输入：r[3]	位置向量，m; 	v[3]	速度向量，m/s
		输出：SixE[6]：六根数'''
		r, v = np.array(rv[:3]), np.array(rv[3:])
		r_norm, v_norm = np.linalg.norm(r, 2), np.linalg.norm(v, 2)
		h, ux, uz = np.cross(r, v), np.array([1, 0, 0]), np.array([0, 0, 1])
		h_norm = np.linalg.norm(h, 2)
		E = v_norm**2 / 2 - MIU / r_norm
		line_nodes = np.cross(uz, h)
		N = line_nodes / np.linalg.norm(line_nodes, 2)
		i = acos( np.dot(h/h_norm, uz) )
		omiga = acos(np.dot(N, ux))
		p, a = h_norm**2 / MIU, -MIU / (2*E)
		e = np.sqrt(1-p/a)
		f = acos((p-r_norm) / (r_norm*e))
		w = acos(np.dot(r/r_norm, N)) - f
		M = self.f2M(f, e)
		return [a, e, i, omiga, w, M]
		
		
	def M2f(self, M, e):
		'''功能：平近点角到真近点角转换	See page77 of book<航天器轨道确定>
		输入：平近点角 M,  偏心率 e;	输出：返回真近点角 f'''
		e = self.LimitUpLow(e,0.999,0.0)
		N = int(5000 // (1+(1000-1000*e)))
		E = M
		for i in range(1, N):
			E = E + (M-E+e*sin(E)) / (1-e*cos(E))
		f = 2*atan(sqrt((1+e)/(1-e) * tan(E/2)))
		return f
		
		
	def f2M(self, f, e):
		'''功能：真近点角到平近点角转换
		输入：真近点角 f,  偏心率 e;	输出：返回平近点角 M'''
		tan_E = tan(f/2) / sqrt((1+e)/(1-e))
		E = atan(tan_E) * 2
		M = E - e*sin(E)
		return M
		
		
	def sixEle2rv_Geng(self, sixEle, MIU=MIU_E):
		'''功能：轨道六根数到惯性位置速度转换, single-time
		输入：六根数 sixEle[6]	single time
		输出：返回位置向量 r[3]		m;  速度向量 v[3]		m/s'''
		a, e, i, omiga, w, M = sixEle
		a = self.LimitUpLow(a, CSD_MAX, 1.0);
		e = self.LimitUpLow(e, 1-CSD_EPS, 0.0);
		f = self.M2f(M, e)
		u = w + f
		r = self.LimitUpLow(a*(1.0-e*e)/(1.0+e*cos(f)), CSD_MAX, 1.0)
		v = self.LimitUpLow(sqrt(MIU*abs(2/r - 1/a)), 3e8, 1.0)
		Vt = self.LimitUpLow(sqrt(MIU/r/r*a*(1.0-e*e)),3.0e8,1)
		asin_Vt = asin(Vt/v) if Vt<=v else pi/2
		if sin(f) < 0.0:
			gama = pi - asin_Vt
		else:
			gama = asin_Vt
		uv = u + gama
		vec_r = np.array([cos(omiga)*cos(u) - sin(omiga)*sin(u)*cos(i), sin(omiga)*cos(u) + cos(omiga)*sin(u)*cos(i), sin(u)*sin(i)])
		vec_v = np.array([cos(omiga)*cos(uv) - sin(omiga)*sin(uv)*cos(i), sin(omiga)*cos(uv) + cos(omiga)*sin(uv)*cos(i), sin(uv)*sin(i)])
		return [r*vec_r, v*vec_v]
		
	
	def rv2sixEle_Geng(self, rv, MIU=MIU_E):
		'''功能：位置速度到轨道六根数惯性转换, single-time
		输入：R[3]	位置向量，m;	V[3]	速度向量，m/s
		输出：SixE[6]：六根数'''
		r, v = np.array(rv[:3]), np.array(rv[3:])
		R = self.LimitUpLow(sqrt(np.dot(r, r)), CSD_MAX, 1.0)
		V = self.LimitUpLow(sqrt(np.dot(v, v)), CSD_MAX, 1.0)
		h = np.cross(r, v)	#角动量
		H = sqrt(np.dot(h, h))			#角动量范数
		Energy =  np.dot(v, v)/2 - MIU/R		#机械能
		a = self.LimitUpLow(R/(2-R*V**2/MIU), CSD_MAX, 1.0)
		esinE = np.dot(r, v) / sqrt(a*MIU)
		ecosE = 1 - R/a
		e = self.LimitUpLow(sqrt(esinE**2 + ecosE**2), 1-CSD_EPS, 0)
		E = self.Atan4(esinE, ecosE)
		f = 2.0*atan(sqrt((1.0+e)/(1.0-e))*tan(0.5*E))
		H = self.LimitUpLow(H, np.linalg.norm(h, 1)+CSD_EPS, abs(h[2])+CSD_EPS)		
		i = acos(self.divided(h[2], H))
		omiga = self.Atan4(h[0], -h[1])
		ksinu = self.divided(r[2], sin(i))
		kcosu = r[0]*cos(omiga) + r[1]*sin(omiga)
		u = self.Atan4(ksinu, kcosu)
		w = u - f
		M = E-esinE
		return [a, e, i, omiga, w, M]
		
		
	def divided(self, y, x):
		'''功能：除法'''
		if x<CSD_EPS and x>=0:
			return y/CSD_EPS
		elif x>-CSD_EPS and x<0:
			return -y/CSD_EPS
		else:
			return y/x
			
			
	def Atan4(self, y, x):
		'''功能：反正切'''
		return 0 if (abs(y)<CSD_EPS and abs(x)<CSD_EPS) \
				else atan2(y, x)
			
		
	def utc2JD(self, time_utc):
		'''由UTC时间计算对应的儒略日, 用于计算天体位置, single-time, 李博-基于星间定向观测的导航星座(公式2.8)
		输入： UTC时间								datetime					
		输出：返回UTC时间对应的儒略日时间			np.array'''
		a = floor(14 - time_utc.month)
		y = time_utc.year + 4800 - a
		m = time_utc.month + 12*a - 3
		JDN = time_utc.day + floor((153*m+2)/5) + 365*y + floor(y/4) - floor(y/100) + floor(y/400) - 32045
		JD = JDN - 0.5 + time_utc.hour/24 + time_utc.minute/1440 + time_utc.second/86400
		return JD
		
		
	def moon_Cbi(self, time_jd):
		'''返回月惯系到月固系的方向余弦矩阵, single-time，张巍-月球物理天平动对环月轨道(公式12)'''
		jd_day = time_jd - 2451545.0	# 自历元J2000起的儒略日数
		jd_t = jd_day / 36525.0		# 自历元J2000起的儒略世纪数
		l_moon = mod( radians(134.963413889 + 13.06499315537*jd_day + 0.0089939*jd_day**2), 2*pi )
		l_sun = mod( radians(357.52910 + 35999.05030*jd_t - 0.0001559*jd_t**2 - 0.0000048*jd_t**3), 2*pi )
		w_moon = mod( radians(318.308686110 - 6003.1498961*jd_t + 0.0124003*jd_t**2), 2*pi ) #需要化为弧度
		omega_moon = mod( radians(125.044555556 - 1934.1361850*jd_t + 0.0020767*jd_t**2), 2*pi) #需要化为弧度
		phi = mod(l_moon + w_moon + pi, 2*pi)
		I = 0.02691686;
		tao_1 , tao_2, tao_3 = 2.9e-4, -0.58e-4, -0.87e-4	#张巍(公式3)
		tao = tao_1*sin(l_sun) + tao_2*sin(l_moon) + tao_3*sin(2*w_moon)
		rho_1, rho_2, rho_3 = -5.18e-4, 1.8e-4, -0.53e-4	#张巍(公式4)
		rho = mod( rho_1*cos(l_moon) + rho_2*cos(l_moon+2*w_moon) + rho_3*cos(2*(l_moon+w_moon)), 2*pi )
		sigma = mod( asin( rho_1*sin(l_moon) + rho_2*sin(l_moon+2*w_moon) + rho_3*sin(2*(l_moon+w_moon)) ) / I, 2*pi )
		Rx = lambda x: np.array([[1, 0, 0], [0, cos(x), sin(x)], [0, -sin(x), cos(x)]])
		Ry = lambda y: np.array([[cos(y), 0, -sin(y)], [0, 1, 0], [sin(y),0, cos(x)]])
		Rz = lambda z: np.array([[cos(z), sin(z), 0], [-sin(z), cos(z), 0], [0, 0, 1]])
		HL = np.dot( ( ( Rz(phi+tao-sigma).dot(Rx(-I-rho)) ).dot(Rz(sigma)) ).dot(Rx(I)), Rz(omega_moon) )
		return HL
		
		
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
		
		
	def legendre_spher_col(self, phi, lm=CSD_LM):
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

		
	def legendre_unnormalize(self, phi, lm=CSD_LM):
		P = [ np.array([1,0]), np.array([sin(phi), cos(phi), 0]) ]
		for i in range(2, lm+1):
			P_i0 = [ ((2*i-1)*sin(phi)*P[i-1][0] - (i-1)*P[i-2][0]) / i ]
			P_ij = [ P[i-2][j] + (2*i-1)*cos(phi)*P[i-1][j-1] for j in range(1, i) ]
			P_ii = [ (2*i-1)*cos(phi)*P[i-1][i-1], 0 ]
			P_i0.extend(P_ij); P_i0.extend(P_ii); P.append(P_i0)
		P = np.array(P)
		return P
		
		
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
		
		
	def nonspherG_cart(self, r_sat, tdb_jd, miu=MIU_M, Re=RM, lm=CSD_LM):
		'''计算中心天体的非球形引力加速度，使用直角坐标形式，single-time，王正涛(公式4.3.9)'''
		I2F = self.moon_Cbi(tdb_jd)	# 月惯系到月固系的方向余弦矩阵 3*3
		r_fixed = np.dot(I2F, r_sat)		# 转换为月固系下的位置矢量
		E, F = self.legendre_cart(r_fixed, Re, lm); 	# 去除0阶项？？
		ax, ay, az = 0, 0, 0
		const = miu/Re**2
		for i in range(0, lm):	# E,F均存在0阶项，直接从1阶项开始
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
		g1 = np.dot(I2F.T, g1)	# 将月固系下加速度转换到月惯系下
		return g1
		
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
		输出：一阶导数，包含dP_00项		np.array'''
		dP, tan_phi = [], tan(phi)
		for i in range(0, l+1):
			dP_i0 = [ -sqrt((i+1)*i / 2) * P[i][1] ]	# j=0时公式不同，单独计算
			dP_ij = [ j*tan_phi * P[i][j] - sqrt((i+j+1)*(i-j)) * P[i][j+1] for j in range(1, i+1) ]
			dP_i0.extend(dP_ij); dP.append(np.array(dP_i0))
		return np.array(dP)
	
		
	def diff_legendre_col(self, phi, P, lm=CSD_LM):
		'''计算完全规格化缔合勒让德函数的一阶导数，标准向前 列递推(雷伟伟_完全规格化_2016), Plm'(cos(theta))
		输入：地心纬度phi， 勒让德函数P
		输出：一阶导数，包含dP_00项		np.array'''
		dP, tan_phi, cos_phi = [], tan(phi), cos(phi)
		for i in range(0, lm+1):
			temp = (2*i+1)/(2*i-1)
			dP_ij = [ -i*tan_phi * P[i][j] + sqrt((i-j)*(i+j)*temp) / cos_phi * P[i-1][j] for j in range(0, i+1) ]
			dP.append(np.array(dP_ij))
		return np.array(dP)
		
		
	def diff_legendre_spher(self, phi, P, lm=CSD_LM):
		'''计算完全规格化缔合勒让德函数的一阶导数，球坐标形式，王正涛-卫星跟踪卫星测量(公式2-4-6)
		输入：地心纬度phi， 勒让德函数P
		输出：一阶导数，包含dP_00项		np.array'''
		deri_P = []
		for i in range(0, lm+1):
			dp_i0 = [ sqrt(i*(i+1) / 2) * P[i][1] ]  # j=0
			dp_ij = [ sqrt((i-j)*(i+j+1)) * P[i][j+1] - j*tan(phi) * P[i][j] for j in range(1, i+1) ]
			dp_i0.extend(dp_ij); deri_P.append(np.array(dp_i0))
		return np.array(deri_P)
		
		
	def nonspher_moon(self, r_sat, tdb_jd, miu=MIU_M, Re=RM, lm=CSD_LM):
		'''计算月球的非球形引力加速度，single-time, Brandon A. Jones - Efficient Models for the Evaluation and Estimation(eq. 2.14)
		输入：惯性系下卫星位置矢量r_sat，np.array, 单位 km;		TDB的儒略日时间;	miu默认为月球;
		输出：返回非球形引力摄动加速度, np.array，单位 km/s^2'''
		I2F = self.moon_Cbi(tdb_jd)	# 月惯系到月固系的方向余弦矩阵 3*3
		r_fixed = np.dot(I2F, r_sat)		# 应该在固连系下建立，王正涛
		xy_norm, r_norm = np.linalg.norm(r_fixed, 2), np.linalg.norm(r_fixed[:2], 2)	# Keric A. Hill - Autonomous Navigation
		phi, lamda, tan_phi = asin(r_fixed[2] / r_norm), atan2(r_fixed[1], r_fixed[0]), r_fixed[2]/xy_norm	
		P = self.legendre_spher_alfs(phi, lm)	# 勒让德函数
		rRatio, pow_r2 = Re/r_norm, pow(r_norm, 2)
		dR_dr = r_fixed / r_norm	# 球坐标对直角坐标的偏导数, Keric Hill(eq. 8.10), Brandon Jones(eq. 2.13)
		dphi_dr = 1/xy_norm * np.array([ -r_fixed[0]*r_fixed[2]/pow_r2,  -r_fixed[1]*r_fixed[2]/pow_r2,  1 - pow(r_fixed[2], 2)/pow_r2 ])
		dlamda_dr = 1/pow(xy_norm, 2) * np.array([ -r_fixed[1], r_fixed[0], 0 ])
		cos_m = np.array([ cos(j*lamda) for j in range(0, lm+1) ])
		sin_m = np.array([ sin(j*lamda) for j in range(0, lm+1) ])
		dU_dr = -miu/pow_r2 * sum([ pow(rRatio, i)*(i+1) * np.dot(P[i][:-1], C[i]*cos_m[:i+1] + S[i]*sin_m[:i+1]) for i in range(2, lm+1) ]) 
		dU_dphi = miu/r_norm * sum([ pow(rRatio, i) * np.dot(P[i][1:] * pi_dot[i] -  tan_phi*M_Array[:i+1] * P[i][:-1], \
							C[i]*cos_m[:i+1] + S[i]*sin_m[:i+1]) for i in range(2, lm+1) ])
		dU_dlamda = miu/r_norm * sum([ pow(rRatio, i) * np.dot(M_Array[:i+1]*P[i][:-1], S[i]*cos_m[:i+1] - C[i]*sin_m[:i+1]) for i in range(2, lm+1) ])
		a_fixed = dR_dr * dU_dr + dphi_dr * dU_dphi + dlamda_dr * dU_dlamda
		a_inertial = np.dot(I2F.T, a_fixed)
		return a_inertial	# km/s^2
		

	def nonspherGravity(self, r_sat, tdb_jd, miu=MIU_M, Re=RM, lm=CSD_LM):
		'''计算中心天体的非球形引力加速度，single-time, 输入 km, 输出 m/s^2, 王正涛-卫星跟踪卫星测量(公式2-4-7)
		输入：惯性系下卫星位置矢量r_sat，均为np.array, 单位 km;	tdb的儒略时间, float;	miu默认为月球;
		输出：返回中心天体的引力加速度, np.array，单位 km/s^2'''
		I2F = self.moon_Cbi(tdb_jd)	# 月惯系到月固系的方向余弦矩阵 3*3
		r_fixed = np.dot(I2F, r_sat)		# 应该在固连系下建立，王正涛
		r_norm = np.linalg.norm(r_fixed, 2)
		xy_norm = np.linalg.norm(r_fixed[:2], 2)
		phi, lamda = atan2(r_fixed[2], xy_norm), atan2(r_fixed[1], r_fixed[0])
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
						( pi_dot[i][j] * P[i][j+1] - j * r_fixed[2]/xy_norm * P[i][j] ) * temp 	# check dP_lm!
				Vlamda += ( -C[i][j]*sin(j*lamda) + S[i][j]*cos(j*lamda) ) * P[i][j] * temp * j
		g1 = np.array([-miu/r_norm**2*Vr, miu/r_norm*Vphi, miu/r_norm*Vlamda])	# 非球形引力 3*1
		g1 = np.dot(g1, spher2rect)	 # 球坐标到直角坐标，乘积顺序不要反了
		g1 = np.dot(I2F.T, g1)	# 将月固系下加速度转换到月惯系下
		return g1	# km/s^2
		
		
	def nonspher_moongravity(self, r_sat, tdb_jd, miu=MIU_M, Re=RM, lm=CSD_LM):
		'''计算月球的非球形引力加速度，迭代版本, Brandon A. Jones - Efficient Models for the Evaluation and Estimation(eq. 2.14)'''
		I2F = self.moon_Cbi(tdb_jd)	# 月惯系到月固系的方向余弦矩阵 3*3
		r_fixed = np.dot(I2F, r_sat)		# 应该在固连系下建立，王正涛
		xy_norm, r_norm = np.linalg.norm(r_fixed, 2), np.linalg.norm(r_fixed[:2], 2)	# Keric A. Hill - Autonomous Navigation
		phi, lamda, tan_phi = asin(r_fixed[2] / r_norm), atan2(r_fixed[1], r_fixed[0]), r_fixed[2]/xy_norm	
		P = self.legendre_spher_alfs(phi, lm)	# 勒让德函数
		rRatio, pow_r2 = Re/r_norm, pow(r_norm, 2)
		dR_dr = r_fixed / r_norm	# 球坐标对直角坐标的偏导数, Keric Hill(eq. 8.10), Brandon Jones(eq. 2.13)
		dphi_dr = 1/xy_norm * np.array([ -r_fixed[0]*r_fixed[2]/pow_r2,  -r_fixed[1]*r_fixed[2]/pow_r2,  1 - pow(r_fixed[2], 2)/pow_r2 ])
		dlamda_dr = 1/pow(xy_norm, 2) * np.array([ -r_fixed[1], r_fixed[0], 0 ])
		dU_dr, dU_dphi, dU_dlamda = 0, 0, 0
		rRatio = Re / r_norm
		cos_m, sin_m = np.zeros(lm+2), np.zeros(lm+2)
		cos_m[0], cos_m[1], sin_m[0], sin_m[1] = 1, cos(lamda), 0, sin(lamda)
		for m in range(2, lm+2):
			cos_m[m] = 2*cos_m[1] * cos_m[m-1] - cos_m[m-2]
			sin_m[m] = 2*cos_m[1] * sin_m[m-1] - sin_m[m-2]
		for i in range(2, lm+1):
			rRatio_n = pow(rRatio, i)
			for j in range(i+1):
				dU_dr +=  P[i][j] * ( C[i][j] * cos_m[j] + S[i][j] * sin_m[j] ) * rRatio_n * (i+1)
				dU_dphi +=  ( P[i][j+1] * pi_dot[i][j] - j * r_fixed[2]/xy_norm * P[i][j] ) * \
							( C[i][j]*cos_m[j] + S[i][j]*sin_m[j] )* rRatio_n
				dU_dlamda += j * P[i][j] * ( S[i][j]*cos_m[j] - C[i][j]*sin_m[j] )* rRatio_n
		dU_dr, dU_dphi, dU_dlamda = -miu/r_norm**2*dU_dr, miu/r_norm*dU_dphi, miu/r_norm*dU_dlamda	# 非球形引力 3*1
		a_fixed = dR_dr * dU_dr + dphi_dr * dU_dphi + dlamda_dr * dU_dlamda
		a_inertial = np.dot(I2F.T, a_fixed)
		return a_inertial	# km/s^2
		
		
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
			
		
	def partial_centre_2(self, r_sat, miu=MIU_M):
		'''中心引力加速度对 (x, y, z) 的偏导数在月惯系下的表达, 单位 1/s^2'''
		r_norm = np.linalg.norm(r_sat, 2)
		x, y, z, pow_r2 = r_sat[0], r_sat[1], r_sat[2], pow(r_norm, 2)
		da_dr = miu/pow(r_norm, 3) * np.array([ [3*pow(x,2)/pow_r2 - 1, 	3*x*y/pow_r2, 			3*x*z/pow_r2], 
										 [3*x*y/pow_r2, 			3*pow(y,2)/pow_r2 - 1, 	3*y*z/pow_r2], 
										 [3*x*z/pow_r2, 			3*y*z/pow_r2, 			3*pow(z,2)/pow_r2 - 1] ])
		return da_dr
	
		
	def partial_third_2(self, r_sat, time_tdb):
		'''第三天体引力摄动加速度对 (x, y, z) 的偏导数在月惯系下的表达, 单位 1/s^2'''
		earth = self.moon2Earth(time_tdb)	# moon -> earth
		sun = self.moon2Sun(time_tdb)	# moon -> sun
		l_earth, l_sun = r_sat-earth, r_sat-sun
		norm_earth, norm_sun = np.linalg.norm(l_earth, 2), np.linalg.norm(l_sun, 2)
		pow_e2, pow_s2 = pow(norm_earth, 2), pow(norm_sun, 2)
		xe, ye, ze = l_earth; xs, ys, zs = l_sun
		earth_dadr = MIU_E/pow(norm_earth, 3) * np.array([ [3*pow(xe,2)/pow_e2 - 1, 	3*xe*ye/pow_e2, 	3*xe*ze/pow_e2],
														[3*xe*ye/pow_e2, 		3*pow(ye,2)/pow_e2 - 1, 	3*ye*ze/pow_e2],
														[3*xe*ze/pow_e2, 		3*ye*ze/pow_e2, 	3*pow(ze,2)/pow_e2 - 1] ])
		sun_dadr = MIU_S/pow(norm_sun, 3) * np.array([ [3*pow(xs,2)/pow_s2 - 1, 	3*xs*ys/pow_s2, 		3*xs*zs/pow_s2],
														[3*xs*ys/pow_s2, 		3*pow(ys,2)/pow_s2 - 1, 	3*ys*zs/pow_s2],
														[3*xs*zs/pow_s2, 		3*ys*zs/pow_s2, 			3*pow(zs,2)/pow_s2 - 1] ])
		return earth_dadr + sun_dadr
		
		
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
	