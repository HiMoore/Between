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
		R = self.LimitUpLow(sqrt(np.dot(self.r, self.r)), CSD_MAX, 1.0)
		V = self.LimitUpLow(sqrt(np.dot(self.v, self.v)), CSD_MAX, 1.0)
		h = np.cross(self.r, self.v)	#角动量
		H = sqrt(np.dot(h, h))			#角动量范数
		Energy =  np.dot(self.v, self.v)/2 - MIU/R		#机械能
		a = self.LimitUpLow(R/(2-R*V**2/MIU), CSD_MAX, 1.0)
		esinE = np.dot(self.r, self.v) / sqrt(a*MIU)
		ecosE = 1 - R/a
		e = self.LimitUpLow(sqrt(esinE**2 + ecosE**2), 1-CSD_EPS, 0)
		E = self.Atan4(esinE, ecosE)
		f = 2.0*atan(sqrt((1.0+e)/(1.0-e))*tan(0.5*E))
		H = self.LimitUpLow(H, np.linalg.norm(h, 1)+CSD_EPS, abs(h[2])+CSD_EPS)		
		i = acos(self.divided(h[2], H))
		omiga = self.Atan4(h[0], -h[1])
		ksinu = self.divided(self.r[2], sin(i))
		kcosu = self.r[0]*cos(omiga) + self.r[1]*sin(omiga)
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
			
		
	def orbit_rk4(self, RV, H=120, Time=0, MIU=MIU_E):
		'''四阶龙格库塔法递推位置和速度, single-time				单位
		输入：	月惯系下相对月球位置速度, np.array					m	m/s
				积分步长	s;		当前时刻																	
		输出：	下一时刻月惯系下卫星位置、速度, np.array			m	m/s'''
		DT = 0.5 * H
		Drv = self.twobody_dynamic(RV, Time)
		YS = np.array(RV)
		RK1 = Drv * DT
		RV = YS + RK1
		
		Drv = self.twobody_dynamic(RV, Time+DT)
		RK2 = Drv * DT
		RV = YS + RK2
		
		Drv = self.twobody_dynamic(RV, Time+DT)
		RK3 = Drv * H
		RV = YS + RK3
		
		Drv = self.twobody_dynamic(RV, Time+H)
		RV = YS + (2*(RK1+RK3) + 4*RK2 + Drv*H) / 6
		return np.array(RV)
		
	
	def twobody_dynamic(self, RV, Time, MIU=MIU_E):
		'''二体动力学方程，仅考虑中心引力u
		输入：	惯性系下位置速度	m, m/s, np.array
		输出：	返回d(RV)/dt，动力学增量, np.array'''
		R_Norm = sqrt(np.dot(RV[:3], RV[:3]))
		drdv = []
		drdv.extend(RV[3:])
		drdv.extend(-MIU*(RV[:3]) / R_Norm**3)
		return np.array(drdv)
		
		
	def generate_time(self, start_t="20171231", end_t="20180101"):
		'''产生datetime列表，用于计算对应的儒略日, multi-time
		输入：起始和终止时间		str类型
		输出：返回产生的utc时间列表	datetime构成的list'''
		start_t = datetime.strptime(start_t, "%Y%m%d")
		end_t = datetime.strptime(end_t, "%Y%m%d")
		time_list = pd.date_range(start_t, end_t, freq="120S")	#固定步长120s
		return time_list
		
		
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
		
		
	def moon2Earth(self, time_utc):
		'''计算地球相对于月球的位置矢量, single-time
		输入：UTC时间										datetime			
		输出：返回地球相对于月球的位置矢量(J2000下), m		np.array'''
		time_JD = self.utc2JD(time_utc)
		m2e_bc = -kernel[3, 301].compute(time_JD) * 1000
		return m2e_bc
		
		
	def moon2Sun(self, time_utc):		
		'''计算太阳相对于月球的位置矢量, single-time
		输入：由UTC时间														datetime
		输出：返回太阳相对于月球的位置矢量(J2000下), m,	[[x], [y], [z]]		np.array'''
		time_JD = self.utc2JD(time_utc)
		e_bc2m = kernel[3, 301].compute(time_JD) * 1000
		s_bc2e_bc = kernel[0, 3].compute(time_JD) * 1000
		return -(s_bc2e_bc + e_bc2m)
		
		
	def moon_Cbi(self, time_utc):
		'''返回月惯系到月固系的方向余弦矩阵, single-time，张巍-月球物理天平动对环月轨道(公式12)'''
		JD_day = self.utc2JD(time_utc)
		JD_T = (JD_day - 2451525.0) / 36525
		l_moon = ((134.963413889 + 13.06499315537*JD_day + 0.0089939*JD_day**2) * (pi/180)) % (2*pi)
		l_sun = ((357.52910 + 35999.05030*JD_T - 0.0001559*JD_T**2 - 0.0000048*JD_T**3) * (pi/180)) % (2*pi)
		w_moon = ((318.308686110 - 6003.1498961*JD_T + 0.0124003*JD_T**2) * (pi/180)) % (2*pi) #需要化为弧度
		omega_moon = ((125.044555556 - 1934.1361850*JD_T + 0.0020767*JD_T**2) * (pi/180)) % (2*pi) #需要化为弧度
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
		
		
	def thirdSun(self, time_utc, r_sat, miu=MIU_S):
		'''计算太阳对卫星产生的第三天体引力摄动, single-time
		输入：UTC时间time_utc，卫星相对中心天体的矢量，m, 均为np.array；
		输出：返回第三体太阳摄动加速度, m/s^2, np.array'''
		r_sun = self.moon2Sun(time_utc)
		norm_st = np.linalg.norm(r_sat-r_sun, 2)
		norm_et = np.linalg.norm(r_sun, 2)
		a_sun = -miu * ((r_sat-r_sun)/norm_st**3 + r_sun/norm_et**3)
		return a_sun
		
		
	def thirdEarth(self, time_utc, r_sat, miu=MIU_E):
		'''计算地球对卫星产生的第三天体引力摄动, single-time
		输入：UTC时间time_utc，卫星相对中心天体的矢量，m, 均为np.array;
		输出：返回第三体地球摄动加速度, m/s^2, np.array'''
		r_earth = self.moon2Earth(time_utc)
		norm_st = np.linalg.norm(r_sat-r_earth, 2)
		norm_et = np.linalg.norm(r_earth, 2)
		a_earth = -miu * ((r_sat-r_earth)/norm_st**3 + r_earth/norm_et**3)
		return a_earth
		
		
	def readCoffients(self, number=496, n=30):
		'''获取中心天体球谐引力系数，默认前30阶，包括0阶项, np.array'''
		global df
		f = df[:number]
		f.columns = ["l", "m", "Clm", "Slm"]
		f = f.set_index([f["l"], f["m"]]); del f['l'], f['m']
		Clm, Slm = f["Clm"], f["Slm"]
		Clm = [list(Clm.loc[i]) for i in range(0, n+1)]
		Slm = [list(Slm.loc[i]) for i in range(0, n+1)]
		return [np.array(Clm), np.array(Slm)]	
		
		
	def legendre_spher_col(self, phi, l=30, m=30):
		'''计算完全规格化缔合勒让德函数，球坐标形式，张飞-利用函数计算重力场元(公式2.14)，标准向前列递推算法
		输入：地心纬度phi, rad;		阶次l和m
		输出：可用于向量直接计算的勒让德函数，包含P_00项	list'''
		P = [ np.array([1, 0]), np.array([sqrt(3)*sin(phi), sqrt(3)*cos(phi), 0]) ]
		for i in range(2, l+1):
			p_ij = [ sqrt( (4*i**2-1) / (i**2-j**2) ) * sin(phi) * P[i-1][j] - \
					sqrt( (2*i+1)*(i+j-1)*(i-j-1) / ((i**2-j**2)*(2*i-3)) ) * P[i-2][j] for j in range(i) ]
			p_ii = sqrt( (2*i+1)/(2*i) ) * cos(phi) * P[i-1][i-1]
			p_ij.extend([p_ii, 0])
			P.append(np.array(p_ij))
		return P
		
		
	def legendre_cart(self, r_fixed, Re=RM, l=30, m=30):
		'''计算缔合勒让德函数，直角坐标形式，钟波-基于GOCE卫星(公式2.2.12)
		输入：月固系下的卫星位置矢量, r_fixed, 		np.array
		输出：直角坐标下的勒让德函数，包含0阶项, 	list'''
		X, Y, Z = r_fixed[0], r_fixed[1], r_fixed[2]
		r = np.linalg.norm(r_fixed, 2)
		V = [ np.array([Re/r, 0]), np.array([ sqrt(3)*Z*Re**2/r**3, sqrt(3)*X*Re**2/r**3, 0]) ]
		W = [ np.array([0, 0]), np.array([ 0, sqrt(3)*Y*Re**2/r**3, 0]) ]
		const = Re/r**2
		for i in range(2, l+2):
			Vij = [ sqrt( (4*i**2-1) / (i**2-j**2) ) * Z*const * V[i-1][j] - \
					sqrt( (2*i+1)*(i-j-1)*(i+j-1) / ((i**2-j**2)*(2*i-3)) ) * (Re/r)**2 * V[i-2][j] for j in range(i) ]
			Vii = sqrt( (2*i+1) / (2*i) ) * const * ( X * V[i-1][i-1] - Y * W[i-1][i-1] )
			Vij.extend([Vii, 0]); V.append(np.array(Vij))
			
			Wij = [ sqrt( (4*i**2-1) / (i**2-j**2) ) * Z*const * V[i-1][j] - \
					sqrt( (2*i+1)*(i-j-1)*(i+j-1) / ((i**2-j**2)*(2*i-3)) ) * (Re/r)**2 * W[i-2][j] for j in range(i) ]
			Wii = sqrt((2*i+1) / (2*i)) * const * ( X * W[i-1][i-1] + Y * V[i-1][i-1] )	#王正涛此处为减号
			Wij.extend([Wii, 0]); W.append(np.array(Wij))
		return (V, W)
		
		
	def legendre_cart_1(self, r_fixed, Re=RM, l=30, m=30):
		'''计算缔合勒让德函数，直角坐标形式，王正涛-卫星跟踪卫星测量确定地球重力场(公式4-2-5)
		输入：月固系下的卫星位置矢量, r_fixed, 		np.array
		输出：直角坐标下的勒让德函数，包含0阶项, 	list'''
		X, Y, Z = r_fixed[0], r_fixed[1], r_fixed[2]
		r = np.linalg.norm(r_fixed, 2)
		E = [ np.array([Re/r, 0]), np.array([ sqrt(3)*Z*Re**2/r**3, sqrt(3)*X*Re**2/r**3, 0]) ]
		F = [ np.array([0, 0]), np.array([ 0, sqrt(3)*Y*Re**2/r**3, 0]) ]
		const = Re/r**2
		for i in range(2, l+2):
			Eij = [ sqrt( (4*i**2-1) / (i**2-j**2) ) * Z*const * E[i-1][j] - \
					sqrt( (2*i+1)*(i-j-1)*(i+j-1) / ((i**2-j**2)*(2*i-3)) ) * (Re/r)**2 * E[i-2][j] for j in range(i) ]
			Eii = sqrt((2*i+1) / (2*i)) * const * ( X* E[i-1][i-1] - Y * F[i-1][i-1] )
			Eij.extend([Eii, 0]); E.append(np.array(Eij))
			
			Fij = [ sqrt( (4*i**2-1) / (i**2-j**2) ) * Z*const * F[i-1][j] - \
					sqrt( (2*i+1)*(i-j-1)*(i+j-1) / ((i**2-j**2)*(2*i-3)) ) * (Re/r)**2 * F[i-2][j] for j in range(i) ]
			Fii = sqrt((2*i+1) / (2*i)) * const * ( X * F[i-1][i-1] + Y * E[i-1][i-1] )	# 钟波此处为加号
			Fij.extend([Fii, 0]); F.append(np.array(Fij))
		return ( E, F )
		
		
	def diff_legendre_spher(self, phi, P, l=30, m=30):
		'''计算完全规格化缔合勒让德函数的一阶导数，球坐标形式，王正涛-卫星跟踪卫星测量(公式2-4-6)
		输入：地心纬度phi， 勒让德函数P
		输出：一阶导数，包含dP_00项		list'''
		deri_P, tan_phi = [], tan(phi)
		for i in range(0, l+1):
			dp_i0 = [ sqrt(i*(i+1) / 2) * P[i][1] ]  # j=0
			dp_ij = [ sqrt((i-j)*(i+j+1)) * P[i][j+1] - j*tan_phi * P[i][j] for j in range(1, i+1) ]
			dp_i0.extend(dp_ij); deri_P.append(np.array(dp_i0))
		return deri_P
		
		
	def centreGravity(self, r_sat, miu=MIU_M):
		'''计算中心天体的引力加速度，single-time, np.array'''
		r_norm = np.linalg.norm(r_sat, 2)
		g0 = -miu*r_sat / r_norm**3		# 中心引力 3*1
		return g0

		
	def nonspherGravity(self, r_sat, time_utc, miu=MIU_M, Re=RM, l=30, m=30):
		'''计算中心天体的非球形引力加速度，single-time, 王正涛-卫星跟踪卫星测量(公式2-4-7)
		输入：惯性系下卫星位置矢量r_sat，均为np.array;	utc时间(datetime);	miu默认为月球;
		输出：返回中心天体的引力加速度, np.array'''
		HL = self.moon_Cbi(time_utc)	# 月惯系到月固系的方向余弦矩阵 3*3
		r_fixed = np.dot(HL, r_sat)		# 应该在固连系下建立，王正涛
		r_norm = np.linalg.norm(r_fixed, 2)
		phi, lamda = atan(r_fixed[2] / sqrt(r_fixed[0]**2+r_fixed[1]**2)), atan(r_fixed[1]/r_fixed[0])
		spher2rect = np.array([ [cos(phi)*cos(lamda), cos(phi)*sin(lamda), sin(phi)], \
					[(-1/r_norm)*sin(phi)*cos(lamda), (-1/r_norm)*sin(phi)*sin(lamda),  (-1/r_norm)*cos(phi)], \
					[(-1/r_norm)*sin(lamda)/cos(phi), (1/r_norm)*cos(lamda)/cos(phi), 0] ])	#球坐标到直角坐标 3*3
		P = self.legendre_spher_col(phi, l, m)	# 勒让德函数
		tan_phi = tan(phi)
		Clm, Slm = self.readCoffients(number=495, n=l)	# 包括0阶项
		Vr, Vphi, Vlamda, const = 0, 0, 0, Re/r_norm
		for i in range(2, l):	# 王正涛i从2开始
			temp_r, temp = (i+1)*const**i, const**i
			Vr +=  Clm[i][0] * P[i][0] * temp_r		# j=0时dP不同，需要单独计算
			Vphi += Clm[i][0] * ( sqrt(i*(i+1)/2) * P[i][1] ) * temp
			for j in range(1, i+1):
				Vr += ( Clm[i][j]*cos(j*lamda) + Slm[i][j]*sin(j*lamda) ) * P[i][j] * temp_r
				Vphi += ( Clm[i][j]*cos(j*lamda) + Slm[i][j]*sin(j*lamda) ) * \
						( sqrt((i-j)*(i+j+1)) * P[i][j+1] - j * tan_phi * P[i][j] ) * temp 	# check dP_lm!
				Vlamda += ( -Clm[i][j]*sin(j*lamda) + Slm[i][j]*cos(j*lamda) ) * P[i][j] * temp * j
		g1 = np.array([-miu/r_norm**2*Vr, miu/r_norm*Vphi, miu/r_norm*Vlamda])	# 非球形引力 3*1
		g1 = np.dot(g1, spher2rect)	 # 球坐标到直角坐标，乘积顺序不要反了
		g1 = np.dot(HL.T, g1)	# 将月固系下加速度转换到月惯系下
		return g1
		
		
	def nonspherG_cart(self, r_sat, time_utc, miu=MIU_M, Re=RM, l=30, m=30):
		'''计算中心天体的非球形引力加速度，使用直角坐标形式，single-time，钟波-基于GOCE卫星(公式2.2.14)'''
		HL = self.moon_Cbi(time_utc)	# 月惯系到月固系的方向余弦矩阵 3*3
		r_fixed = np.dot(HL, r_sat)		# 转换为月固系下的位置矢量
		V, W = self.legendre_cart(r_fixed, Re, l, m)	
		C, S = self.readCoffients(number=495, n=l)
		ax, ay, az = 0, 0, 0
		const = miu/Re**2
		for i in range(0, l):	# V,W均存在0阶项，直接从1阶项开始
			temp = (2*i+1) / (2*i+3)
			a1 = sqrt((i+1)*(i+2) * temp/2)
			ax += const * (-a1*V[i+1][1] * C[i][0])		# j=0时公式不同，单独计算
			ay += const * (-a1*W[i+1][1] * C[i][0])
			az += const * ( -(i+1)*sqrt(temp) * (V[i+1][0]*C[i][0] + W[i+1][0]*S[i][0]) )	 # az需要j从0开始
			for j in range(1, i):
				a2 = sqrt( (i+j+1)*(i+j+2) * temp )
				b1_b2 = (i-j+1) * (i-j+2)
				a3 = sqrt( temp/b1_b2 ) if j != 1 else sqrt( 2*temp / b1_b2)
				b1_b2_a3 = (i-j+1)*(i-j+2) * a3 
				a4 = sqrt( (i+j+1)/(i-j+1) * temp )
				ax += const/2 * ( -a2 * (V[i+1][j+1]*C[i][j] + W[i+1][j+1]*S[i][j]) + \
								  b1_b2_a3 * ( V[i+1][j-1]*C[i][j] + W[i+1][j-1]*S[i][j]) )
				ay += const/2 * ( -a2 * (W[i+1][j+1]*C[i][j] - V[i+1][j+1]*S[i][j] ) - \
								  b1_b2_a3 * (W[i+1][j-1]*C[i][j] - V[i+1][j-1]*S[i][j] ) )
				az += const * ( -(i-j+1)*a4 * (V[i+1][j]*C[i][j] + W[i+1][j]*S[i][j]) )
		g1 = np.array([ax, ay, az])
		g1 = np.dot(HL.T, g1)	# 将月固系下加速度转换到月惯系下
		return g1
		
		
	def nonspherG_cart_1(self, r_sat, time_utc, miu=MIU_M, Re=RM, l=30, m=30):
		'''计算中心天体的非球形引力加速度，使用直角坐标形式，single-time，王正涛(公式4.3.9)'''
		HL = self.moon_Cbi(time_utc)	# 月惯系到月固系的方向余弦矩阵 3*3
		r_fixed = np.dot(HL, r_sat)		# 转换为月固系下的位置矢量
		E, F = self.legendre_cart_1(r_fixed, Re, l, m)	
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


	def solarPress(self, beta_last, time_utc, r_sat, step=120):
		'''计算太阳光压摄动, single-time
		输入：beta = (1+ita)*S(t);  utc时间(datetime);	卫星位置矢量，m, np.array
		输出：返回太阳光压在r_sat处造成的摄动, np.array'''
		r_sun = self.moon2Sun(time_utc)		#月球->太阳矢量
		ro = 4.5605e-6	#太阳常数
		delta_0 = 1.495978707e+11	#日地距离
		delta = r_sat - r_sun
		norm_delta = np.linalg.norm(delta, 2)
		# 计算太阳光压摄动中的系数，使用随机漫步过程模拟
		v = np.random.normal(0, 1)	# 0均值，协方差为1的高斯白噪声序列
		sigma = np.random.normal(0, 1)	#不知道是个什么东西，暂时作为噪声处理
		beta = beta_last + sigma*sqrt(step)*v
		F = beta * ro*(delta_0**2/norm_delta**2) * delta/norm_delta
		return F
		
		
	def jacobian(self, r_sat, time_utc, miu=MIU_M, Re=RM, l=30, m=30):
		'''计算系统动力学的Jacobian矩阵'''
		
		
		

		
	
		
		
		
if __name__ == "__main__":
	initial_rv = [6832.842724e3, 801.612273e3, 435.239952e3, -1.003679e3, 6.641038e3, 3.605789e3]
	initial_six = [6928.14e3, 0.00505, 28.5, 0, 0, 7.527*pi/180]
	ob = Orbit(rv = initial_rv)
	sixGeng = ob.rv2sixEle_Geng(initial_rv)
	sixZhang = ob.rv2sixEle_Zhang(initial_rv)
	time_list = ob.generate_time(start_t="20171231", end_t="20180131")
	C, S = ob.readCoffients()
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
	(V, W) = ob.legendre_cart(r_fixed, Re=RM, l=30, m=30)
	(E, F) = ob.legendre_cart_1(r_fixed, Re=RM, l=30, m=30)
	print((np.array(V) - np.array(E))[:10])
	print((np.array(W) - np.array(F))[:10])
	a1 = [ np.linalg.norm(ob.nonspherGravity(r_sat, time_utc), 2) for (r_sat, time_utc) in zip(r_array, utc_list) ]
	plt.plot(a1, label="a1 spher")
	a3 = [ np.linalg.norm(ob.nonspherG_cart(r_sat, time_utc)) for (r_sat, time_utc) in zip(r_array, utc_list) ]
	plt.plot(a3, "--", label="a3 cart")
	a4 = [ np.linalg.norm(ob.nonspherG_cart_1(r_sat, time_utc)) for (r_sat, time_utc) in zip(r_array, utc_list) ]
	plt.plot(a4, "--", label="a4 cart")
	plt.legend(); plt.show()
	