# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import re
from math import sin, cos, tan, asin, acos, pi, sqrt, atan, atan2


MIU_E, CSD_MAX, CSD_EPS = 3.986004360e14, 1e+30, 1e-10


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
		'''
		功能：变量限幅
		输入：变量, 变量上界, 变量下界
		输出：返回限幅后的变量
		'''
		if x > x_up:
			return x_up
		elif x < x_low:
			return x_low
		else:
			return x
		
	
	def sixEle2rv(self, MIU=MIU_E):
		'''
		功能：轨道六根数到惯性位置速度转换
		输入：六根数 sixEle[6]
		输出：位置向量 R[3]		m
			  速度向量 V[3]		m/s
		'''
		a, e, i, omiga, w, M = self.sixEle
		a = self.LimitUpLow(a, CSD_MAX, 1.0);
		e = self.LimitUpLow(e, 1-CSD_EPS, 0.0);
		f = self.M2f(M, e)
		u = w + f
		r = self.LimitUpLow(a*(1.0-e*e)/(1.0+e*cos(f)), CSD_MAX, 1.0)
		v = self.LimitUpLow(sqrt(MIU*abs(2/r - 1/a)), 3e8, 1.0)
		P = np.array([cos(w)*cos(omiga) - sin(w)*sin(omiga)*cos(i), cos(w)*sin(omiga) + sin(w)*cos(omiga)*cos(i), sin(w)*sin(i)])
		Q = np.array([-sin(w)*cos(omiga) - cos(w)*sin(omiga)*cos(i), -sin(w)*sin(omiga) + cos(w)*cos(omiga)*cos(i), cos(w)*sin(i)])
		self.r = list(r*cos(f)*P + r*sin(f)*Q)
		self.v = list(sqrt(MIU/a/(1-e**2)) * (-sin(f)*P + (e+cos(f))*Q))
		
		
	def M2f(self, M, e):
		'''
		功能：平近点角到真近点角转换	See page77 of book<航天器轨道确定>
		输入：平近点角 M,  偏心率 e
		输出：返回真近点角 f
		'''
		e = self.LimitUpLow(e,0.999,0.0)
		N = int(5000 // (1+(1000-1000*e)))
		E = M
		for i in range(1, N):
			E = E + (M-E+e*sin(E)) / (1-e*cos(E))
		f = 2*atan(sqrt((1+e)/(1-e) * tan(E/2)))
		return f
		
	
	def rv2sixEle(self, MIU=MIU_E):
		'''
		功能：位置速度到轨道六根数惯性转换
		输入：R[3]	位置向量，m
			  V[3]	速度向量，m/s
		输出：SixE[6]：六根数
		'''
		R = self.LimitUpLow(sqrt(np.dot(self.r, self.r)), CSD_MAX, 1.0)
		V = self.LimitUpLow(sqrt(np.dot(self.v, self.v)), CSD_MAX, 1.0)
		h = np.cross(self.r, self.v)
		H = sqrt(np.dot(h, h))
		Energy =  np.dot(self.v, self.v)/2 - MIU/R
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
		self.sixEle = [a, e, i, omiga, w, M]
		
		
	def divided(self, y, x):
		'''功能：除法
		输入：x：分母		y：分子
		输出：y/x
		'''
		if x<CSD_EPS and x>=0:
			return y/CSD_EPS
		elif x>-CSD_EPS and x<0:
			return -y/CSD_EPS
		else:
			return y/x
			
			
	def Atan4(self, y, x):
		'''功能：反正切
		输入：x：x坐标		y：y坐标
		输出：反正切
		'''
		return 0 if (abs(y)<CSD_EPS and abs(x)<CSD_EPS) \
				else atan2(y, x)
			
		
	def orbit_rk4(self, H=60, Time=0, MIU=MIU_E):
		'''四阶龙格库塔法递推位置和速度											单位
		输入：	地心J2000下相对地球位置速度 或 月惯系下相对月球位置速度			m	m/s
				积分步长														s
				当前时刻																	
		输出：	下一时刻月惯系下卫星位置、速度									m	m/s
		'''
		DT = 0.5 * H
		RV = np.array(self.r + self.v)
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
		self.r = list(RV[:3]); self.v = list(RV[3:])
		
	
	def twobody_dynamic(self, RV, Time, MIU=MIU_E):
		'''二体动力学方程，仅考虑中心引力u
		输入：	惯性系下位置速度	m, m/s, np.array
		输出：	返回d(RV)/dt，动力学增量, np.array
		'''
		R_Norm = sqrt(np.dot(RV[:3], RV[:3]))
		drdv = []
		drdv.extend(RV[3:])
		drdv.extend(-MIU*(RV[:3]) / R_Norm**3)
		return np.array(drdv)
		
		
		
class Auto_Navigation:
		
		def __init__(self, X):
			self.X = X
			
			
		def observability_matrix(self, r, v):
			
		
		
		
		
		
		
if __name__ == "__main__":
	initial_rv = [6893.149908e3, 0.000000, 0.000000, -0.000000, 6.699654e3, 3.637615e3]
	initial_six = [6928.14e3, 0.00505, 28.5, 0, 0, 0]
	ob = Orbit(rv = initial_rv)
	#ob.sixEle2rv()
	for i in range(3):
		ob.orbit_rk4(H=120, Time=ob.Time)
		print(ob.r, ob.v)
	