# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import re
from math import *
from datetime import datetime
from jplephem.spk import SPK
from pprint import pprint


MIU_E, MIU_M, MIU_S = 3.986004415e+14, 4.902801056e+12, 1.327122e+20
CSD_MAX, CSD_EPS, number = 1e+30, 1e-10, 495
STEP = 120
kernel = SPK.open(r"C:\Users\Mooreq\Notepad++\de421\de421.bsp")
df = pd.read_csv("STK/LP165P.grv", sep="\t", header=None)


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
		N = line_nodes / np.linalg.norm(line_nodes)
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
			
		
	def orbit_rk4(self, H=60, Time=0, MIU=MIU_E):
		'''四阶龙格库塔法递推位置和速度, single-time							单位
		输入：	地心J2000下相对地球位置速度 或 月惯系下相对月球位置速度			m	m/s
				积分步长	s;		当前时刻																	
		输出：	下一时刻月惯系下卫星位置、速度									m	m/s'''
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
		输出：	返回d(RV)/dt，动力学增量, np.array'''
		R_Norm = sqrt(np.dot(RV[:3], RV[:3]))
		drdv = []
		drdv.extend(RV[3:])
		drdv.extend(-MIU*(RV[:3]) / R_Norm**3)
		return np.array(drdv)
		
		
	def generate_time(self, start_t="20171231", end_t="20171231"):
		'''产生datetime列表，用于计算对应的儒略日, multi-time
		输入：起始和终止时间		str类型
		输出：返回产生的utc时间列表	datetime构成的list'''
		start_t = datetime.strptime(start_t, "%Y%m%d")
		end_t = datetime.strptime(end_t, "%Y%m%d")
		time_list = pd.date_range(start_t, end_t, freq="120S")	#固定步长120s
		return time_list
		
		
	def utc2JD(self, time_utc):
		'''由UTC时间计算对应的儒略日，用于计算天体位置, single-time
		输入： UTC时间								datetime					
		输出：返回UTC时间对应的儒略日时间			np.array'''
		if time_utc.month == 1 or time_utc.month == 2:
			time_utc = datetime(time_utc.year-1, time_utc.month+12, time_utc.day)
		J0 = floor(365.25 * time_utc.year) - floor(time_utc.year/100) + floor(time_utc.year/400) + 1721116.5
		S = floor(30.6001 * (time_utc.month+1)) + time_utc.day - 122
		return J0+S
		
		
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
		'''返回月惯系到月固系的方向余弦矩阵, single-time'''
		JD_day = self.utc2JD(time_utc)
		JD_T = (JD_day - 2451525.0) / 36525
		l_moon = ((134.963413889 + 13.06499315537*JD_day + 0.0089939*JD_day**2) * (pi/180)) % (2*pi)
		l_sun = ((357.52910 + 35999.05030*JD_T - 0.0001559*JD_T**2 - 0.0000048*JD_T**3) * (pi/180)) % (2*pi)
		w_moon = ((318.308686110 - 6003.1498961*JD_T + 0.0124003*JD_T**2) * (pi/180)) % (2*pi) #需要化为弧度
		omega_moon = ((125.044555556 - 1934.1361850*JD_T + 0.0020767*JD_T**2) * (pi/180)) % (2*pi) #需要化为弧度
		kesai = ((l_moon + w_moon + omega_moon)) % (2*pi)
		tao_1 , tao_2, tao_3 = 2.9e-4, -0.58e-4, -0.87e-4
		tao = tao_1*sin(l_sun) + tao_2*sin(l_moon) + tao_3*sin(2*w_moon)
		I = 0.026917;
		phi_1, phi_2, phi_3 = -5.18e-4, 1.8e-4, -0.53e-4
		phi = sin(phi_1*sin(l_moon) + phi_2*sin(l_moon+2*w_moon) + phi_3*sin(2*l_moon+2*w_moon)) 
		theta_1, theta_2, theta_3= phi_1 + (1/30)*(pi/180),  phi_2,  phi_3;
		theta = theta_1*cos(l_moon) + theta_2*cos(l_moon+2*w_moon) + theta_3*cos(2*l_moon+2*w_moon)
		Rx = lambda x: np.array([[1, 0, 0], [0, cos(x), sin(x)], [0, -sin(x), cos(x)]])
		Ry = lambda y: np.array([[cos(y), 0, -sin(y)], [0, 1, 0], [sin(y),0, cos(x)]])
		Rz = lambda z: np.array([[cos(z), sin(z), 0], [-sin(z), cos(z), 0], [0, 0, 1]])
		HL = np.dot( Rz(kesai+tao-phi).dot(Rx(-I-theta)).dot(Rz(phi)).dot(Rx(I)), Rz(omega_moon) )
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
		
		
	def readCoffients(self, number=495, n=30):
		'''获取中心天体球谐引力系数，默认前30阶'''
		global df
		f = df[:number]
		f.columns = ["l", "m", "Clm", "Slm"]
		f = f.set_index([f["l"], f["m"]]); del f['l'], f['m']
		Clm, Slm = f["Clm"], f["Slm"]
		C = [list(Clm.loc[i]) for i in range(1, n+1)]
		S = [list(Slm.loc[i]) for i in range(1, n+1)]
		return [np.array(C), np.array(S)]	
		
		
	def legendre(self, phi, l=30, m=30):
		'''计算缔合勒让德函数
		输入：地心纬度phi, rad;		阶次l和m
		输出：可用于向量直接计算的勒让德函数	np.array'''
		P = [[1], [sqrt(3)*sin(phi), sqrt(3)*cos(phi)]]
		for i in range(2, l+1):
			p_ij = [ sqrt((4*i**2-1)/(i**2-j**2)) * sin(phi) * P[i-1][j] - sqrt((2*i+1)/(2*i-3) - ((i-1)**2-j**2)/(i**2-j**2)) * P[i-2][j] for j in range(i-1)]
			p_im = sqrt(2*i+1) * sin(phi) * P[i-1][i-1]
			p_ii = sqrt((2*i+1)/(2*l)) * cos(phi) * P[i-1][i-1]
			p_ij.extend([p_im, p_ii])
			P.append(p_ij)
		P.pop(0)
		return np.array(P)
		
		
	def deri_legendre(self, phi, Plm, l=30, m=30):
		'''计算缔合勒让德函数的一阶导数'''
		deri_P = [[0], [sqrt(3)*cos(phi), -sqrt(3)*sin(phi)]]
		for i in range(2, l+1):
			DP_ij = [ sqrt((4*i**2-1)/(i**2-j**2)) * ( cos(phi)*Plm[i-1][j] + sin(phi)*deri_P[i-1][j] ) \
					- sqrt((2*i+1)/(2*i-3) * ((i-1)**2-j**2)/(i**2-j**2)) * deri_P[i-2][j] for j in range(i-1) ]
			DP_im = sqrt(2*i+1) * ( cos(phi)*Plm[i-1][i-1] + sin(phi)*deri_P[i-1][i-1] )
			DP_ii = sqrt((2*i+1)/(2*i)) * ( cos(phi)*deri_P[i-1][i-1] - sin(phi)*Plm[i-1][i-1] )
			DP_ij.extend([DP_im, DP_ii])
			deri_P.append(DP_ij)
		deri_P.pop(0)
		return np.array(deri_P)
		
		
	def centreGravity(self,  r_sat, time_utc, miu=MIU_M, Re=1.738e+06, l=30, m=30):
		'''计算中心天体的引力加速度，包含非球形引力, single-time, 
		输入：miu; 	卫星位置矢量r_sat，，均为np.array;	utc时间(datetime)
		输出：返回中心天体的引力加速度, np.array'''
		r_norm = np.linalg.norm(r_sat)
		phi, lamda = acos(r_sat[2] / sqrt(r_sat[0]**2+r_sat[1]**2)), atan(r_sat[1]/r_sat[0])
		spher2rect = np.array([ [cos(phi)*cos(lamda), cos(phi)*sin(lamda), sin(phi)], \
					[-1/r_norm*sin(phi)*cos(lamda), -1/r_norm*sin(phi)*sin(lamda),  \
					1/r_norm*cos(phi)], [-1/r_norm*sin(lamda)/cos(phi), 1/r_norm*cos(lamda)/cos(phi), 0] ])	#球坐标到直角坐标 3*3
		Plm = self.legendre(phi, l, m)
		deri_Plm = self.deri_legendre(phi, Plm, l, m)
		Clm, Slm = self.readCoffients()
		g0 = -miu*r_sat / r_norm**3		# 中心引力 3*1
		HL = self.moon_Cbi(time_utc)	# 月惯系到月固系的方向余弦矩阵 3*3
		Vr, Vphi, Vlamda = 0, 0, 0
		for i in range(1, l):
			temp_r, temp = (i+1)*(Re/r_norm)**i, (Re/r_norm)**i
			for j in range(i+1):
				Vr += ( Clm[i][j]*cos(j*lamda) + Slm[i][j]*sin(j*lamda) ) * Plm[i][j] * temp_r
				Vphi += ( Clm[i][j]*cos(j*lamda) + Slm[i][j]*sin(j*lamda) ) * deri_Plm[i][j] * temp
				Vlamda += ( -Clm[i][j]*sin(j*lamda) + Slm[i][j]*cos(j*lamda) ) * Plm[i][j] * temp
		g1 = [-miu/r_norm**2*Vr, miu/r_norm*Vphi, miu/r_norm*Vlamda]	# 非球形引力 3*1
		g1 = np.dot(spher2rect, g1)	
		g1 = np.dot(HL.T, g1)	#将月固系下加速度转换到月惯系下
		return g0+g1
		

	def solarPress(self, beta_last, time_utc, r_sat):
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
		F = beta * ro*(delta_0**2/norm_delta) * delta/norm_delta
		return F
		
		
	def acceler_disturb(self, time_utc, r_sat, da_last, step=120):
		'''对高阶非球形引力做保守计算，使用一阶Gauss/Markov过程模拟，返回产生的加速度
		输入：time_utc(datetime)；卫星位置矢量r_sat； 步长 step  s；
		输出：返回当前时刻的扰动加速度 da  m/s**2	np.array'''
		r_norm = np.linalg.norm(r_sat)
		tao, sigma = r_norm ** 1.5, r_norm ** (-3)	#不知道为什么缩放
		va = np.random.normal(0, 1, 3)# 0均值，协方差为1的高斯白噪声序列
		da = exp(-step/tao) * da_last + sigma*sqrt(1-exp(-2*step/tao)) * va
		return da
		
		
	def solar_beta(self, time_utc, beta_last, step=120):
		'''计算太阳光压摄动中的系数，使用随机漫步过程模拟
		输入：time_utc(datetime)；上一时刻产生的beta_last； 步长 120s
		输出：返回太阳光压摄动中的系数beta(也是待估参数)	标量'''
		v = np.random.normal(0, 1)	# 0均值，协方差为1的高斯白噪声序列
		sigma = np.random.normal(0, 1)	#不知道是个什么东西，暂时作为噪声处理
		beta = beta_last + sigma*sqrt(step)*v
		return beta
		
		
	def dynamic_model(self, RV, time_utc, da_last, beta_last):
		'''系统完整动力学模型，包含月球引力，太阳和地球引力，以及太阳光压摄动
		输入：位置和速度RV  m m/s； time_utc(datetime)
		输出：返回包括速度和加速度动力学模型'''
		DX = RV[:3]
		g = self.centreGravity(r_sat=RV[:3], time_utc, miu=MIU_M, Re=1.738e+06, l=30, m=30)
		dirta_a = self.acceler_disturb(time_utc, r_sat=RV, da_last, step=120)
		a_sun = self.thirdSun(time_utc, r_sat=RV, miu=MIU_S)
		a_earth = self.thirdEarth(time_utc, r_sat=RV, miu=MIU_E)
		a_solar = self.solarPress(beta, time_utc, r_sat)
		
		
		
	
		
		
		
if __name__ == "__main__":
	initial_rv = [6832.842724e3, 801.612273e3, 435.239952e3, -1.003679e3, 6.641038e3, 3.605789e3]
	initial_six = [6928.14e3, 0.00505, 28.5, 0, 0, 7.527*pi/180]
	ob = Orbit(rv = initial_rv)
	sixGeng = ob.rv2sixEle_Geng(initial_rv)
	sixZhang = ob.rv2sixEle_Zhang(initial_rv)
	print("Geng: ", sixGeng)
	print("Zhang: ", sixZhang)
	print("Geng: ", ob.sixEle2rv_Geng(sixGeng))
	print("Zhang: ", ob.sixEle2rv_Zhang(sixZhang))
	time_list = ob.generate_time(start_t="20171231", end_t="20180131")
	print(ob.moon_Cbi(time_list[0]))
	legendres = ob.legendre(phi=30)
	Clm, Slm = ob.readCoffients()
	print(ob.thirdEarth(time_utc=time_list[0], r_sat=np.array(initial_rv[:3]), miu=MIU_E))
	print(ob.thirdSun(time_utc=time_list[0], r_sat=np.array(initial_rv[:3]), miu=MIU_S))
	print(ob.centreGravity(r_sat=np.array(initial_rv[:3]), time_utc=time_list[0], miu=MIU_M, Re=1.738e+06, l=30, m=30))
	