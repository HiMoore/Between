# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import re
from math import *
from datetime import datetime
from jplephem.spk import SPK
from orbit import Orbit
from pprint import pprint



class Navigation(Orbit):
	
	def __init__(self):
		return
		
	def __del__(self):
		return
		
		
	def jacobian_double(self, r1, r2, time_utc):
		'''计算双星系统的Jacobian矩阵'''
		A1 = self.jacobian_single(r1, time_utc)	# 6*6
		A2 = self.jacobian_single(r2, time_utc)	# 6*6
		zeros = np.zeros((6, 6))
		A1 = np.hstack((A1, zeros)); A2 = np.hstack((zeros, A2))	# 6*12, 6*12
		J = np.vstack((A1, A2))	# 12*12
		return J
		
		
	def jacobian_measure(self, r1, r2):
		'''计算双星测量模型的Jacobian矩阵'''
		r1_r2, zeros = r1-r2, np.zeros((3,3))
		norm = np.linalg.norm(r1_r2, 2)
		h1_r1, h1_r2 = r1_r2/norm, -r1_r2/norm	# 1*3, 1*3
		h1_r1 = np.hstack((h1_r1, zeros[0]));  h1_r2 = np.hstack((h1_r2, zeros[0]))	# 1*6, 1*6
		up = np.hstack((h1_r1, h1_r2))	# 1*12
		h2_r1 = np.identity(3)/norm - np.outer(r1_r2, r1_r2)/norm**3	# 3*3
		h2_r2 = -np.identity(3) + np.outer(r1_r2, r1_r2)/norm**3	#3*3
		h2_r1 = np.hstack((h2_r1, zeros));  h2_r2 = np.hstack((h2_r2, zeros))	# 3*6, 3*6
		low = np.hstack((h2_r1, h2_r2))	# 3*12
		H = np.vstack((up, low))	# 4*12
		return H
		
	
	def srif_ekf(self, X1_k, X2_k):
		'''SRIF形式的EKF'''
		PH_0, Rvv_0 = np.identity(12), np.diag([1.5, 1/360, 1/360, 1/360])	# 距离测量精度1.5m, 角度10角秒
		H0 = self.jacobian_measure(X1_k[:3], X2_k[:3])
		Rxx_0 = np.vstack( (np.zeros((8, 12)), np.dot(Rvv_0, H0)) )
		mean, cov = np.zeros(12), np.identity(12)
		for k in range(1, MAX):
			Rxx_k = -np.random.multivariate_normal(mean, cov)
			time_utc = utc_array[k]
			X1_kp1, X2_kp1 = self.orbit_rk4(X1_k, time_utc), self.orbit_rk4(X2_k, time_utc)
			
			
		
		
		
if __name__ == "__main__":
	
	ob = Orbit()
	nav = Navigation()
	
	number = 40
	data = pd.read_csv("STK/Moon.csv")[:number]	# 取前number个点进行试算
	del data["Time (UTCG)"]
	data *= 1000
	data.columns = ['x (m)', 'y (m)', 'z (m)', 'vx (m/sec)', 'vy (m/sec)', 'vz (m/sec)']
	r_array = data[['x (m)', 'y (m)', 'z (m)']].values
	utc_list = (ob.generate_time(start_t="20171231", end_t="20180101"))[:number]
	r1, r2, utc = r_array[0], r_array[1], utc_list[0]
	PH = nav.jacobian_double(r1, r2, utc)
	Hk = nav.jacobian_measure(r1, r2, utc)