# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp, odeint, ode
from matplotlib import pyplot as plt
import re
from math import *
from datetime import datetime
from jplephem.spk import SPK
from orbit import *
from pprint import pprint
import matlab.engine



class Test_Orbit(Orbit):

	def __init__(self):
		return

	def singleDay_error(self):
		number, t0 = 720, 0
		rv_0 = [1837553.088459, -100877.893987, -0.369920, 59.176544, 1077.932044, 1425.301239]
		HPOP = pd.read_csv("STK/Moon_HPOP.csv")[:number]	# 取前number个点进行试算
		TwoBody = pd.read_csv("STK/Moon_TwoBody.csv")[:number]	# 取前number个点进行试算
		del HPOP["Time (UTCG)"]
		del TwoBody["Time (UTCG)"]
		HPOP = np.array(HPOP).T
		TwoBody = np.array(TwoBody).T
		orbit = self.integrate_orbit(rv_0, number)
		delta_1 = HPOP - TwoBody
		delta_2 = orbit - TwoBody
		delta_3 = HPOP - orbit
		
		plt.figure(1)
		plt.plot(delta_1[0], label="(HPOP - TwoBody)_x")
		plt.plot(delta_1[1], label="(HPOP - TwoBody)_y")
		plt.plot(delta_1[2], label="(HPOP - TwoBody)_z")
		plt.legend()
		
		plt.figure(2)
		plt.plot(delta_2[0], label="(orbit - TwoBody)_x")
		plt.plot(delta_2[1], label="(orbit - TwoBody)_y")
		plt.plot(delta_2[2], label="(orbit - TwoBody)_z")
		plt.legend()
		
		plt.figure(3)
		plt.plot(delta_3[0], label="(HPOP - orbit)_x")
		plt.plot(delta_3[1], label="(HPOP - orbit)_y")
		plt.plot(delta_3[2], label="(HPOP - orbit)_z")
		plt.legend()
		
		plt.show()
		
		
	def inertial2fixed(self, r_array, utc_array):
		HL_array = np.array([ self.moon_Cbi(time_utc) for time_utc in utc_array ])
		r_fixed = np.array([ np.dot(HL, r_sat) for HL, r_sat in zip(HL_array, r_array) ])
		return (r_fixed, HL_array)


if __name__ == "__main__":

	test = Test_Orbit()
	test.singleDay_error()
	# eng = matlab.engine.start_matlab()
	
	
	# number = 360
	# data = pd.read_csv("STK/Moon_HPOP.csv")[:number]	# 取前number个点进行试算
	# del data["Time (UTCG)"]
	# r_array = data[['x (m)', 'y (m)', 'z (m)']].values
	# utc_array = (test.generate_time(start_t="20180101", end_t="20180331"))[:number]
	# r_fixed, HL_array = test.inertial2fixed(r_array, utc_array)
	# mat_g = np.array([ np.dot( HL.T, np.array( eng.gravitysphericalharmonic( matlab.double(rf.tolist()), 'LP165P', 30.0, nargout=3 ) ) ) \
					# for HL, rf in zip(HL_array, r_fixed) ])
	# py_g = np.array([ test.centreGravity(r_sat)+test.nonspherGravity(r_sat, time_utc) \
					# for r_sat, time_utc in zip(r_array, utc_array) ])
	# py_gvec = np.array([ test.centreGravity(r_sat)+test.nonspher_Gvec(r_sat, time_utc) \
					# for r_sat, time_utc in zip(r_array, utc_array) ])
	# print(mat_g[:5])
	# pyg_matg = py_g - mat_g
	# pyg_vec = py_g - py_gvec
	# vec_matg = py_gvec - mat_g
	
	# plt.figure(1)
	# plt.plot(pyg_matg)
	
	# plt.figure(2)
	# plt.plot(pyg_vec)
	
	# plt.figure(3)
	# plt.plot(vec_matg)
	
	# plt.show()
	eng.quit()