# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp, odeint, ode
from matplotlib import pyplot as plt
from math import *
from orbit_km import *



class Test_Orbit(Orbit):

	def __init__(self):
		return

	def singleDay_error(self):
		number, t0 = 720, 0
		# HPOP = pd.read_csv("STK/Inertial_HPOP_30d.csv", nrows=number, usecols=range(1,7))	# 取前number个点进行试算
		HPOP = np.load("STK/HPOP_1.npy")[:number]
		TwoBody = pd.read_csv("STK/Part_2/Inertial_TwoBody_30d.csv", nrows=number, usecols=range(1,7))		# 取前number个点进行试算
		HPOP = np.array(HPOP).T
		TwoBody = np.array(TwoBody).T
		rv_0 = HPOP[:, 0]
		print(rv_0)
		orbit = self.integrate_orbit(rv_0, number)
		delta_1 = HPOP - TwoBody
		delta_2 = (orbit - TwoBody)
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
		
		
	def AccMatlab_error(self):
		ob = Orbit()
		number = 20
		data = pd.read_csv("STK/Inertial HPOP_30d.csv", nrows=number)	# 取前number个点进行试算
		del data["Time (UTCG)"]
		RV_array = data.values
		r_array = RV_array[:, :3]
		utc_array = (ob.generate_time(start_t="20180101", end_t="20180331"))[:number]
		utcJD_list = [ time_utc.to_julian_date() for time_utc in utc_array ]
		tdbJD_list = [ time_utc.to_julian_date() + 69.184/86400 for time_utc in utc_array ]
		I2F_list = [ ob.moon_Cbi(tdb_jd) for tdb_jd in tdbJD_list ]
		rFixed_list = [ np.dot(I2F, r_sat) for (I2F, r_sat) in zip(I2F_list, r_array) ]
		r_sat, RV, time_utc = r_array[0], RV_array[0], utc_array[0]
		tdb_jd = time_utc.to_julian_date() + 69.184/86400
		
		lg = ob.legendre_spher_alfs(1, lm=6)
		a1 = np.array([ ob.nonspherGravity(r_sat, tdb_jd) for (r_sat, tdb_jd) in zip(r_array, tdbJD_list) ])
		a2 = np.array([ ob.nonspher_moon(r_sat, tdb_jd) for (r_sat, tdb_jd) in zip(r_array, tdbJD_list) ])
		a3 = np.array([ ob.nonspher_moongravity(r_sat, tdb_jd) for (r_sat, tdb_jd) in zip(r_array, tdbJD_list) ])
		# a_matlab = np.load("a_matlab.npy")
		a_matlab = np.array([ ob.nonspher_matlab(r_sat, tdb_jd) for (r_sat, tdb_jd) in zip(r_array, tdbJD_list) ])
		np.save("a_matlab.npy", a_matlab)



if __name__ == "__main__":

	test = Test_Orbit()
	test.singleDay_error()
