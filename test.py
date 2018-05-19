# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
from math import *
from orbit_km import *



class Test_Orbit(Orbit):

	def __init__(self):
		return

	def singleDay_error(self):
		number, t0 = 720, 0
		HPOP = np.array( pd.read_csv("STK/Part_2/1_Inertial_HPOP_660.csv", nrows=number, usecols=range(1,7)) ).T	# 取前number个点进行试算
		TwoBody = np.array( pd.read_csv("STK/Part_2/Inertial_TwoBody_30d.csv", nrows=number, usecols=range(1,7)) ).T
		rv_0 = TwoBody[:, 0]
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
		
		
	def draw_accelerate(self, number=200, step=120):
		data = pd.read_csv("STK/Moon.csv")[:number]
		del data["Time (UTCG)"]
		data *= 1000
		data.columns = ['x (m)', 'y (m)', 'z (m)', 'vx (m/sec)', 'vy (m/sec)', 'vz (m/sec)']
		r_array = data[['x (m)', 'y (m)', 'z (m)']].values
		orb = Orbit()
		a0 = np.array([ np.linalg.norm(orb.centreGravity(r_sat=r_sat, miu=orbit.MIU_M, Re=1.738e+06), 2) for r_sat in r_array ])
		utc_list = orb.generate_time(start_t="20171231", end_t="20180101")
		a1 = np.array([ np.linalg.norm(orb.nonspherGravity(r_sat, time_utc)) for (r_sat, time_utc) in zip(r_array, utc_list) ])
		a_solar = np.array([ np.linalg.norm(orb.solarPress(beta_last=0.75, time_utc=time_utc, r_sat=r_sat)) \
				for (r_sat, time_utc) in zip(r_array, utc_list) ])
		a_sun = np.array([ np.linalg.norm(orb.thirdSun(time_utc=time_utc, r_sat=r_sat, miu=orbit.MIU_S)) \
				for (r_sat, time_utc) in zip(r_array, utc_list) ])
		a_earth = np.array([ np.linalg.norm(orb.thirdEarth(time_utc=time_utc, r_sat=r_sat, miu=orbit.MIU_E)) \
				for (r_sat, time_utc) in zip(r_array, utc_list) ])
		time_range = range(0, number*step, step)
		plt.figure(1)
		plt.xlabel("Time / s", fontsize=18); plt.ylabel(r"$a / (m/s^2)$", fontsize=18)
		plt.plot(time_range, a0, "r-", label="centre gravity")
		a = a0+a1+a_solar+a_sun+a_earth
		plt.plot(time_range, a, "g-", label="Combined acceleration")
		plt.legend(fontsize=18); 
		
		plt.figure(2)
		plt.xlabel("Time / s", fontsize=18); plt.ylabel(r"$a / (m/s^2)$", fontsize=18)
		plt.plot(time_range, a1, "g--", label="nonspher gravity")
		plt.plot(time_range, a_sun, "r-", label="third-sun gravity")
		plt.plot(time_range, a_earth, "y-", label="third-earth gravity")
		plt.plot(time_range, a_solar, "b-", label="solar pressure gravity")
		plt.legend(fontsize=18); plt.show()



if __name__ == "__main__":

	test = Test_Orbit()
	test.singleDay_error()
