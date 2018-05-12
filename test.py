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
		# orbit = np.array( pd.read_csv("STK/Part_2/LP165_Inertial_HPOP_30.csv", nrows=number, usecols=range(1,7)) ).T
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
		



if __name__ == "__main__":

	test = Test_Orbit()
	test.singleDay_error()
