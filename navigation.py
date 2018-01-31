# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import re
from math import *
from datetime import datetime
from jplephem.spk import SPK
from orbit import Orbit



class Navigation:
	
	def __init__(self):
		self.orbit = Orbit()
		
	def __del__(self):
		return
		
		
	def jacobian(self, r_sat, time_utc):
		'''计算系统的Jacobian矩阵'''
		A1 = self.orbit.jacobian_nonspher(r_sat, time_utc)
		A2 = self.orbit.jacobian_third(r_sat, time_utc)
		return A1 + A2