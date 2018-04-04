# -*- coding: utf-8 -*-

C = 1

class Orbit:
	
	def __init__(self):
		return
		
	def func(self, x):
		global C
		b = C*x
		return b
		
		
if __name__ == "__main__":
	
	ob = Orbit()
	print(ob.func(3))