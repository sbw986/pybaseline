""" pybaseline.py:  Returns baseline from noisy quantal signal """

import numpy as np
import cmath
import matplotlib.pyplot as plt
import time
from numba import jit

def pybaseline(data, Rsqd, sigma_xi2, current, multiplier = 1, updateR2 = False, Nch = 1):
	"""
	Parameters
	----------
	data : 1D numpy array
		Raw data signal
	Rsqd : Double
		Noise estimate
	sigma_xi2 : Double
	current : Double
		Signal amplitude estimate
	multiplier : Double
		Signal multiplier value
	updateR2 : Boolean
	Nch : Integer
		Number of levels

	Returns
	-------
	1D numpy array
		Extracted baseline
	"""
	logLold = 1.e50
	n = 0

	size = len(data)

	data = multiply_array(data, multiplier)

	# initial guess
	background = init_b(data, current)

	davg = np.mean(data)

	main_d, lower_d, upper_d = set_matrix(size, Rsqd)

	j = 0
	not_converged = True
	while not_converged:
		t0 = time.time()
		j += 1

		p, p2 = set_n2(data, background, current, Rsqd, size, Nch, sigma_xi2)

		rhs = set_rhs(p, data, current, Rsqd)

		if j % 250 == 0:
			print("Iteration = ", j)

		main_d, lower_d, upper_d = set_matrix(size, Rsqd)

		background, main_d, rhs = solveMatrix(size, lower_d, main_d, upper_d, rhs, background)

		main_d, lower_d, upper_d = set_matrix(size, Rsqd)
		tmp = A_minus_B(data, background)

		if j > 1:
			if np.sum(p2) > 0:
				current = AdotB(p, tmp) / np.sum(p2)

		D2 = AdotB(tmp, tmp) - 2 * current * AdotB(tmp, p) + current * current * np.sum(p2)

		GRADBSQRD = gradbsquared(background)

		BRsqd = (GRADBSQRD/ ((size - 1) * sigma_xi2)) ** 2

		

		if j > 1 and updateR2 == True:
			Rsqd = updatedRsqd(BRsqd)

		sigma_xi2 = (GRADBSQRD/Rsqd + D2)/(size - 1)
		logL = lL(background, data,  Rsqd,  sigma_xi2,  current, size, Nch)

		print('logL: ', logL)

		if np.abs(logL - logLold) > 0.0001:
			logLold = logL
		else:
			not_converged = False
	print("converged")

	print('Iteration Total: ', j)

	p, p2 = set_n2(data, background, current, Rsqd, size, Nch, sigma_xi2)
	
	tmp1 = set_reconstruction(p, background, current)
	tmp2 = set_res(p, data, background, current)

	return background

def set_res(p, d, b, current):
	res = d - b - current * p
	return res

def set_reconstruction(p, background, current):
	tmp = background + current * p
	return tmp

def multiply_array(array, b):	
	array_out = array * b
	return array_out

def init_b(data, current):
	#davg = np.mean(data)
	#b = np.empty(len(data))
	
	"""
	for j,_ in enumerate(data):
		
		###
		b[j] = .85 * data[j]

		b[j] = data[j] - 0.5 * current
		b[j] = 0.9 * davg + 0.1 * data[j]
		b[j] = data[j] - current
		b[j] = 12 * np.exp(-1 * j / 11000) + 26

		###
		b[j] = (davg + data[j])/2
		b[j] = 100 * np.exp(-1 * j / 100000)

		b[j] = 130 * np.exp(-1 * j / 11000) + 26
		b[j] = 0.95 * data[j]

		b[j] = data[j] - current / 2

		b[j] = davg

		b[j] = 0.95 * data[j]

		###
		b[j] = 18
		b[j] = 0.95 * data[j]
	"""
	b = 0.95 * data

	return b

def set_matrix(size, Rsqd):
	main = np.ones(size) * -1 * Rsqd - 2.0
	main[0] = -1 * Rsqd - 1
	main[-1] = -1 * Rsqd - 1 
	lower = np.ones(size)
	upper = np.ones(size)
	return main, lower, upper

@jit
def set_n2(d, background, current, Rsqd, T, Nch, sigma_xi2):
	
	"""
	eta_list = []
	for j in range(Nch+1):
		eta_tmp = (d - background - j * current) * (d - background - j * current) / (2 * sigma_xi2)
		eta_list.append(eta_tmp)

	sum_ = 0
	sum_2 = 0
	denom_array_tmp = np.zeros(len(d))
	for j in range(Nch+1):
		for k in range(Nch+1):
			denom_array_tmp += np.exp(eta_list[j] - eta_list[k])
		sum_ += j/denom_array_tmp
		sum_2  += j*j/denom_array_tmp

	n = sum_
	n2 = sum_2

	
	"""
	n = np.empty(T)
	n2 = np.empty(T)
	for t in range(T):
		eta = np.empty(Nch+1)
		for j in range(Nch+1):
			eta[j] = (d[t] - background[t] - j * current) * (d[t] - background[t] - j * current)/(2 * sigma_xi2)

		sum_ = 0
		sum_2 = 0

		for j in range(Nch+1):
			denom = 0
			for k in range(Nch+1):
				denom += np.exp(eta[j] - eta[k])
			sum_ += j/denom
			sum_2 += j*j/denom

		n[t] = sum_
		n2[t] = sum_2

	return n, n2

def set_rhs(p, d, current, Rsqd):
	rhs = -1 * Rsqd * (d - current * p)
	return rhs

@jit
def solveMatrix(n, a, b, c, v, x):
	for i in range(1,n):
		m = a[i] / b[i - 1]
		b[i] = b[i] - m * c[i - 1]
		v[i] = v[i] - m * v[i - 1]

	x[n - 1] = v[n - 1] / b[n - 1]

	for i in range(n-2, -1, -1):
		x[i] = (v[i] - c[i] * x[i+1])/b[i]

	return x, b, v

def A_minus_B(A, B):
	AmB = A - B
	return AmB

def AdotB(A, B):
	ret = np.sum(A * B)
	return ret

def gradbsquared(b):
	ret = np.sum((b[1::] - b[0:-1]) ** 2)
	return ret

def updatedRsqd(BRsqd):
	disc = -3 * (BRsqd - 108) * BRsqd * BRsqd
	
	F = 18 * BRsqd + cmath.sqrt(disc)

	Fnum = (3 ** .33333333) * BRsqd + F ** .666666666666
	Fdenom = (3 ** .66666666) * (F ** .333333333)
	ret = Fnum/Fdenom
	return ret.real

@jit
def lL(background, data, Rsqd, sigma_xi2, current, T, Nch):
	retval = 0
	sigma_b2 = Rsqd * sigma_xi2
	
	for t in range(T):
		if t:
			retval += ((background[t] - background[t - 1]) ** 2) / (2 * sigma_b2)
		tmp = 0

		for i in range(Nch + 1):
			tmp += np.exp((-1 * (data[t] - background[t] - i * current) ** 2)/(2 * sigma_xi2))
		retval -= np.log(tmp)
	
	retval += 0.5 * (T - 1) * np.log(sigma_xi2)

	tmp = 0.5 * (np.sqrt(Rsqd + 4) + np.sqrt(Rsqd))

	retval += (T - 1) * np.log(tmp)
	return retval

if __name__ == "__main__":
	
	test_data_len = 5000000
	#test_data = np.zeros(test_data_len)
	

	test_data = np.random.rand(test_data_len)
	#test_data = np.arange(5)
	test_data_slant = np.arange(1,test_data_len+1,1)
	test_data = test_data + test_data_slant/(test_data_len/2)

	for i in range(test_data_len):
		if i % 100 < 10 and (i % 10000) < 5000:
			test_data[i] += 5

	Rsqd = 0.001
	sigma_xi2 = 0.1
	current = 5
	multiplier = 1
	updateR2 = False
	Nch = 1

	baseline = pybaseline(test_data, Rsqd, sigma_xi2, current, multiplier, updateR2, Nch)

	plt.plot(test_data)
	plt.plot(baseline)

	plt.show()


