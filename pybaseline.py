import numpy as np
import cmath
import matplotlib.pyplot as plt
import time
from numba import jit

def pybaseline(data, Rsqd, sigma_xi2, current, multiplier, updateR2, Nch):

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
		
		

		t1 = time.time()
		rhs = set_rhs(p, data, current, Rsqd, size)

		

		t2 = time.time()
		if j % 250 == 0:
			print("j = ", j)

		main_d, lower_d, upper_d = set_matrix(size, Rsqd)

		

		background, main_d, rhs = solveMatrix(size, lower_d, main_d, upper_d, rhs, background)
		
		

		main_d, lower_d, upper_d = set_matrix(size, Rsqd)
		t3 = time.time()
		tmp = A_minus_B(data, background, size)
		
		

		if j > 1:
			if np.sum(p2) > 0:
				current = AdotB(p, tmp, size) / np.sum(p2)

		D2 = AdotB(tmp, tmp, size) - 2 * current * AdotB(tmp, p, size) + current * current * np.sum(p2)

		

		t4 = time.time()
		GRADBSQRD = gradbsquared(background,size)



		t5 = time.time()
		BRsqd = (GRADBSQRD/ ((size - 1) * sigma_xi2)) ** 2

		

		if j > 1 and updateR2 == True:
			Rsqd = updatedRsqd(BRsqd)

		sigma_xi2 = (GRADBSQRD/Rsqd + D2)/(size - 1)
		t6 = time.time()
		logL = lL(background, data,  Rsqd,  sigma_xi2,  current, size, Nch)

		if j == 1:
			print(background)
			print(data)
			print(Rsqd)
			print(sigma_xi2)
			print(current)
			print(size)
			print(Nch)
			print(logL)

		print('logL: ', logL)

		if np.abs(logL - logLold) > 0.0001:
			logLold = logL
		else:
			not_converged = False
		t7 = time.time()
		print("times: ", t7-t6, t6-t5, t5-t4, t4-t3, t3-t2,t2-t1, t1-t0)
	print("converged")

	print('Iteration count: ', j)

	p, p2 = set_n2(data, background, current, Rsqd, size, Nch, sigma_xi2)
	
	tmp = set_reconstruction(p, background, current)
	plt.plot(p)
	tmp = set_res(p, data, background, current)

	plt.show()




	return background

def set_res(p, d, b, current):
	res = d - b - current * p
	return res

def set_reconstruction(p, background, current):
	tmp = background + current * p
	return tmp

def multiply_array(array, b):	
	#array_out = [val * b for val in array] SBW Simplified
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
	"""
	main = np.empty(size)
	lower = np.empty(size)
	upper = np.empty(size)
	for i in range(size):
		main[i] = -1 * Rsqd - 2.0
		lower[i] = 1
		upper[i] = 1

		main[0] = -1 * Rsqd - 1
		main[-1] = -1 * Rsqd - 1
	"""
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

def set_rhs(p, d, current, Rsqd, l):
	"""
	rhs = np.empty(l)
	for j in range(l):
		rhs[j] = -1 * Rsqd * (d[j] - current * p[j])
	"""
	rhs = -1 * Rsqd * (d - current * p)

	return rhs

@jit
def solveMatrix(n, a, b, c, v, x):
	#m = a[1::]/b[0:-1]
	#b[1::] = b[1::] - m * c[0:-1]
	for i in range(1,n):
		m = a[i] / b[i - 1]
		b[i] = b[i] - m * c[i - 1]
		v[i] = v[i] - m * v[i - 1]
		#v[i] = v[i] - m[i - 1] * v[i - 1]

	x[n - 1] = v[n - 1] / b[n - 1]

	for i in range(n-2, -1, -1):
		x[i] = (v[i] - c[i] * x[i+1])/b[i]

	return x, b, v

def A_minus_B(A, B, l):
	#AmB = np.empty(l)
	"""
	for i in range(l):
		AmB[i] = A[i] - B[i]
	"""
	AmB = A - B
	return AmB

def AdotB(A, B, l):
	#ret = 0
	"""
	for i in range(l):
		ret += A[i] * B[i]
	"""
	ret = np.sum(A * B)
	return ret

def gradbsquared(b, l):
	#ret = 0
	#for i in range(1,l):
	#	ret += (b[i] - b[i - 1]) ** 2
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

	"""

	retval_array = ((background[1::] - background[0:-1]) ** 2) / (2 * sigma_b2)
	retval = np.sum(retval_array)

	tmp = 0
	tmp_array_logsum = 0
	for i in range(Nch+1):
		tmp_array = np.exp(-1 * (data - background - i * current) ** 2)/(2 * sigma_xi2)
		tmp += np.sum(tmp_array)
		tmp_array_logsum += np.sum(np.log(tmp_array))

	retval = retval - tmp_array_logsum
	"""
	
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
	
	test_data_len = 50000
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


