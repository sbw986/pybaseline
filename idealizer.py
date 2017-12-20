import numpy as np
import cmath
import matplotlib.pyplot as plt

def run(data, Rsqd, sigma_xi2, current, multiplier, updateR2, Nch):

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
		j += 1

		p, p2 = set_n2(data, background, current, Rsqd, size, Nch, sigma_xi2)

		rhs = set_rhs(p, data, current, Rsqd, size)

		if j % 250 == 0:
			print("j = ", j)

		main_d, lower_d, upper_d = set_matrix(size, Rsqd)
		background, main_d, rhs = solveMatrix(size, lower_d, main_d, upper_d, rhs, background)
		main_d, lower_d, upper_d = set_matrix(size, Rsqd)

		tmp = A_minus_B(data, background, size)

		if j > 1:
			if sum(p2,size) > 0:
				current = AdotB(p, tmp, size) / sum(p2, size)

		D2 = AdotB(tmp, tmp, size) - 2 * current * AdotB(tmp, p, size) + current * current * sum(p2, size)

		GRADBSQRD = gradbsquared(background,size)
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

	return background


def multiply_array(array, b):	
	array_out = [val * b for val in array]
	return array_out

def init_b(data, current):
	davg = np.mean(data)
	b = np.empty(len(data))
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

	return b

def set_matrix(size, Rsqd):
	main = np.empty(size)
	lower = np.empty(size)
	upper = np.empty(size)
	for i in range(size):
		main[i] = -1 * Rsqd - 2.0
		lower[i] = 1
		upper[i] = 1

		main[0] = -1 * Rsqd - 1
		main[-1] = -1 * Rsqd - 1
	return main, lower, upper

def set_n2(d, background, current, Rsqd, T, Nch, sigma_xi2):
	n = np.empty(T)
	n2 = np.empty(T)
	for t in range(T):
		eta = np.empty(Nch)
		for j in range(Nch):
			eta[j] = (d[t] - background[t] - j * current) * (d[t] - background[t] - j * current)/(2 * sigma_xi2)

		sum_ = 0
		sum_2 = 0

		for j in range(Nch):
			denom = 0
			for k in range(Nch):
				denom += np.exp(eta[j] - eta[k])
			sum_ += j/denom
			sum_2 += j*j/denom

		n[t] = sum_
		n2[t] = sum_2
	return n, n2

def set_rhs(p, d, current, Rsqd, l):
	rhs = np.empty(l)
	for j in range(l):
		rhs[j] = -1 * Rsqd * (d[j] - current * p[j])
	return rhs

def solveMatrix(n, a, b, c, v, x):
	for i in range(1,n):
		m = a[i] / b[i - 1]
		b[i] = b[i] - m * c[i - 1]
		v[i] = v[i] - m * v[i - 1]

	x[n - 1] = v[n - 1] / b[n - 1]

	for i in range(n-2, -1, -1):
		x[i] = (v[i] - c[i] * x[i+1])/b[i]

	return x, b, v

def A_minus_B(A, B, l):
	AmB = np.empty(l)
	for i in range(l):
		AmB[i] = A[i] - B[i]
	return AmB

def AdotB(A, B, l):
	ret = 0
	for i in range(l):
		ret += A[i] * B[i]
	return ret

def gradbsquared(b, l):
	ret = 0
	for i in range(1,l):
		ret += (b[i] - b[i - 1]) ** 2
	return ret

def updatedRsqd(BRsqd):
	disc = -3 * (BRsqd - 108) * BRsqd * BRsqd
	
	F = 18 * BRsqd + cmath.sqrt(disc)

	Fnum = 3 ** .33333333 * BRsqd + F ** .666666666666
	Fdenom = (3 ** .66666666) * (F ** .333333333)
	ret = Fnum/Fdenom
	return ret.real

def lL(background, data, Rsqd, sigma_xi2, current, T, Nch):
	retval = 0
	sigma_b2 = Rsqd * sigma_xi2

	for t in range(T):
		if t:
			retval += ((background[t] - background[t - 1]) ** 2) / (2 * sigma_b2)
		tmp = 0
		for i in range(Nch):
			tmp += np.exp(-1 * (data[t] - background[t] - i * current) ** 2)/(2 * sigma_xi2)
		retval -= np.log(tmp)

	retval += 0.5 * (T - 1) * np.log(sigma_xi2)

	tmp = 0.5 * (np.sqrt(Rsqd + 4) + np.sqrt(Rsqd))

	retval += (T - 1) * np.log(tmp)

	return retval

if __name__ == "__main__":
	
	test_data_len = 5000

	test_data = np.random.rand(test_data_len)
	test_data_slant = np.arange(0,test_data_len,1)
	test_data = test_data + test_data_slant/(test_data_len/2)

	Rsqd = 0.001
	sigma_xi2 = 0.2
	current = 0.3
	multiplier = 1
	updateR2 = False
	Nch = 1

	baseline = run(test_data, Rsqd, sigma_xi2, current, multiplier, updateR2, Nch)

	plt.plot(test_data)
	plt.plot(baseline)

	plt.show()


