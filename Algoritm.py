
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def td_embedding(data,emb,tau):
	indexes = np.arange(0,emb,1)*tau
	return np.array([data[indexes +i] for i in range(len(data)-(emb-1)*tau)])


def logarithmic_r(min_n, max_n, factor):
	if max_n <= min_n:
		raise ValueError("arg1 has to be < arg2")
	if factor <= 1:
		raise ValueError("factor(arg3) has to be > 1")
	max_i = int(np.floor(np.log(1.0 * max_n / min_n) / np.log(factor)))
	return np.array([min_n * (factor ** i) for i in range(max_i + 1)])


def grassberg_procaccia(data,emb_dim,time_delay,plot = None):
	orbit = td_embedding(data, emb_dim, time_delay)
	n_points = len(orbit)
	data_std = np.std(data)
	r_vals = logarithmic_r(0.1 * data_std, 0.7 * data_std, 1.03)
	distances = np.zeros(shape=(n_points,n_points))
	r_matrix_base = np.zeros(shape=(n_points,n_points))

	for i in range(n_points):
		for j in range(i,n_points):
			distances[i][j] = np.linalg.norm(orbit[i]-orbit[j])
			r_matrix_base[i][j] = 1
	C_r = []
	for r in r_vals:
		r_matrix = r_matrix_base*r
		heavi_matrix = np.heaviside( r_matrix - distances, 0)
		corr_sum = (2/float(n_points*(n_points-1)))*np.sum(heavi_matrix)
		C_r.append(corr_sum)

	gradients = np.gradient(np.log2(C_r),np.log2(r_vals))
	gradients.sort()
	D = np.mean(gradients[-5:])

	if plot:
		plt.plot(np.log2(r_vals),np.log2(C_r))
		plt.xlabel("Distance r")
		plt.ylabel("C(r)")
		plt.title("Correlation sum in log2-log2 plot. Dimension D is "+str(round(D,2)))
		plt.show()
	
	return D

