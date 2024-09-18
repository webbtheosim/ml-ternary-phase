import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import FreeEnergy as FE
import linecache 
import pickle
import warnings 
import time
import mu
import random
import mpltern
import argparse
import ternary
import random
from scipy.optimize import fsolve
import os
from scipy.spatial.distance import pdist 
from scipy.stats import mode

parser = argparse.ArgumentParser(description='Create a skeleton solution for the binodal. This is a memory-intensive computation.')
parser.add_argument('--hull', metavar='hull', dest='hull', type=str, action='store', help='location of serialized hull structure.')
parser.add_argument('--deposit', metavar='dep', dest='deposit', type=str, action='store', help='location to deposit database.')
args = parser.parse_args()

###################################################
def custom_warning_format(message, category, filename, lineno, line=None):
    line = linecache.getline(filename, lineno).strip()
    return f"There is a RunTimeWarning taking place on line {lineno}.\n"

warnings.formatwarning = custom_warning_format
###################################################

def custom_sort(arrays):
	# Convert the list of arrays into a NumPy array
	arrays = np.array(arrays)

	# Sort the arrays based on the first element
	sorted_indices = np.argsort(arrays[:, 0]) 
	sorted_arrays = arrays[sorted_indices]

	# Check for elements with first elements within 1e-6 of each other
	first_elements = sorted_arrays[:, 0]
	diff           = np.abs(np.diff(first_elements))
	split_indices  = np.where(diff > 1e-6)[0] + 1 

	# Sort the subarrays based on the second element
	final_sorted_arrays = np.split(sorted_arrays, split_indices)
	final_sorted_arrays = [np.array(sorted(subarr, key=lambda x: x[1])) for subarr in final_sorted_arrays]

	# Concatenate the sorted subarrays
	sorted_arrays = np.concatenate(final_sorted_arrays)

	return sorted_arrays[:,0:2]

def triangle_area(p1, p2, p3):
	area = 0.5 * np.abs(np.cross(p2-p1, p3-p1))
	return area 

def generate_points(triangle, N):
	# print(f"triangle = {triangle}", flush=True)
	# Generate random barycentric coordinates
	r1 = np.random.rand(N)
	r2 = np.random.rand(N)

	# Ensure the points lie within the triangle by limiting the sum of r1 and r2 to 1
	mask = r1 + r2 > 1
	r1[mask] = 1 - r1[mask]
	r2[mask] = 1 - r2[mask]

	# Calculate the third barycentric coordinate
	r3 = 1 - r1 - r2
	weights = np.array([r1, r2, r3]).T

	# Convert barycentric coordinates to cartesian coordinates
	points = r1.reshape(-1,1) * triangle[0] + r2.reshape(-1,1) * triangle[1] + r3.reshape(-1,1) * triangle[2]
	# print(f"points  = {points}")
	# print(f"weights = {weights}")

	return points, weights

def make(hull_pkl, deposit):
	# fig = plt.figure(figsize=(3,3))
	# ax  = fig.add_subplot(projection="ternary")

	hull_file = open(hull_pkl, 'rb')
	F = pickle.load(hull_file)
	hull_file.close() 

	inputs = dict()
	inputs["vs"] = F.vs
	inputs["vc"] = F.vc
	inputs["vp"] = F.vp 
	inputs["chi_sc"] = F.chi_sc
	inputs["chi_ps"] = F.chi_ps 
	inputs["chi_pc"] = F.chi_pc

	nsimplices = len(F.simplices)
	M = mu.sym_mu_ps(inputs, F.spinodal)

	def dubs_delta_mu(P, phi_s):
		eq1 = M.delta_mu_s(phi_s, P[0], P[1], P[2])
		eq2 = M.delta_mu_p(phi_s, P[0], P[1], P[2])
		eq3 = M.delta_mu_c(phi_s, P[0], P[1], P[2])
		return [eq1, eq2, eq3]
	def trips_delta_mu(P):
		eq1 = M.delta_mu_s(P[0], P[1], P[4], P[5])
		eq2 = M.delta_mu_p(P[0], P[1], P[4], P[5])
		eq3 = M.delta_mu_c(P[0], P[1], P[4], P[5])
		eq4 = M.delta_mu_s(P[2], P[3], P[4], P[5])
		eq5 = M.delta_mu_p(P[2], P[3], P[4], P[5])
		eq6 = M.delta_mu_c(P[2], P[3], P[4], P[5])
		return [eq1, eq2, eq3, eq4, eq5, eq6]

	SINGLE_PHASE       = list()
	EXTRA_SINGLE_PHASE = list()
	DOUBLE_PHASE       = [np.empty((0,2)), np.empty((0,2))]
	TRIPLE_PHASE       = list()

	npoints = 1000

	one_phase_idx   = np.arange(nsimplices)[F.num_phases==1]
	two_phase_idx   = np.arange(nsimplices)[F.num_phases==2]
	three_phase_idx = np.arange(nsimplices)[F.num_phases==3]

	print(f"len(two_phase_idx) = {len(two_phase_idx)}", flush=True)

	np.random.shuffle(one_phase_idx)
	np.random.shuffle(two_phase_idx)
	np.random.shuffle(three_phase_idx)

	print(f"Inside one phase...", end=' ', flush=True)
	for opi in one_phase_idx[:1000]:
		simplex  = F.surface[F.simplices[opi]]
		vertices = F.from_xy_to_phi(simplex)
		point    = np.mean(vertices, axis=0)
		SINGLE_PHASE.append(point[0:2])
	print(f"done!", flush=True, end=' ')

	print(f"Inside two phase...", end=' ', flush=True)
	print(f"len(two_phase_idx) = {len(two_phase_idx)}", flush=True)
	for tpi in two_phase_idx:
		simplex      = F.surface[F.simplices[tpi]]
		vertices     = F.from_xy_to_phi(simplex)
		weights      = np.random.uniform(low=0, high=1, size=3)
		weights      = weights/np.sum(weights) 
		point        = np.sum(weights.reshape(-1,1)*vertices, axis=0)
		results      = F.calc_phase_split_specific(point, tpi)
		phase_comps  = custom_sort(results[1][0:2, 0:2])
		p1_, p2_     = phase_comps[0], phase_comps[1]
		phi_tests    = [p1_[1], p2_[0], p2_[1]]
		root         = fsolve(dubs_delta_mu, phi_tests, args=(p1_[0],))
		p1           = np.array([p1_[0]  , root[0]])
		p2           = np.array([root[1], root[2]])
		if (np.abs(dubs_delta_mu(root, p1[0]))>1e-6).any():
			continue
		elif np.linalg.norm(p1-p2)<1e-3:
			# print(f"dist = {np.linalg.norm(p1_-p2_)}", flush=True)
			EXTRA_SINGLE_PHASE.append(p1_)
		else:
			p = np.array([p1,p2])
			p = custom_sort(p)
			print(f"root = {p[0]}, {p[1]}", flush=True)
			DOUBLE_PHASE[0] = np.vstack((DOUBLE_PHASE[0], p[0]))
			DOUBLE_PHASE[1] = np.vstack((DOUBLE_PHASE[1], p[1]))
	print(f"done!", flush=True)

	print(f"Inside three phase with {len(three_phase_idx)}...", flush=True, end=' ')
	for thpi in three_phase_idx:
		simplex      = F.surface[F.simplices[thpi]]
		vertices     = F.from_xy_to_phi(simplex)
		weights      = np.random.uniform(low=0, high=1, size=3)
		weights      = weights/np.sum(weights) 
		point        = np.sum(weights.reshape(-1,1)*vertices, axis=0)
		results      = F.calc_phase_split_specific(point, thpi)
		phase_comps  = custom_sort(results[1])
		p1, p2, p3   = phase_comps[0], phase_comps[1], phase_comps[2]
		phi_tests    = [p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]]
		root         = fsolve(trips_delta_mu, phi_tests)
		print(f"root = {root}")
		p1           = np.array([root[0], root[1]])
		p2           = np.array([root[2], root[3]])
		p3           = np.array([root[4], root[5]])
		area         = triangle_area(p1, p2, p3)
		print(f"err = {trips_delta_mu(root)}")
		if (np.abs(trips_delta_mu(root)) > 1e-5).any():
			continue
		elif np.linalg.norm(p1-p2)<1e-6 or np.linalg.norm(p1-p3)<1e-6 or np.linalg.norm(p2-p3)<1e-6 or area < 1e-3:
			continue
		else:
			print(f"Good root!")
			p = np.array([p1,p2,p3])
			p = custom_sort(p) 
			TRIPLE_PHASE.append(p)
	print(f"done!", flush=True)

	SINGLE_PHASE          = np.array(SINGLE_PHASE)
	if len(EXTRA_SINGLE_PHASE) > 0:
		EXTRA_SINGLE_PHASE = np.array(EXTRA_SINGLE_PHASE)
		if len(EXTRA_SINGLE_PHASE) >=50:
			SINGLE_PHASE          = np.vstack((SINGLE_PHASE, EXTRA_SINGLE_PHASE[::len(EXTRA_SINGLE_PHASE)//50]))
		else:
			SINGLE_PHASE          = np.vstack((SINGLE_PHASE, EXTRA_SINGLE_PHASE))
	DOUBLE_PHASE[0], keep = ternary.remove_close_rows(DOUBLE_PHASE[0], 1e-9)
	if len(keep) > 0:
		DOUBLE_PHASE[1]       = DOUBLE_PHASE[1][keep]
	DOUBLE_PHASE[1], keep = ternary.remove_close_rows(DOUBLE_PHASE[1], 1e-9)
	if len(keep) > 0:
		DOUBLE_PHASE[0]       = DOUBLE_PHASE[0][keep]

	# start creating the database
	database = open(deposit+"/"+f"chisc_{F.chi_sc}-chips_{F.chi_ps}-chipc_{F.chi_pc}-vs_{F.vs}-vp_{F.vp}-vc_{F.vc}.db", "w")
	database.write("vs | vc | vp | chi_sc | chi_ps | chi_pc | phi_s | phi_p | label0 | label1 | label2 | phi_s1 | phi_p1 | phi_s2 | phi_p2 | phi_s3 | phi_p3 | w1 | w2 | w3\n")

	# only use 10k points from single phase 
	np.random.shuffle(SINGLE_PHASE)
	if npoints > len(SINGLE_PHASE):
		tot_loop = len(SINGLE_PHASE)
	else:
		tot_loop = npoints

	inputs = f"{F.vs} | {F.vc} | {F.vp} | {F.chi_sc} | {F.chi_ps} | {F.chi_pc} | "
	for i in range(tot_loop):
		labels  = "1 | 0 | 0 | "
		weights = "1 | 0 | 0\n"
		comp    = f"{SINGLE_PHASE[i][0]} | {SINGLE_PHASE[i][1]} | "
		splits  = f"{SINGLE_PHASE[i][0]} | {SINGLE_PHASE[i][1]} | 0 | 0 | 0 | 0 | "
		line    = inputs + comp + labels + splits + weights 
		database.write(line)
	
	if len(DOUBLE_PHASE[0]) != 0:
		if npoints > len(DOUBLE_PHASE[0]):
			tot_loop = len(DOUBLE_PHASE[0])
		else:
			tot_loop = npoints
		rindices = np.arange(tot_loop)
		np.random.shuffle(rindices)
		along_binodal = npoints//tot_loop
		if along_binodal == 0:
			along_binodal = 1
		print(f"along_binodal = {along_binodal}", flush=True)
		for i in rindices:
			for _ in range(along_binodal):
				labels       = "0 | 1 | 0 | "
				weights      = np.random.uniform(low=0, high=1, size=2)
				weights      = weights/np.sum(weights) 
				vertices     = np.array([DOUBLE_PHASE[0][i], DOUBLE_PHASE[1][i]])
				point        = np.sum(weights.reshape(-1,1)*vertices, axis=0)
				comp         = f"{point[0]} | {point[1]} | "
				splits       = f"{DOUBLE_PHASE[0][i][0]} | {DOUBLE_PHASE[0][i][1]} | {DOUBLE_PHASE[1][i][0]} | {DOUBLE_PHASE[1][i][1]} | 0 | 0 | "
				weights      = f"{weights[0]} | {weights[1]} | 0\n"
				line         = inputs + comp + labels + splits + weights 
				database.write(line)
	
	ntriangle        = len(TRIPLE_PHASE)
	if ntriangle > 0:

		print(f"Number of triangles = {ntriangle}", flush=True)
		pts_per_triangle = npoints // ntriangle 
		print(f"Points per triangle is {pts_per_triangle}.", flush=True)
		labels = "0 | 0 | 1 | "
		for i in range(ntriangle):
			# ax.scatter(TRIPLE_PHASE[i][:,0], 1-TRIPLE_PHASE[i][:,0]-TRIPLE_PHASE[i][:,1], TRIPLE_PHASE[i][:,1], c="gold", edgecolors='k', s=1, zorder=10)
			pts, wts = generate_points(TRIPLE_PHASE[i], pts_per_triangle)
			for j in range(pts_per_triangle):
				comp    = f"{pts[j][0]} | {pts[j][1]} | "
				splits  = f"{TRIPLE_PHASE[i][0][0]} | {TRIPLE_PHASE[i][0][1]} | {TRIPLE_PHASE[i][1][0]} | {TRIPLE_PHASE[i][1][1]} | {TRIPLE_PHASE[i][2][0]} | {TRIPLE_PHASE[i][2][1]} | "
				weights = f"{wts[j][0]} | {wts[j][1]} | {wts[j][2]}\n"
				line    = inputs + comp + labels + splits + weights
				database.write(line)
	else:
		print(f"No triangles.", flush=True)
	
	# end of writing loop    
	database.close()

	# ax.scatter(SINGLE_PHASE[:,0], 1-SINGLE_PHASE[:,0]-SINGLE_PHASE[:,1], SINGLE_PHASE[:,1], c='steelblue', s=0.5)
	# ax.scatter(DOUBLE_PHASE[0][:,0], 1-DOUBLE_PHASE[0][:,0]-DOUBLE_PHASE[0][:,1], DOUBLE_PHASE[0][:,1], c='coral', s=0.5)
	# ax.scatter(DOUBLE_PHASE[1][:,0], 1-DOUBLE_PHASE[1][:,0]-DOUBLE_PHASE[1][:,1], DOUBLE_PHASE[1][:,1], c='salmon', s=0.5)
	# fig.savefig("splits.png", dpi=1000, bbox_inches="tight")

	return

if __name__=="__main__":

	start = time.time()

	make(args.hull, args.deposit)

	stop = time.time()
	print(f"Time for database creation is {stop-start} seconds.", flush=True)
