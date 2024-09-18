import numpy as np
import matplotlib.pyplot as plt
import FreeEnergy as FE
import linecache 
import pickle
import warnings 
import time
import random
import mpltern
import argparse

parser = argparse.ArgumentParser(description='Create a skeleton solution for the binodal. This is a memory-intensive computation.')
parser.add_argument('--hull', metavar='hull', dest='hull', type=str, action='store', help='location of serialized hull structure.')
args = parser.parse_args()

###################################################
def custom_warning_format(message, category, filename, lineno, line=None):
    line = linecache.getline(filename, lineno).strip()
    return f"There is a RunTimeWarning taking place on line {lineno}.\n"

warnings.formatwarning = custom_warning_format
###################################################

def custom_sort(arrays, wts):
	# Convert the list of arrays into a NumPy array
	arrays = np.array(arrays)
	wts    = np.array(wts)
	arrays = np.hstack((arrays, wts.reshape(-1,1)))

	# Sort the arrays based on the first element
	sorted_indices = np.argsort(arrays[:, 0]) 
	sorted_arrays = arrays[sorted_indices]

	# Check for elements with first elements within 1e-6 of each other
	first_elements = sorted_arrays[:, 0]
	diff = np.abs(np.diff(first_elements))
	split_indices = np.where(diff > 1e-6)[0] + 1 

	# Sort the subarrays based on the second element
	final_sorted_arrays = np.split(sorted_arrays, split_indices)
	final_sorted_arrays = [np.array(sorted(subarr, key=lambda x: x[1])) for subarr in final_sorted_arrays]

	# Concatenate the sorted subarrays
	sorted_arrays = np.concatenate(final_sorted_arrays)
	return sorted_arrays[:,0:2], sorted_arrays[:, -1]

if __name__=="__main__":

	start = time.time()
	fig = plt.figure(figsize=(8,8))
	ax  = fig.add_subplot(projection="ternary")
	ax.set_tlabel ("$\\phi _{S}$") # ('Vol. frac. A')
	ax.set_llabel ("$\\phi _{C}$") # ('Vol. frac. C')
	ax.set_rlabel ("$\\phi _{P}$") # ('Vol. frac. B')
	ax.set_tlim(0, 1)
	ax.set_llim(0, 1)
	ax.set_rlim(0, 1)
	positions = ['tick1', 'tick2']
	for position in positions:
		ax.taxis.set_ticks_position(position)
		ax.laxis.set_ticks_position(position)
		ax.raxis.set_ticks_position(position)

	ax.taxis.set_ticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0])
	ax.raxis.set_ticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0])
	ax.laxis.set_ticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0])

	hull_file = open(args.hull, 'rb')
	F = pickle.load(hull_file)
	hull_file.close() 

	database = open("combined.db", "w")
	single   = open("single.db", "w")
	double   = open("double.db", 'w')
	triple   = open("triple.db", 'w')
	database.write("vs | vc | vp | chi_sc | chi_ps | chi_pc | phi_s | phi_p | label0 | label1 | label2 | phi_s1 | phi_p1 | phi_s2 | phi_p2 | phi_s3 | phi_p3 | w1 | w2 | w3\n")
	single.write("vs | vc | vp | chi_sc | chi_ps | chi_pc | phi_s | phi_p | label0 | label1 | label2 | phi_s1 | phi_p1 | phi_s2 | phi_p2 | phi_s3 | phi_p3 | w1 | w2 | w3\n")
	double.write("vs | vc | vp | chi_sc | chi_ps | chi_pc | phi_s | phi_p | label0 | label1 | label2 | phi_s1 | phi_p1 | phi_s2 | phi_p2 | phi_s3 | phi_p3 | w1 | w2 | w3\n")
	triple.write("vs | vc | vp | chi_sc | chi_ps | chi_pc | phi_s | phi_p | label0 | label1 | label2 | phi_s1 | phi_p1 | phi_s2 | phi_p2 | phi_s3 | phi_p3 | w1 | w2 | w3\n")

	print(f"len(F.num_phases) = {len(F.num_phases)}")
	print(f"Number of triple splits = {np.sum(F.num_phases==3)}")
	print(f"len(F.simplices)  = {len(F.simplices)}")

	nsimplices = len(F.simplices)
	for i in range(nsimplices):
		print(f"nphases = {F.num_phases[i]} @ {i}/{nsimplices}...", flush=True)
		inputs  = f"{F.vs} | {F.vc} | {F.vp} | {F.chi_sc} | {F.chi_ps} | {F.chi_pc} | "
		if F.num_phases[i] == 1:
			continue
			labels   = "1 | 0 | 0 | "
			weights  = "1 | 0 | 0\n"
			simplex  = F.surface[F.simplices[i]]
			vertices = F.from_xy_to_phi(simplex)
			point    = np.mean(vertices, axis=0)
			comp     = f"{point[0]} | {point[1]} | "
			splits   = f"{point[0]} | {point[1]} | 0 | 0 | 0 | 0 | "
			line     = inputs + comp + labels + splits + weights 
			single.write(line) 
		elif F.num_phases[i] == 2:
			labels       = "0 | 1 | 0 | "
			simplex      = F.surface[F.simplices[i]]
			vertices     = F.from_xy_to_phi(simplex)
			weights      = np.random.uniform(low=0, high=1, size=3)
			weights      = weights/np.sum(weights) 
			point        = np.sum(weights.reshape(-1,1)*vertices, axis=0)
			# point        = np.mean(vertices, axis=0)
			comp         = f"{point[0]} | {point[1]} | "
			results      = F.calc_phase_split_specific(point, i)
			phase_comps, phase_wts = custom_sort(results[1][0:2, 0:2], results[2][0:2])
			splits      = f"{phase_comps[0][0]} | {phase_comps[0][1]} | {phase_comps[1][0]} | {phase_comps[1][1]} | 0 | 0 | "
			weights     = f"{phase_wts[0]} | {phase_wts[1]} | 0\n"
			line        = inputs + comp + labels + splits + weights 
			double.write(line)
			p1 = phase_comps[0]
			p2 = phase_comps[1]
			if np.linalg.norm(p1-p2) < 1e-3:
				print(f"too close!")
				pass 
			else:
				ax.plot([p1[0],p2[0]], [1-p1[0]-p1[1], 1-p2[0]-p2[1]], [p1[1], p2[1]], c='coral', lw=0.5, ls='--', alpha=0.1)
			
		elif F.num_phases[i] == 3:
			labels   = "0 | 0 | 1 | "
			simplex  = F.surface[F.simplices[i]]
			vertices = F.from_xy_to_phi(simplex)
			weights      = np.random.uniform(low=0, high=1, size=3)
			weights      = weights/np.sum(weights) 
			point        = np.sum(weights.reshape(-1,1)*vertices, axis=0)
			comp     = f"{point[0]} | {point[1]} | "
			results  = F.calc_phase_split_specific(point, i)
			phase_comps, phase_wts = custom_sort(results[1], results[2])
			splits      = f"{phase_comps[0][0]} | {phase_comps[0][1]} | {phase_comps[1][0]} | {phase_comps[1][1]} | {phase_comps[2][0]} | {phase_comps[2][1]} | "
			weights     = f"{phase_wts[0]} | {phase_wts[1]} | {phase_wts[2]}\n"
			line        = inputs + comp + labels + splits + weights 
			triple.write(line)
		else:
			print(f"Number of splits is {F.num_phases[i]}. Exiting...")
			exit() 
		database.write(line) 

	# end of simplices loop     
	database.close()
	single.close()
	double.close()
	triple.close()
	fig.savefig(f"raw.png", dpi=1000, bbox_inches="tight")

	stop = time.time()
	print(f"Time for database creation is {stop-start} seconds.", flush=True)
