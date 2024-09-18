import numpy as np
import tangent
from scipy.optimize import fsolve

def add_rows_between_largest_gap (array, M):

	diff        = np.diff (array, axis=0)
	distances   = np.linalg.norm (diff, axis=1)
	max_dists   = np.argmax (distances)
	insert_rows = np.linspace (array[max_dists], array[max_dists+1], M)
	array       = np.insert (array, max_dists+1, insert_rows[1:-1], axis=0)

	return array, max_dists

def add_rows_at_index (array, idx, M):

	insert_rows = np.linspace (array[idx], array[idx+1], M)
	array       = np.insert (array, idx+1, insert_rows[1:-1],axis=0)

	return array

def remove_close_rows(array, threshold=1e-6):
	kept_indices   = []
	filtered_array = np.empty ((0,array.shape[1]))
	for i, elem in enumerate(array):
		if i == 0:
			filtered_array = np.vstack((filtered_array, elem))
			kept_indices.append(i)
			continue
		else:
			sieve = (np.linalg.norm(filtered_array - elem, axis=1) < threshold).any()
			if sieve:
				continue
			else:
				filtered_array = np.vstack((filtered_array, elem))
				kept_indices.append(i)

	return filtered_array, np.array(kept_indices)

def crit_condition (vs, vc, vp, phi_p, phi_s, chi_sc, chi_ps, chi_pc):

	phi_c = 1-phi_p-phi_s
	t1    = 1/(vc*phi_c) + 1/(phi_p*vp) - 2*chi_pc
	t2    = (1/(vc*(phi_c)**2) - 1/(vs*phi_s**2))*(1/(vc*(phi_c)) + 1/(phi_p*vp) - 2*chi_pc) + (1/(vc*phi_c) + 1/(vs*phi_s) - 2*chi_sc)/(vc*(phi_c)**2) - 2*(1/(vc*phi_c) - chi_pc - chi_sc + chi_ps)/(vc*(phi_c)**2)

	u1    = (1/(vc*phi_c) + 1/(phi_p*vp) - 2*chi_pc)/(vc*(phi_c)**2) + (1/(vc*phi_c**2) - 1/(phi_p**2 * vp))*(1/(vc*phi_c) + 1/(vs*phi_s) - 2*chi_sc) - 2*(1/(vc*phi_c) + chi_ps - chi_sc - chi_pc)/(vc*phi_c**2)
	u2    = 1/(vc*phi_c) - chi_pc - chi_sc + chi_ps

	return t1*t2 - u1*u2

def row_exists(array, row):
	"""
	Checks if a row exists in a numpy array.

	Args:
	array: A numpy array.
	row: A numpy array representing the row to search for.

	Returns:
	True if the row exists in the array, False otherwise.
	"""
	if array.shape[1] != row.shape[0]:
		raise ValueError("Row dimension mismatch")

	return np.any(np.all(array == row, axis=1))


neighbors     = np.array([[1,0], [1,1], [0,1], [-1,1], [-1,0], [-1,-1], [0, -1], [1, -1]], dtype=int)

def annex_islands(my_island, stable_points, loc):
	options = []
	for n in neighbors:
		if row_exists(stable_points, loc+n) and (not row_exists(my_island, loc+n)): # and (loc+n>=0).all():
			my_island = np.vstack((my_island,(loc+n)))
			options.append(loc+n)
	return np.array(options), my_island

def annex_islands_with_options(my_island, stable_points, options):
	new_options = np.empty((0,2), dtype=int)
	for o in options:
		for n in neighbors:
			if row_exists(stable_points, o+n) and (not row_exists(my_island, o+n)) and (not row_exists(new_options, o+n)): # and (o+n>=0).all():
				my_island   = np.vstack((my_island,o+n))
				new_options = np.vstack((new_options,o+n))

	return new_options, my_island

def find_islands(mesh):
	"""
	Finds and returns islands of ones in a 2D numpy mesh.

	Args:
	mesh: A 2D numpy array of ones and zeros.

	Returns:
	A list of lists, where each sublist contains the coordinates of a single island.
	"""
	islands = []
	# get the indices to loop over
	stable_points = np.argwhere(mesh==1)
	print(f"Length of points = {len(stable_points)}", flush=True)
	print("==========================", flush=True)

	if len(stable_points) == 0:
		return [] 
	else: 
		loop   = True
		spoint = stable_points[0]
		print(f"starter point = {spoint}", flush=True)
		print(f"stable_points[-1] = {stable_points[-1]}", flush=True)


		while loop:
			my_island = np.array([spoint], dtype=int)
			options, my_island = annex_islands(my_island, stable_points, spoint)
			while len(options)>0:
				options, my_island = annex_islands_with_options(my_island, stable_points, options)
				# print(f"my_island = {my_island[0:5],my_island[-5:]} with len(islands) = {len(islands)}", flush=True)
			
			to_del = []
			for midx, isl in enumerate(my_island):
				check = np.logical_and(isl[0]==stable_points[:,0], isl[1]==stable_points[:,1])
				if check.any():
					to_del.append(np.arange(len(stable_points))[check][0])
				# for sidx, sp in enumerate(stable_points):
				#	if isl[0] == sp[0] and isl[1] == sp[1]:
				#		to_del.append(sidx)
				#		continue 
			stable_points = np.delete(stable_points, to_del, axis=0)

			# eliminate all points in islands from stable_points
			# print(f"Cleaning up stable points...")
			# mask = np.isin(stable_points, my_island).all(axis=1)
			# stable_points = stable_points[~mask]

			print(f"================================")
			print(f"stable_points = {len(stable_points)}...", flush=True)
			print(f"================================")

			if len(stable_points) > 0:
				spoint = stable_points[0]
			else:
				loop = False
			islands.append(my_island)

		return islands

def find_crit_point (vs, vc, vp, chi_sc, chi_ps, chi_pc, root_up_p, root_up_s, root_lo_p, root_lo_s):

	def send_to_fsolve_r1 (phi_s):
		phi_p_upper = root_up_s (phi_s)
		return crit_condition (vs, vc, vp, phi_p_upper, phi_s, chi_sc, chi_ps, chi_pc)

	def send_to_fsolve_r2 (phi_s):
		phi_p_lower = root_lo_s (phi_s)
		return crit_condition (vs, vc, vp, phi_p_lower, phi_s, chi_sc, chi_ps, chi_pc)

	def send_to_fsolve_r3 (phi_p):
		phi_s_upper = root_up_p (phi_p)
		return crit_condition (vs, vc, vp, phi_p, phi_s_upper, chi_sc, chi_ps, chi_pc)

	def send_to_fsolve_r4 (phi_p):
		phi_s_lower = root_lo_p (phi_p)
		return crit_condition (vs, vc, vp, phi_p, phi_s_lower, chi_sc, chi_ps, chi_pc)

	guesses = np.linspace (0, 1, 10000)
	roots_up   = np.empty ((0,2))
	roots_down = np.empty ((0,2))

	print(f"Solving for critical points in four sweeps. ", flush=True)
	print("\tIn sweep one...", flush=True)
	for g in guesses:
		root = fsolve (send_to_fsolve_r1, g)

		if abs(send_to_fsolve_r1(root)) < 1e-6:

			if root >= 1 or root <= 0 or np.isnan(root):
				pass
			else:
				r_up  = root_up_s (root)[0]
				r_tup = np.array([root[0], r_up])
				if (r_tup >= 1).any() or (r_tup <= 0).any() or np.sum(r_tup) >= 1 or np.isnan(r_up):
					pass

				elif r_tup in roots_up:
					pass

				else:
					if len(roots_up) == 0:
						roots_up = np.vstack ((roots_up,r_tup))
					else:
						similarity = (np.linalg.norm(roots_up - r_tup, axis=1) < 1e-3).any()
						if similarity:
							pass
						else:
							roots_up = np.vstack ((roots_up,r_tup))   
		else:
			pass

	print ("\tIn sweep two...", flush=True)
	for g in guesses:
		root = fsolve (send_to_fsolve_r3, g)

		if abs(send_to_fsolve_r3(root)) < 1e-6:

			if root >= 1 or root <= 0 or np.isnan(root):
				pass
			else:
				r_up  = root_up_p (root)[0]
				r_tup = np.array([r_up, root[0]])
				if (r_tup >= 1).any() or (r_tup <= 0).any() or np.sum(r_tup) >= 1 or np.isnan(r_up):
					pass

				elif r_tup in roots_up:
					pass

				else:
					if len(roots_up) == 0:
						roots_up = np.vstack ((roots_up,r_tup))
					else:
						similarity = (np.linalg.norm(roots_up - r_tup, axis=1) < 1e-3).any()
						if similarity:
							pass
						else:
							roots_up = np.vstack ((roots_up,r_tup))   
		else:
			pass

	print ("\tIn sweep three...", flush=True)
	for g in guesses:
		root = fsolve (send_to_fsolve_r2, g)

		if abs(send_to_fsolve_r2(root)) < 1e-6:

			if root >= 1 or root <= 0 or np.isnan(root):
				pass
			else:
				r_lo = root_lo_s(root)[0]
				r_tup = np.array([root[0], r_lo])
				if (r_tup >= 1).any() or (r_tup <= 0).any() or np.sum(r_tup) >= 1 or np.isnan(r_lo):
					pass

				elif r_tup in roots_down:
					pass

				else: 
					if len(roots_down) == 0:
						roots_down = np.vstack ((roots_down,r_tup))
					else:
						similarity = (np.linalg.norm(roots_down - r_tup, axis=1) < 1e-3).any ()
						if similarity:
							pass
						else:
							roots_down = np.vstack ((roots_down,r_tup))

		else:
			pass

	print ("\tIn sweep four...", flush=True)
	for g in guesses:
		root = fsolve (send_to_fsolve_r4, g)

		if abs(send_to_fsolve_r4(root)) < 1e-6:

			if root >= 1 or root <= 0 or np.isnan(root):
				pass
			else:
				r_lo = root_lo_p(root)[0]
				r_tup = np.array([r_lo, root[0]])
				if (r_tup >= 1).any() or (r_tup <= 0).any() or np.sum(r_tup) >= 1 or np.isnan(r_lo):
					pass

				elif r_tup in roots_down:
					pass

				else:
					if len(roots_down) == 0:
						roots_down = np.vstack ((roots_down,r_tup))
					else:
						similarity = (np.linalg.norm(roots_down - r_tup, axis=1) < 1e-3).any ()
						if similarity:
							pass
						else:
							roots_down = np.vstack ((roots_down,r_tup))

		else:
			pass

	print(f"roots_up = {roots_up}")
	print(f"roots_down = {roots_down}")
	# check for consistency
	# First determinant
	def first_det(phi_s, phi_p):
		phi_c = 1-phi_s-phi_p
		t1 = (1/(vp*phi_p) + 1/(vc * (1 - phi_p - phi_s)) - 2*chi_pc) 
		t2 = (1/(vc * phi_c**2) - 1/(vs * phi_s**2))*(1/(vp*phi_p) + 1/(vc*phi_c) - 2 * chi_pc) + (1/(vc * phi_c) + 1/(vs * phi_s) - 2*chi_sc)/(vc * phi_c**2) - 2*(1/(vc * phi_c) - chi_pc + chi_ps - chi_sc)/(vc * phi_c**2) 
		t3 = 1/(vc * phi_c) - chi_pc + chi_ps - chi_sc 
		t4 = (1/(vp * phi_p) + 1/(vc * phi_c) - 2 * chi_pc)/(vc * phi_c**2) + (-1/(vp * phi_p**2) + 1/(vc * phi_c**2)) * (1/(vc * phi_c) + 1/(vs * phi_s) - 2*chi_sc) - 2*(1/(vc * phi_c) - chi_pc + chi_ps - chi_sc)/(vc * phi_c**2)
		return t1*t2 - t3*t4
	
	def second_det(phi_s, phi_p):
		phi_c = 1-phi_s-phi_p
		t1 = 1/(vc * phi_c) + 1/(vs * phi_s) - 2*chi_sc
		t2 = (1/(vp*phi_p) + 1/(vc * phi_c) - 2 * chi_pc)/(vc * phi_c ** 2) + (-1/(vp * phi_p**2) + 1/(vc * phi_c**2))*(1/(vc*phi_c) + 1/(vs*phi_s) - 2*chi_sc) - 2*(1/(vc * phi_c) - chi_pc + chi_ps - chi_sc)/(vc * phi_c ** 2)
		t3 = 1/(vc * phi_c) - chi_pc + chi_ps - chi_sc 
		t4 = (1/(vc*phi_c**2) - 1/(vs*phi_s**2))*(1/(vp*phi_p) + 1/(vc * phi_c) - 2*chi_pc) + (1/(vc*phi_c) + 1/(vs*phi_s) - 2*chi_sc)/(vc * phi_c**2) - 2*(1/(vc * phi_c) - chi_pc + chi_ps - chi_sc)/(vc * phi_c**2)
		return t1*t2 - t3*t4

	print(f"first_det_up  = {first_det (roots_up[:,0], roots_up[:,1])}")
	print(f"second_det_up = {second_det(roots_up[:,0], roots_up[:,1])}")

	check1_up = np.abs(first_det (roots_up[:,0], roots_up[:,1])) < 1e-4
	check2_up = np.abs(second_det(roots_up[:,0], roots_up[:,1])) < 1e-4
	print(f"up_check = {check1_up}")
	print(f"up_check = {check2_up}")
	roots_up = roots_up[np.logical_and(check1_up, check2_up)]
	print(f"roots_up = {roots_up}")

	print(f"first_det_up  = {first_det (roots_down[:,0], roots_down[:,1])}")
	print(f"second_det_up = {second_det(roots_down[:,0], roots_down[:,1])}")

	check1_down = np.abs(first_det (roots_down[:,0], roots_down[:,1])) < 1e-4
	check2_down = np.abs(second_det(roots_down[:,0], roots_down[:,1])) < 1e-4
	print(f"check1_down = {check1_down}")
	print(f"check2_down = {check2_down}")
	roots_down  = roots_down[np.logical_and(check1_down, check2_down)]
	print(f"roots_down = {roots_down}")

	print("Complete critical point sweeps!", flush=True)
	return roots_up, roots_down

#############################################################################################################

def stab_crit (p_s, p_p, vs, vc, vp, c_ps, c_pc, c_sc):
	return (1/(vp*p_p) + 1/(vc*(1-p_s - p_p)) - 2 * c_pc) * (1/(vs*p_s) + \
	1/(vc*(1-p_s - p_p)) - 2 * c_sc) - (1/(vc*(1-p_s-p_p)) + c_ps - c_pc - c_sc) ** 2

#############################################################################################################

def embelish(ax, tern_b):

	if tern_b:

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

	else:
		ax.set_xlabel ("$\\phi _{S}$")
		ax.set_ylabel ("$\\phi _{P}$")
		ax.set_xlim   (0,1)
		ax.set_ylim   (0,1)

	return

#############################################################################################################

def add_tang_norm(ax, tang_b, tern_b, crit_b, crits, vs, vc, vp, chi_pc, chi_ps, chi_sc, root_up_s, root_lo_s):

	if tang_b and crit_b:
		cols = ["gold", "limegreen", "pink", "maroon"]
		for idx,crit in enumerate(crits):    
			idx = idx%len(crits)
			slope               = tangent.tangent2 (vs, vc, vp, crit[0], crit[1], chi_pc, chi_ps, chi_sc, root_up_s, root_lo_s)
			if np.isnan(slope) or np.isinf(slope):
				tangent_vector = np.array([0,1], dtype=np.float64)
				normal_vector  = np.array([1,0], dtype=np.float64)
			elif abs(slope) < 1e-12:
				tangent_vector = np.array([1,0], dtype=np.float64)
				normal_vector  = np.array([0,1], dtype=np.float64)		
			else:
				perp_slope          = -1/slope
				tangent_vector      = np.array([1, slope]) / np.sqrt(1+slope**2)
				normal_vector       = np.array([1, perp_slope]) / np.sqrt(1+perp_slope**2)
			print(f"slope = {slope}")
			print(f"tangent vec = {tangent_vector}")
			print(f"normal vector = {normal_vector}")
			
			points_along_tangent = lambda L: np.array([crit[0] + L * tangent_vector[0], crit[1] + L * tangent_vector[1]])
			points_along_normal  = lambda L: np.array([crit[0] + L * normal_vector [0], crit[1] + L * normal_vector[1] ])

			l = np.linspace (-10, 10, 100)
			if tern_b:
				# ax.plot (points_along_tangent(l)[0], 1-points_along_tangent(l)[0]-points_along_tangent(l)[1], points_along_tangent(l)[1], c=cols[idx%len(cols)], lw=1)
				ax.plot (points_along_normal(l)[0],  1-points_along_normal(l)[0]-points_along_normal(l)[1],   points_along_normal(l)[1],  c=cols[idx%len(cols)], lw=1)

			else:
				ax.plot (points_along_tangent(l)[0],  points_along_tangent(l)[1],  c=cols[idx%len(cols)], lw=1)
				ax.plot (points_along_normal (l)[0],  points_along_normal(l) [1],  c=cols[idx%len(cols)], lw=1)

	return

#############################################################################################################

def plot(ax, tern_b, edges_b, crits_b, crits, chi_ps, chi_pc, chi_sc, p_s, p_p, cols, root_up_p, root_lo_p, root_up_s, root_lo_s):

	if tern_b:
		ax.scatter(p_s, 1-p_p-p_s, p_p, s=1, color=cols)

		if crits_b and not(crits is None):
			ax.scatter(crits[:,0], 1-crits[:,0]-crits[:,1], crits[:,1], color='darkred', edgecolors='darkred',s=4, zorder=15)
			# ax.scatter(np.mean(crits, axis=0)[0], 1-np.mean(crits, axis=0)[0]-np.mean(crits, axis=0)[1], np.mean(crits, axis=0)[1], color="limegreen", edgecolors="limegreen", s=8, zorder=15)

		else:
			pass

		if edges_b:
			meshsize            = 1000
			phi_s               = np.linspace (0.001, 1-0.001, meshsize*10)
			chi_ps              = chi_ps
			chi_pc              = chi_pc
			chi_sc              = chi_sc

			r1 = root_up_s (p_s)
			r2 = root_lo_s (p_s)

			to_keep_1 = (~np.isnan(r1)) * (r1 <= 1) * (r1 >= 0)
			r1        = r1[to_keep_1]

			to_keep_2 = (~np.isnan(r2)) * (r2 <= 1) * (r2 >= 0)
			r2        = r2[to_keep_2]

			# Plot the points
			ax.scatter (p_s[to_keep_1], 1-p_s[to_keep_1]-r1, r1, color='seagreen',  s=1)
			ax.scatter (p_s[to_keep_2], 1-p_s[to_keep_2]-r2, r2, color='darkgreen', s=1)

			r3 = root_up_p(p_p)
			r4 = root_lo_p(p_p)

			to_keep_3 = (~np.isnan(r3)) * (r3 <= 1) * (r3 >= 0)
			r3        = r3[to_keep_3]

			to_keep_4 = (~np.isnan(r4)) * (r4 <= 1) * (r4 >= 0)
			r4        = r4[to_keep_4]

			ax.scatter (r3, 1-p_p[to_keep_3]-r3, p_p[to_keep_3], color='seagreen',  s=1)
			ax.scatter (r4, 1-p_p[to_keep_4]-r4, p_p[to_keep_4], color='darkgreen', s=1)

			spinodal_phi_s = np.hstack([p_s[to_keep_1], p_s[to_keep_2], r3, r4])
			spinodal_phi_p = np.hstack([r1, r2, p_p[to_keep_3], p_p[to_keep_4]])

		else:
			spinodal_phi_s = np.array([])
			spinodal_phi_p = np.array([])

	else:
		ax.scatter (p_s, p_p, s=1, color=cols)
		if crits_b and not(crits is None):
			ax.scatter (crits[:,0], crits[:,1], color='darkred', edgecolors='darkred', s=4, zorder=15)
			ax.scatter (np.mean (crits, axis=0)[0], np.mean(crits, axis=0)[1], color='limegreen', edgecolors='limegreen', s=8, zorder=15)
		else:
			pass

		if edges_b:
			meshsize            = 1000
			phi_s               = np.logspace (-6, np.log10(1-1e-6), meshsize*100)
			chi_ps              = chi_ps
			chi_pc              = chi_pc
			chi_sc              = chi_sc

			r1 = root_up_s (p_s)
			r2 = root_lo_s (p_s)

			to_keep_1 = (~np.isnan(r1)) * (r1 <= 1) * (r1 >= 0)
			r1 = r1[to_keep_1]

			to_keep_2 = (~np.isnan(r2)) * (r2 <= 1) * (r2 >= 0)
			r2 = r2[to_keep_2]

			# Plot the points
			ax.scatter(p_s[to_keep_1], r1, color='darkgreen', s=1)
			ax.scatter(p_s[to_keep_2], r2, color='seagreen',  s=1)

			r3 = root_up_p(p_p)
			r4 = root_lo_p(p_p)

			to_keep_3 = (~np.isnan(r3)) * (r3 <= 1) * (r3 >= 0)
			r3        = r3[to_keep_3]

			to_keep_4 = (~np.isnan(r4)) * (r4 <= 1) * (r4 >= 0)
			r4        = r4[to_keep_4]

			# plot the points
			ax.scatter (p_s[to_keep_1], r1, color='darkgreen', s=1)
			ax.scatter (p_s[to_keep_2], r2, color='seagreen',  s=1)

			spinodal_phi_s = np.hstack([p_s[to_keep_1], p_s[to_keep_2], r3, r4])
			spinodal_phi_p = np.hstack([r1, r2, p_p[to_keep_3], p_p[to_keep_4]])

		else:
			spinodal_phi_s = np.array([])
			spinodal_phi_p = np.array([])

	return spinodal_phi_s, spinodal_phi_p

#############################################################################################################


