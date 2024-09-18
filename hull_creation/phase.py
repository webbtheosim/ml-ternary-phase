import numpy as np
import mu
import spinodal

class Phase:
	def __init__(self, inputs, crits=None):
		self.chi_sc    = inputs["chi_sc"]
		self.chi_ps    = inputs["chi_ps"]
		self.chi_pc    = inputs["chi_pc"]
		self.vs        = inputs["vs"]
		self.vc        = inputs["vc"]
		self.vp        = inputs["vp"]
		self.spinodal  = spinodal.Spinodal(inputs)
		self.crits     = self.spinodal.crits
		self.sym_mu_ps = mu.sym_mu_ps(inputs, self.spinodal)
		self.sym_mu_sc = mu.sym_mu_sc(inputs, self.spinodal)
		self.sym_mu_pc = mu.sym_mu_pc(inputs, self.spinodal)
		return
	
	def tangent_tracing_unity(self, BINODALS, uidx):
		
		idx = BINODALS["groupings"][uidx]["alpha"]["idx"]
		print(f"Get the binodal curve...", flush=True, end=' ')

		# start with mu_ps
		print("Binodal run for ps along s2...")
		phi_s1, phi_s2, phi_p1, phi_p2 = self.sym_mu_ps.binodal_run_in_s2(self.crits[idx])

		# if (phi_s1[-1]<1e-2 or phi_p1[-1]<1e-2 or (1-phi_s1[-1]-phi_p1[-1]<1e-2)) and len(phi_s1) > 10:
		if (phi_s1[-1]<1e-9 or phi_p1[-1]<1e-9 or (1-phi_s1[-1]-phi_p1[-1]<1e-9)) and len(phi_s1) > 10:
			# start plotting out the curve
			# ax.scatter(phi_s1, 1-phi_s1-phi_p1, phi_p1, c='gold', s=3)
			# ax.scatter(phi_s2, 1-phi_s2-phi_p2, phi_p2, c='pink', s=3)
			print("Found a complete binodal. Moving on...")
			phi_1 = np.array([phi_s1, phi_p1]).T
			phi_2 = np.array([phi_s2, phi_p2]).T
			BINODALS["groupings"][uidx]["alpha"]["binodals"] = [phi_1, phi_2]
			return 
			

		# do the other run 
		print("Binodal run for ps along p2...")
		phi_s1, phi_s2, phi_p1, phi_p2 = self.sym_mu_ps.binodal_run_in_p2(self.crits[idx])

		# if (phi_s1[-1]<1e-2 or phi_p1[-1]<1e-2 or (1-phi_s1[-1]-phi_p1[-1]<1e-2)) and len(phi_s1) > 10:
		if (phi_s1[-1]<1e-9 or phi_p1[-1]<1e-9 or (1-phi_s1[-1]-phi_p1[-1]<1e-9)) and len(phi_s1) > 10:
			# start plotting out the curve
			# ax.scatter(phi_s1, 1-phi_s1-phi_p1, phi_p1, c='gold', s=3)
			# ax.scatter(phi_s2, 1-phi_s2-phi_p2, phi_p2, c='pink', s=3)
			print("Found a complete binodal. Moving on...")
			phi_1 = np.array([phi_s1, phi_p1]).T
			phi_2 = np.array([phi_s2, phi_p2]).T
			BINODALS["groupings"][uidx]["alpha"]["binodals"] = [phi_1, phi_2]
			return 
			

		
		# move on to mu_pc
		print("Binodal run for pc along c2...")
		phi_p1, phi_p2, phi_c1, phi_c2 = self.sym_mu_pc.binodal_run_in_c2(self.crits[idx])
		
		# if (phi_p1[-1]<1e-2 or phi_c1[-1]<1e-2 or (1-phi_c1[-1]-phi_p1[-1]<1e-2)) and len(phi_s1) > 10:
		if (phi_p1[-1]<1e-9 or phi_c1[-1]<1e-9 or (1-phi_c1[-1]-phi_p1[-1]<1e-9)) and len(phi_s1) > 10:
			# start plotting 
			# ax.scatter(1-phi_p1-phi_c1, phi_c1, phi_p1, c='gold', s=3)
			# ax.scatter(1-phi_p2-phi_c2, phi_c2, phi_p2, c='lime', s=3)
			print("Found a complete binodal. Moving on...")
			phi_1 = np.array([1-phi_p1-phi_c1, phi_p1]).T
			phi_2 = np.array([1-phi_p2-phi_c2, phi_p2]).T
			BINODALS["groupings"][uidx]["alpha"]["binodals"] = [phi_1, phi_2]
			return 
			



		print("Binodal run for pc along p2...")
		phi_p1, phi_p2, phi_c1, phi_c2 = self.sym_mu_pc.binodal_run_in_p2(self.crits[idx])
		
		# if (phi_p1[-1]<1e-2 or phi_c1[-1]<1e-2 or (1-phi_c1[-1]-phi_p1[-1]<1e-2)) and len(phi_s1) > 10:
		if (phi_p1[-1]<1e-9 or phi_c1[-1]<1e-9 or (1-phi_c1[-1]-phi_p1[-1]<1e-9)) and len(phi_s1) > 10:
			# start plotting 
			phi_1 = np.array([1-phi_p1-phi_c1, phi_p1]).T
			phi_2 = np.array([1-phi_p2-phi_c2, phi_p2]).T			
			print("Found a complete binodal. Moving on...")
			BINODALS["groupings"][uidx]["alpha"]["binodals"] = [phi_1, phi_2]
			return 
			

		
		print("Binodal run for sc along c2...")
		phi_s1, phi_s2, phi_c1, phi_c2 = self.sym_mu_sc.binodal_run_in_c2(self.crits[idx])
		
		# if (phi_s1[-1]<1e-2 or phi_c1[-1]<1e-2 or (1-phi_c1[-1]-phi_s1[-1]<1e-2)) and len(phi_s1) > 10:
		if (phi_s1[-1]<1e-9 or phi_c1[-1]<1e-9 or (1-phi_c1[-1]-phi_s1[-1]<1e-9)) and len(phi_s1) > 10:
			# start plotting 
			# ax.scatter(phi_s1, phi_c1, 1-phi_s1-phi_c1, c='gold', s=3)
			# ax.scatter(phi_s2, phi_c2, 1-phi_s2-phi_c2, c='lime', s=3)
			print("Found a complete binodal. Moving on...")
			phi_1 = np.array([phi_s1, 1-phi_s1-phi_c1]).T
			phi_2 = np.array([phi_s2, 1-phi_s2-phi_c2]).T
			BINODALS["groupings"][uidx]["alpha"]["binodals"] = [phi_1, phi_2]
			return 
			

		
		print("Binodal run for sc along s2...")
		phi_s1, phi_s2, phi_c1, phi_c2 = self.sym_mu_sc.binodal_run_in_s2(self.crits[idx])
		
		# if (phi_s1[-1]<1e-2 or phi_c1[-1]<1e-2 or (1-phi_c1[-1]-phi_s1[-1]<1e-2)) and len(phi_s1) > 10:
		if (phi_s1[-1]<1e-9 or phi_c1[-1]<1e-9 or (1-phi_c1[-1]-phi_s1[-1]<1e-9)) and len(phi_s1) > 10:
			# start plotting 
			# ax.scatter(phi_s1, phi_c1, 1-phi_s1-phi_c1, c='gold', s=3)
			# ax.scatter(phi_s2, phi_c2, 1-phi_s2-phi_c2, c='lime', s=3)
			print("Found a complete binodal. Moving on...")
			phi_1 = np.array([phi_s1, 1-phi_s1-phi_c1]).T
			phi_2 = np.array([phi_s2, 1-phi_s2-phi_c2]).T
			BINODALS["groupings"][uidx]["alpha"]["binodals"] = [phi_1, phi_2]
			return 
			

		'''
		along_normal = True
		print(f"Get the binodal curve...", flush=True, end=' ')
		phi_s1, phi_s2, phi_p1, phi_p2 = self.sym_mu_ps.binodal_run_in_s2(self.crits[idx], along_normal)

		# if (phi_s1[-1]<1e-2 or phi_p1[-1]<1e-2 or (1-phi_s1[-1]-phi_p1[-1]<1e-2)) and len(phi_s1) > 10:
		if (phi_s1[-1]<1e-9 or phi_p1[-1]<1e-9 or (1-phi_s1[-1]-phi_p1[-1]<1e-9)) and len(phi_s1) > 10:
			# start plotting out the curve
			# ax.scatter(phi_s1, 1-phi_s1-phi_p1, phi_p1, c='gold', s=3)
			# ax.scatter(phi_s2, 1-phi_s2-phi_p2, phi_p2, c='lime', s=3)
			print("Found a complete binodal. Moving on...")
			phi_1 = np.array([phi_s1, phi_p1]).T
			phi_2 = np.array([phi_s2, phi_p2]).T
			BINODALS["groupings"][uidx]["alpha"]["binodals"] = [phi_1, phi_2]
			return 
			

		phi_s1, phi_s2, phi_p1, phi_p2 = self.sym_mu_ps.binodal_run_in_p2(self.crits[idx], along_normal)

		# if (phi_s1[-1]<1e-2 or phi_p1[-1]<1e-2 or (1-phi_s1[-1]-phi_p1[-1]<1e-2)) and len(phi_s1) > 10:
		if (phi_s1[-1]<1e-9 or phi_p1[-1]<1e-9 or (1-phi_s1[-1]-phi_p1[-1]<1e-9)) and len(phi_s1) > 10:
			# start plotting out the curve
			# ax.scatter(phi_s1, 1-phi_s1-phi_p1, phi_p1, c='gold', s=3)
			# ax.scatter(phi_s2, 1-phi_s2-phi_p2, phi_p2, c='lime', s=3)
			print("Found a complete binodal. Moving on...")
			phi_1 = np.array([phi_s1, phi_p1]).T
			phi_2 = np.array([phi_s2, phi_p2]).T
			BINODALS["groupings"][uidx]["alpha"]["binodals"] = [phi_1, phi_2]
			return 
			
		
		print(f"Get the binodal curve...", flush=True, end=' ')
		phi_s1, phi_s2, phi_c1, phi_c2 = self.sym_mu_sc.binodal_run_in_s2(self.crits[idx], along_normal)

		# if (phi_s1[-1]<1e-2 or phi_p1[-1]<1e-2 or (1-phi_s1[-1]-phi_p1[-1]<1e-2)) and len(phi_s1) > 10:
		if (phi_s1[-1]<1e-9 or phi_p1[-1]<1e-9 or (1-phi_s1[-1]-phi_p1[-1]<1e-9)) and len(phi_s1) > 10:
			# start plotting out the curve
			# ax.scatter(phi_s1, phi_c1, 1-phi_s1-phi_c1, c='gold', s=3)
			# ax.scatter(phi_s2, phi_c2, 1-phi_s2-phi_c2, c='lime', s=3)
			print("Found a complete binodal. Moving on...")
			phi_1 = np.array([phi_s1, 1-phi_p1-phi_s1]).T
			phi_2 = np.array([phi_s2, 1-phi_p2-phi_s2]).T
			BINODALS["groupings"][uidx]["alpha"]["binodals"] = [phi_1, phi_2]
			return 
			
		phi_s1, phi_s2, phi_c1, phi_c2 = self.sym_mu_sc.binodal_run_in_c2(self.crits[idx], along_normal)

		# if (phi_s1[-1]<1e-2 or phi_p1[-1]<1e-2 or (1-phi_s1[-1]-phi_p1[-1]<1e-2) or (min_dist < 0.01)) and len(phi_s1) > 10:
		if (phi_s1[-1]<1e-9 or phi_p1[-1]<1e-9 or (1-phi_s1[-1]-phi_p1[-1]<1e-9) or (min_dist < 0.01)) and len(phi_s1) > 10:
			# start plotting out the curve
			# ax.scatter(phi_s1, phi_c1, 1-phi_s1-phi_c1, c='gold', s=3)
			# ax.scatter(phi_s2, phi_c2, 1-phi_s2-phi_c2, c='lime', s=3)
			print("Found a complete binodal. Moving on...")
			phi_1 = np.array([phi_s1, 1-phi_s1-phi_c1]).T
			phi_2 = np.array([phi_s2, 1-phi_s2-phi_c2]).T
			BINODALS["groupings"][uidx]["alpha"]["binodals"] = [phi_1, phi_2]
			return 
			
	
		phi_p1, phi_p2, phi_c1, phi_c2 = self.sym_mu_pc.binodal_run_in_p2(self.crits[idx], along_normal)

		if (phi_s1[-1]<1e-2 or phi_p1[-1]<1e-2 or (1-phi_s1[-1]-phi_p1[-1]<1e-2)) and len(phi_s1) > 10:
			# start plotting out the curve
			# ax.scatter(1-phi_p1-phi_c1, phi_c1, phi_p1, c='gold', s=3)
			# ax.scatter(1-phi_p2-phi_c2, phi_c2, phi_p2, c='lime', s=3)
			print("Found a complete binodal. Moving on...")
			phi_1 = np.array([1-phi_p1-phi_c1, phi_p1]).T
			phi_2 = np.array([1-phi_p2-phi_c2, phi_p2]).T
			BINODALS["groupings"][uidx]["alpha"]["binodals"] = [phi_1, phi_2]
			return 
			

		phi_p1, phi_p2, phi_c1, phi_c2 = self.sym_mu_pc.binodal_run_in_c2(self.crits[idx], along_normal)

		if (phi_s1[-1]<1e-2 or phi_p1[-1]<1e-2 or (1-phi_s1[-1]-phi_p1[-1]<1e-2) or (min_dist < 0.01)) and len(phi_s1) > 10:
			# start plotting out the curve
			# ax.scatter(1-phi_c1-phi_p1, phi_c1, phi_p1, c='gold', s=3)
			# ax.scatter(1-phi_c2-phi_p2, phi_c2, phi_p2, c='lime', s=3)
			print("Found a complete binodal. Moving on...")
			phi_1 = np.array([1-phi_p1-phi_c1, phi_p1]).T
			phi_2 = np.array([1-phi_p2-phi_c2, phi_p2]).T
			BINODALS["groupings"][uidx]["alpha"]["binodals"] = [phi_1, phi_2]
			return 
		'''
		
		BINODALS["groupings"][uidx]["alpha"]["binodals"] = [np.empty((0,3)), np.empty((0,3))]

		return
		
	def tangent_tracing_dyad(self, BINODALS, uidx):
		indices = BINODALS["groupings"][uidx]["raw_list"]
		axis    = (self.crits[indices[0]] - self.crits[indices[1]])/np.linalg.norm(self.crits[indices[0]] - self.crits[indices[1]])
		c_crit  = self.crits[indices[1]]
		BINODALS["groupings"][uidx]["alpha"]["binodals"] = [np.empty((0,3)), np.empty((0,3))]
		for count,idx in enumerate(indices):
			
			print(f"Binodal run for ps along s2.", flush=True)
			phi_s1, phi_s2, phi_p1, phi_p2 = self.sym_mu_ps.binodal_run_in_s2(self.crits[idx])
			phi_1 = np.array([phi_s1, phi_p1, 1-phi_s1-phi_p1]).T
			phi_2 = np.array([phi_s2, phi_p2, 1-phi_s2-phi_p2]).T

			adj_phi_1 = (phi_1[:,0:2]-c_crit[0:2])/np.linalg.norm(phi_1[:,0:2]-c_crit[0:2]).reshape(-1,1)
			adj_phi_2 = (phi_2[:,0:2]-c_crit[0:2])/np.linalg.norm(phi_2[:,0:2]-c_crit[0:2]).reshape(-1,1)

			mask1 = (np.sign(np.cross(axis[0:2], adj_phi_1[:,0:2])) == 1) & (np.sign(np.cross(axis[0:2], adj_phi_2[:,0:2])) == -1)
			mask2 = (np.sign(np.cross(axis[0:2], adj_phi_2[:,0:2])) == 1) & (np.sign(np.cross(axis[0:2], adj_phi_1[:,0:2])) == -1)
			
			pos_phi = np.vstack((phi_1[mask1], phi_2[mask2]))
			neg_phi = np.vstack((phi_2[mask1], phi_1[mask2]))

			BINODALS["groupings"][uidx]["alpha"]["binodals"][0] = np.vstack((BINODALS["groupings"][uidx]["alpha"]["binodals"][0], neg_phi))
			BINODALS["groupings"][uidx]["alpha"]["binodals"][1] = np.vstack((BINODALS["groupings"][uidx]["alpha"]["binodals"][1], pos_phi))

			print(f"Binodal run for ps along p2.", flush=True)
			phi_s1, phi_s2, phi_p1, phi_p2 = self.sym_mu_ps.binodal_run_in_p2(self.crits[idx])
			phi_1 = np.array([phi_s1, phi_p1, 1-phi_s1-phi_p1]).T
			phi_2 = np.array([phi_s2, phi_p2, 1-phi_s2-phi_p2]).T

			adj_phi_1 = (phi_1[:,0:2]-c_crit[0:2])/np.linalg.norm(phi_1[:,0:2]-c_crit[0:2]).reshape(-1,1)
			adj_phi_2 = (phi_2[:,0:2]-c_crit[0:2])/np.linalg.norm(phi_2[:,0:2]-c_crit[0:2]).reshape(-1,1)

			mask1 = (np.sign(np.cross(axis[0:2], adj_phi_1[:,0:2])) == 1) & (np.sign(np.cross(axis[0:2], adj_phi_2[:,0:2])) == -1)
			mask2 = (np.sign(np.cross(axis[0:2], adj_phi_2[:,0:2])) == 1) & (np.sign(np.cross(axis[0:2], adj_phi_1[:,0:2])) == -1)
			
			pos_phi = np.vstack((phi_1[mask1], phi_2[mask2]))
			neg_phi = np.vstack((phi_2[mask1], phi_1[mask2]))

			BINODALS["groupings"][uidx]["alpha"]["binodals"][0] = np.vstack((BINODALS["groupings"][uidx]["alpha"]["binodals"][0], neg_phi))
			BINODALS["groupings"][uidx]["alpha"]["binodals"][1] = np.vstack((BINODALS["groupings"][uidx]["alpha"]["binodals"][1], pos_phi))
			
			print(f"Binodal run for sc along s2.", flush=True)
			phi_s1, phi_s2, phi_c1, phi_c2 = self.sym_mu_sc.binodal_run_in_s2(self.crits[idx])
			phi_1 = np.array([phi_s1, 1-phi_s1-phi_c1, phi_c1]).T
			phi_2 = np.array([phi_s2, 1-phi_s2-phi_c2, phi_c2]).T

			adj_phi_1 = (phi_1[:,0:2]-c_crit[0:2])/np.linalg.norm(phi_1[:,0:2]-c_crit[0:2]).reshape(-1,1)
			adj_phi_2 = (phi_2[:,0:2]-c_crit[0:2])/np.linalg.norm(phi_2[:,0:2]-c_crit[0:2]).reshape(-1,1)

			mask1 = (np.sign(np.cross(axis[0:2], adj_phi_1[:,0:2])) == 1) & (np.sign(np.cross(axis[0:2], adj_phi_2[:,0:2])) == -1)
			mask2 = (np.sign(np.cross(axis[0:2], adj_phi_2[:,0:2])) == 1) & (np.sign(np.cross(axis[0:2], adj_phi_1[:,0:2])) == -1)
			
			pos_phi = np.vstack((phi_1[mask1], phi_2[mask2]))
			neg_phi = np.vstack((phi_2[mask1], phi_1[mask2]))

			BINODALS["groupings"][uidx]["alpha"]["binodals"][0] = np.vstack((BINODALS["groupings"][uidx]["alpha"]["binodals"][0], neg_phi))
			BINODALS["groupings"][uidx]["alpha"]["binodals"][1] = np.vstack((BINODALS["groupings"][uidx]["alpha"]["binodals"][1], pos_phi))
			
			print(f"Binodal run for sc along c2.", flush=True)
			phi_s1, phi_s2, phi_c1, phi_c2 = self.sym_mu_sc.binodal_run_in_c2(self.crits[idx])
			phi_1 = np.array([phi_s1, 1-phi_s1-phi_c1, phi_c1]).T
			phi_2 = np.array([phi_s2, 1-phi_s2-phi_c2, phi_c2]).T

			adj_phi_1 = (phi_1[:,0:2]-c_crit[0:2])/np.linalg.norm(phi_1[:,0:2]-c_crit[0:2]).reshape(-1,1)
			adj_phi_2 = (phi_2[:,0:2]-c_crit[0:2])/np.linalg.norm(phi_2[:,0:2]-c_crit[0:2]).reshape(-1,1)

			mask1 = (np.sign(np.cross(axis[0:2], adj_phi_1[:,0:2])) == 1) & (np.sign(np.cross(axis[0:2], adj_phi_2[:,0:2])) == -1)
			mask2 = (np.sign(np.cross(axis[0:2], adj_phi_2[:,0:2])) == 1) & (np.sign(np.cross(axis[0:2], adj_phi_1[:,0:2])) == -1)
			
			pos_phi = np.vstack((phi_1[mask1], phi_2[mask2]))
			neg_phi = np.vstack((phi_2[mask1], phi_1[mask2]))

			BINODALS["groupings"][uidx]["alpha"]["binodals"][0] = np.vstack((BINODALS["groupings"][uidx]["alpha"]["binodals"][0], neg_phi))
			BINODALS["groupings"][uidx]["alpha"]["binodals"][1] = np.vstack((BINODALS["groupings"][uidx]["alpha"]["binodals"][1], pos_phi))
			
			print(f"Binodal run for pc along c2.", flush=True)
			phi_p1, phi_p2, phi_c1, phi_c2 = self.sym_mu_pc.binodal_run_in_c2(self.crits[idx])
			phi_1 = np.array([1-phi_p1-phi_c1, phi_p1, phi_c1]).T
			phi_2 = np.array([1-phi_p2-phi_c2, phi_p2, phi_c2]).T

			adj_phi_1 = (phi_1[:,0:2]-c_crit[0:2])/np.linalg.norm(phi_1[:,0:2]-c_crit[0:2]).reshape(-1,1)
			adj_phi_2 = (phi_2[:,0:2]-c_crit[0:2])/np.linalg.norm(phi_2[:,0:2]-c_crit[0:2]).reshape(-1,1)

			mask1 = (np.sign(np.cross(axis[0:2], adj_phi_1[:,0:2])) == 1) & (np.sign(np.cross(axis[0:2], adj_phi_2[:,0:2])) == -1)
			mask2 = (np.sign(np.cross(axis[0:2], adj_phi_2[:,0:2])) == 1) & (np.sign(np.cross(axis[0:2], adj_phi_1[:,0:2])) == -1)
			
			pos_phi = np.vstack((phi_1[mask1], phi_2[mask2]))
			neg_phi = np.vstack((phi_2[mask1], phi_1[mask2]))

			BINODALS["groupings"][uidx]["alpha"]["binodals"][0] = np.vstack((BINODALS["groupings"][uidx]["alpha"]["binodals"][0], neg_phi))
			BINODALS["groupings"][uidx]["alpha"]["binodals"][1] = np.vstack((BINODALS["groupings"][uidx]["alpha"]["binodals"][1], pos_phi))

			print(f"Binodal run for pc along p2.", flush=True)
			phi_p1, phi_p2, phi_c1, phi_c2 = self.sym_mu_pc.binodal_run_in_p2(self.crits[idx])
			phi_1 = np.array([1-phi_p1-phi_c1, phi_p1, phi_c1]).T
			phi_2 = np.array([1-phi_p2-phi_c2, phi_p2, phi_c2]).T

			adj_phi_1 = (phi_1[:,0:2]-c_crit[0:2])/np.linalg.norm(phi_1[:,0:2]-c_crit[0:2]).reshape(-1,1)
			adj_phi_2 = (phi_2[:,0:2]-c_crit[0:2])/np.linalg.norm(phi_2[:,0:2]-c_crit[0:2]).reshape(-1,1)

			mask1 = (np.sign(np.cross(axis[0:2], adj_phi_1[:,0:2])) == 1) & (np.sign(np.cross(axis[0:2], adj_phi_2[:,0:2])) == -1)
			mask2 = (np.sign(np.cross(axis[0:2], adj_phi_2[:,0:2])) == 1) & (np.sign(np.cross(axis[0:2], adj_phi_1[:,0:2])) == -1)
			
			pos_phi = np.vstack((phi_1[mask1], phi_2[mask2]))
			neg_phi = np.vstack((phi_2[mask1], phi_1[mask2]))

			BINODALS["groupings"][uidx]["alpha"]["binodals"][0] = np.vstack((BINODALS["groupings"][uidx]["alpha"]["binodals"][0], neg_phi))
			BINODALS["groupings"][uidx]["alpha"]["binodals"][1] = np.vstack((BINODALS["groupings"][uidx]["alpha"]["binodals"][1], pos_phi))
		
		return

