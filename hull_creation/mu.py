import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import root as ROOT
import sympy as sym
import tangent
import ternary

# this defines how many iterations your binodal solver is going to perform
max_it = 1e+5

class sym_mu_ps:
	def __init__(self, inputs, spinodal):
		self.chi_sc   = inputs["chi_sc"]
		self.chi_ps   = inputs["chi_ps"]
		self.chi_pc   = inputs["chi_pc"]
		self.vs       = inputs["vs"]
		self.vc       = inputs["vc"]
		self.vp       = inputs["vp"]
		self.spinodal = spinodal
		self.setup()
		return

	def setup(self):

		phi_s1, phi_p1, phi_s2, phi_p2 = sym.symbols('phi_s1 phi_p1 phi_s2 phi_p2')

		mu_s1 = sym.log(phi_s1)          + 1 - phi_s1            - self.vs/self.vp * phi_p1 - self.vs/self.vc * (1-phi_s1-phi_p1) + self.vs * (phi_p1**2 * self.chi_ps + (1-phi_s1-phi_p1)**2 * self.chi_sc + phi_p1 * (1-phi_s1-phi_p1) * (self.chi_ps + self.chi_sc - self.chi_pc))
		mu_p1 = sym.log(phi_p1)          + 1 - phi_p1            - self.vp/self.vs * phi_s1 - self.vp/self.vc * (1-phi_s1-phi_p1) + self.vp * (phi_s1**2 * self.chi_ps + (1-phi_s1-phi_p1)**2 * self.chi_pc + phi_s1 * (1-phi_s1-phi_p1) * (self.chi_ps + self.chi_pc - self.chi_sc))
		mu_c1 = sym.log(1-phi_s1-phi_p1) + 1 - (1-phi_s1-phi_p1) - self.vc/self.vs * phi_s1 - self.vc/self.vp * phi_p1            + self.vc * (phi_s1**2 * self.chi_sc + phi_p1**2            * self.chi_pc + phi_s1 * phi_p1            * (self.chi_sc + self.chi_pc - self.chi_ps))

		mu_s2 = sym.log(phi_s2)          + 1 - phi_s2            - self.vs/self.vp * phi_p2 - self.vs/self.vc * (1-phi_s2-phi_p2) + self.vs * (phi_p2**2 * self.chi_ps + (1-phi_s2-phi_p2)**2 * self.chi_sc + phi_p2 * (1-phi_s2-phi_p2) * (self.chi_ps + self.chi_sc - self.chi_pc))
		mu_p2 = sym.log(phi_p2)          + 1 - phi_p2            - self.vp/self.vs * phi_s2 - self.vp/self.vc * (1-phi_s2-phi_p2) + self.vp * (phi_s2**2 * self.chi_ps + (1-phi_s2-phi_p2)**2 * self.chi_pc + phi_s2 * (1-phi_s2-phi_p2) * (self.chi_ps + self.chi_pc - self.chi_sc))
		mu_c2 = sym.log(1-phi_s2-phi_p2) + 1 - (1-phi_s2-phi_p2) - self.vc/self.vs * phi_s2 - self.vc/self.vp * phi_p2            + self.vc * (phi_s2**2 * self.chi_sc + phi_p2**2            * self.chi_pc + phi_s2 * phi_p2            * (self.chi_sc + self.chi_pc - self.chi_ps))

		# lambdifying the mu calculators 
		L_mu_s = sym.lambdify([phi_s1,phi_p1], mu_s1)
		L_mu_p = sym.lambdify([phi_s1,phi_p1], mu_p1)
		L_mu_c = sym.lambdify([phi_s1,phi_p1], mu_c1)

		self.mu_s = lambda phi_s, phi_p: L_mu_s(phi_s, phi_p)
		self.mu_p = lambda phi_s, phi_p: L_mu_p(phi_s, phi_p)
		self.mu_c = lambda phi_s, phi_p: L_mu_c(phi_s, phi_p)

		delta_mu_s = mu_s1 - mu_s2
		delta_mu_p = mu_p1 - mu_p2
		delta_mu_c = mu_c1 - mu_c2

		# lambdify the function 
		L_dmu_s = sym.lambdify([phi_s1, phi_p1, phi_s2, phi_p2], delta_mu_s)
		L_dmu_p = sym.lambdify([phi_s1, phi_p1, phi_s2, phi_p2], delta_mu_p)
		L_dmu_c = sym.lambdify([phi_s1, phi_p1, phi_s2, phi_p2], delta_mu_c)

		# do the differentiation of mu_s 
		d_delta_mu_s_dps1 = sym.diff(delta_mu_s, phi_s1)
		d_delta_mu_s_dps2 = sym.diff(delta_mu_s, phi_s2)
		d_delta_mu_s_dpp1 = sym.diff(delta_mu_s, phi_p1)
		d_delta_mu_s_dpp2 = sym.diff(delta_mu_s, phi_p2)

		# lambdify the function
		L_d_delta_mu_s_dps1 = sym.lambdify([phi_s1, phi_p1, phi_s2, phi_p2], d_delta_mu_s_dps1)
		L_d_delta_mu_s_dpp1 = sym.lambdify([phi_s1, phi_p1, phi_s2, phi_p2], d_delta_mu_s_dpp1)
		L_d_delta_mu_s_dps2 = sym.lambdify([phi_s1, phi_p1, phi_s2, phi_p2], d_delta_mu_s_dps2)
		L_d_delta_mu_s_dpp2 = sym.lambdify([phi_s1, phi_p1, phi_s2, phi_p2], d_delta_mu_s_dpp2)

		# do the differentiation of mu_p
		d_delta_mu_p_dps1 = sym.diff(delta_mu_p, phi_s1)
		d_delta_mu_p_dps2 = sym.diff(delta_mu_p, phi_s2)
		d_delta_mu_p_dpp1 = sym.diff(delta_mu_p, phi_p1)
		d_delta_mu_p_dpp2 = sym.diff(delta_mu_p, phi_p2)

		# lambdify the function 
		L_d_delta_mu_p_dps1 = sym.lambdify([phi_s1, phi_p1, phi_s2, phi_p2], d_delta_mu_p_dps1)
		L_d_delta_mu_p_dpp1 = sym.lambdify([phi_s1, phi_p1, phi_s2, phi_p2], d_delta_mu_p_dpp1)
		L_d_delta_mu_p_dps2 = sym.lambdify([phi_s1, phi_p1, phi_s2, phi_p2], d_delta_mu_p_dps2)
		L_d_delta_mu_p_dpp2 = sym.lambdify([phi_s1, phi_p1, phi_s2, phi_p2], d_delta_mu_p_dpp2)

		# do the differentiation of mu_c
		d_delta_mu_c_dps1   = sym.diff(delta_mu_c, phi_s1)
		d_delta_mu_c_dps2   = sym.diff(delta_mu_c, phi_s2)
		d_delta_mu_c_dpp1   = sym.diff(delta_mu_c, phi_p1)
		d_delta_mu_c_dpp2   = sym.diff(delta_mu_c, phi_p2)

		# lambdify the function 
		L_d_delta_mu_c_dps1 = sym.lambdify([phi_s1, phi_p1, phi_s2, phi_p2], d_delta_mu_c_dps1)
		L_d_delta_mu_c_dpp1 = sym.lambdify([phi_s1, phi_p1, phi_s2, phi_p2], d_delta_mu_c_dpp1)
		L_d_delta_mu_c_dps2 = sym.lambdify([phi_s1, phi_p1, phi_s2, phi_p2], d_delta_mu_c_dps2)
		L_d_delta_mu_c_dpp2 = sym.lambdify([phi_s1, phi_p1, phi_s2, phi_p2], d_delta_mu_c_dpp2)

		self.delta_mu_s = lambda s1, p1, s2, p2: L_dmu_s(s1, p1, s2, p2)
		self.delta_mu_p = lambda s1, p1, s2, p2: L_dmu_p(s1, p1, s2, p2)
		self.delta_mu_c = lambda s1, p1, s2, p2: L_dmu_c(s1, p1, s2, p2)

		# get the new function
		self.d_delta_mu_s_dps1 = lambda s1, p1, s2, p2: L_d_delta_mu_s_dps1(s1, p1, s2, p2)
		self.d_delta_mu_s_dpp1 = lambda s1, p1, s2, p2: L_d_delta_mu_s_dpp1(s1, p1, s2, p2)
		self.d_delta_mu_s_dps2 = lambda s1, p1, s2, p2: L_d_delta_mu_s_dps2(s1, p1, s2, p2)
		self.d_delta_mu_s_dpp2 = lambda s1, p1, s2, p2: L_d_delta_mu_s_dpp2(s1, p1, s2, p2)

		# get the new function
		self.d_delta_mu_p_dps1 = lambda s1, p1, s2, p2: L_d_delta_mu_p_dps1(s1, p1, s2, p2)
		self.d_delta_mu_p_dpp1 = lambda s1, p1, s2, p2: L_d_delta_mu_p_dpp1(s1, p1, s2, p2)
		self.d_delta_mu_p_dps2 = lambda s1, p1, s2, p2: L_d_delta_mu_p_dps2(s1, p1, s2, p2)
		self.d_delta_mu_p_dpp2 = lambda s1, p1, s2, p2: L_d_delta_mu_p_dpp2(s1, p1, s2, p2)

		# get the new function
		self.d_delta_mu_c_dps1 = lambda s1, p1, s2, p2: L_d_delta_mu_c_dps1(s1, p1, s2, p2)
		self.d_delta_mu_c_dpp1 = lambda s1, p1, s2, p2: L_d_delta_mu_c_dpp1(s1, p1, s2, p2)
		self.d_delta_mu_c_dps2 = lambda s1, p1, s2, p2: L_d_delta_mu_c_dps2(s1, p1, s2, p2)
		self.d_delta_mu_c_dpp2 = lambda s1, p1, s2, p2: L_d_delta_mu_c_dpp2(s1, p1, s2, p2)

		return 

	def find_solution_in_nbrhd_pert_in_p2(self, crit_point, along_normal=False, loglims=[-3,-10,100]):

		# print(f"crit point of interest = {crit_point}", flush=True)
		delta_pp2 = []
		phi_s1    = [crit_point[0]]
		phi_s2    = [crit_point[0]]
		phi_p1    = [crit_point[1]]
		phi_p2    = [crit_point[1]]

		good_root = False

		slope          = tangent.tangent2(self.vs, self.vc, self.vp, crit_point[0], crit_point[1], self.chi_pc, self.chi_ps, self.chi_sc, self.spinodal.root_up_s, self.spinodal.root_lo_s)

		if np.isnan(slope) or np.isinf(slope):
			if along_normal:
				tangent_vector = np.array([1,0], dtype=np.float64)
			else:
				tangent_vector = np.array([0,1], dtype=np.float64)
		else:
			if along_normal:
				if np.abs(slope) < 1e-12:
					tangent_vector = np.array([0, 1])
				else:
					slope          = -1/slope
					tangent_vector = np.array([1, slope])/np.sqrt(1+slope**2)
			else:
				tangent_vector = np.array([1, slope])/np.sqrt(1+slope**2)

		print(f"Tangent vector = {tangent_vector}", flush=True)

		pert_scale = np.logspace(loglims[0], loglims[1], loglims[2])

		for pert in pert_scale:
			def mu_init(phi_):
				eq1 = self.delta_mu_s(phi_[0], phi_[1], phi_[2], crit_point[1] + pert * tangent_vector[1])
				eq2 = self.delta_mu_p(phi_[0], phi_[1], phi_[2], crit_point[1] + pert * tangent_vector[1])
				eq3 = self.delta_mu_c(phi_[0], phi_[1], phi_[2], crit_point[1] + pert * tangent_vector[1])
				return [eq1, eq2, eq3]

			root = fsolve(mu_init, [crit_point[0] - pert*tangent_vector[0], crit_point[1] - pert*tangent_vector[1], crit_point[0] + pert*tangent_vector[0]], xtol=1e-16)
			p1   = np.array([root[0], root[1]])
			p2   = np.array([root[2], crit_point[1] + pert * tangent_vector[1]])

			d1   = p1 - np.array([crit_point[0], crit_point[1]])
			d2   = p2 - np.array([crit_point[0], crit_point[1]])
			d1   = d1/np.linalg.norm(d1)
			d2   = d1/np.linalg.norm(d2)

			if np.linalg.norm(d1-d2)<1e-3:
				# print("Directions are too close.")
				continue

			if root[0] > 1 or root[0] < 0 or root[1] > 1 or root[1] < 0 or root[2] > 1 or root[2] < 0:
				# print("Breaking out...")
				continue
			
			if (np.abs(mu_init(root))>1e-14).any():
				# print(f"Bad root: phi1 = ({root[0], root[1]}), phi2 = ({root[2], crit_point[1] + pert * tangent_vector[1]})...", flush=True)
				continue
			
			elif np.linalg.norm(p1-p2) < 1e-3: 
				# print(f"Too close...")
				continue

			else:
				# print(f"pert = {pert}")
				# print(f"Found root: phi1 = ({root[0], root[1]}), phi2 = ({root[2], crit_point[1] + pert * tangent_vector[1]})...", flush=True)
				if abs(pert*tangent_vector[1] < 1e-20):
					continue
				good_root = True
				delta_pp2.append(pert*tangent_vector[1])
				phi_s1.append(root[0])
				phi_p1.append(root[1])
				phi_s2.append(root[2])
				phi_p2.append(crit_point[1]+pert*tangent_vector[1])

			# breaking out of pert loop
			if good_root:
				break

		# print(f"Broken out of pert loop!", flush=True)

		if not good_root:
			pass
			# print(f"No neighboring solution found...", flush=True)

		# print("I have found the initial root outside of the critical point. Time to go beyond.", flush=True)
		# print(f"phi_s1 = {phi_s1}", flush=True)
		# print(f"phi_p1 = {phi_p1}", flush=True)
		# print(f"phi_s2 = {phi_s2}", flush=True)
		# print(f"phi_p2 = {phi_p2}", flush=True)

		return phi_s1, phi_s2, phi_p1, phi_p2, delta_pp2

	def calc_perts_in_p2(self, phi1, phi2, delta_phip2):

		a1 = np.float64(self.d_delta_mu_s_dps1(phi1[0], phi1[1], phi2[0], phi2[1]))
		a2 = np.float64(self.d_delta_mu_s_dpp1(phi1[0], phi1[1], phi2[0], phi2[1]))
		a3 = np.float64(self.d_delta_mu_s_dps2(phi1[0], phi1[1], phi2[0], phi2[1]))
		a4 = np.float64(-delta_phip2 * self.d_delta_mu_s_dpp2(phi1[0], phi1[1], phi2[0], phi2[1]))

		b1 = np.float64(self.d_delta_mu_p_dps1(phi1[0], phi1[1], phi2[0], phi2[1]))
		b2 = np.float64(self.d_delta_mu_p_dpp1(phi1[0], phi1[1], phi2[0], phi2[1]))
		b3 = np.float64(self.d_delta_mu_p_dps2(phi1[0], phi1[1], phi2[0], phi2[1]))
		b4 = np.float64(-delta_phip2 * self.d_delta_mu_p_dpp2(phi1[0], phi1[1], phi2[0], phi2[1]))

		c1 = np.float64(self.d_delta_mu_c_dps1(phi1[0], phi1[1], phi2[0], phi2[1]))
		c2 = np.float64(self.d_delta_mu_c_dpp1(phi1[0], phi1[1], phi2[0], phi2[1]))
		c3 = np.float64(self.d_delta_mu_c_dps2(phi1[0], phi1[1], phi2[0], phi2[1]))
		c4 = np.float64(-delta_phip2 * self.d_delta_mu_c_dpp2(phi1[0], phi1[1], phi2[0], phi2[1]))

		D  = np.float64(np.linalg.det(np.array([[a1,a2,a3],[b1,b2,b3],[c1,c2,c3]], dtype=np.float64)))
		Dx = np.float64(np.linalg.det(np.array([[a4,a2,a3],[b4,b2,b3],[c4,c2,c3]], dtype=np.float64)))
		Dy = np.float64(np.linalg.det(np.array([[a1,a4,a3],[b1,b4,b3],[c1,c4,c3]], dtype=np.float64)))
		Dz = np.float64(np.linalg.det(np.array([[a1,a2,a4],[b1,b2,b4],[c1,c2,c4]], dtype=np.float64)))

		# print(f"a1 = {a1}, a2 = {a2}, a3 = {a3}, a4 = {a4}")
		# print(f"b1 = {b1}, b2 = {b2}, b3 = {b3}, b4 = {b4}")
		# print(f"c1 = {c1}, c2 = {c2}, b3 = {c3}, b4 = {c4}")
		# print(f"D = {D}, Dx = {Dx}, Dy = {Dy}, Dz = {Dz}")
		# print(f"dx = {Dx/D}, dy = {Dy/D}, dz = {Dz/D}")

		return Dx/D, Dy/D, Dz/D

	def binodal_run_in_p2(self, crit_point, along_normal=False):
		phi_s1, phi_s2, phi_p1, phi_p2, delta_pp2 = self.find_solution_in_nbrhd_pert_in_p2(crit_point, along_normal)

		if len(phi_s1) == 1:
			return np.array(phi_s1), np.array(phi_s2), np.array(phi_p1), np.array(phi_p2)

		print(f"delta_pp2 = {delta_pp2}", flush=True)
		condition = (phi_s1[-1] < 1e-12 or phi_p1[-1] < 1e-12 or 1-phi_s1[-1]-phi_p1[-1] < 1e-12) and (phi_s2[-1] < 1e-12 or phi_p2[-1] < 1e-12 or 1-phi_s2[-1]-phi_p2[-1] < 1e-12)

		# for i in range(ncycles):
		iterr  = 0 
		max_it = 1e+5

		while not condition:
			iterr += 1
			if iterr > max_it:
				print("Completed iterations. Breaking out...", flush=True)
				break
			# preprocess...
			# preprocessor = np.array(delta_pp2[-100:])
			# if (np.abs(preprocessor) < 1e-16).all():
			# 	pass
			# delta_pp2.append(1e-6)

			# print(f"@ i = {iterr}/{max_it}...", flush=True)

			phi1 = [phi_s1[-1], phi_p1[-1]]
			phi2 = [phi_s2[-1], phi_p2[-1]]
			delta_ps1, delta_pp1, delta_ps2 = self.calc_perts_in_p2(phi1, phi2, delta_pp2[-1])
			# print(f"delta_ps1 = {delta_ps1}, delta_pp1 = {delta_pp1}, delta_ps2 = {delta_ps2}, delta_pp2 = {delta_pp2[-1]}", flush=True)

			delta_1 = np.array([delta_ps1, delta_pp1])
			delta_2 = np.array([delta_ps2, delta_pp2[-1]])

			delta_1 = delta_1/np.linalg.norm(delta_1)
			delta_2 = delta_2/np.linalg.norm(delta_2)

			if np.linalg.norm(delta_1 - delta_2) < 1e-3:
				# print("Moving in very similar directions.", flush=True)
				continue

			def dmu(phi_):
				eq1 = self.delta_mu_s(phi_[0], phi_[1], phi_[2], phi_p2[-1]+delta_pp2[-1])
				eq2 = self.delta_mu_p(phi_[0], phi_[1], phi_[2], phi_p2[-1]+delta_pp2[-1])
				eq3 = self.delta_mu_c(phi_[0], phi_[1], phi_[2], phi_p2[-1]+delta_pp2[-1])
				return [eq1, eq2, eq3]

			# print(f"Guess provided: phi1 = ({phi_s1[-1]+delta_ps1, phi_p1[-1]+delta_pp1}), phi2 = {phi_s2[-1]+delta_ps2, phi_p2[-1]+delta_pp2[-1]}", flush=True)
			root = fsolve(dmu, [phi_s1[-1]+delta_ps1, phi_p1[-1]+delta_pp1, phi_s2[-1]+delta_ps2], xtol=1e-30)

			if root[0] > 1 or root[0] < 0 or root[1] > 1 or root[1] < 0 or root[2] > 1 or root[2] < 0 or root[0]+root[1]>1 or (root[2]+phi_p2[-1]+delta_pp2[-1]) > 1:
				delta_pp2.append(delta_pp2[-1]/1.5)
				continue

			p1 = np.array([root[0], root[1]])
			p2 = np.array([root[2], phi_p2[-1]+delta_pp2[-1]])

			p1_ = np.array([phi_s1[-1], phi_p1[-1]])
			p2_ = np.array([phi_s2[-1], phi_p2[-1]])

			d1 = p1 - p1_ 
			d2 = p2 - p2_ 

			p1__ = np.array([phi_s1[-2], phi_p1[-2]])
			p2__ = np.array([phi_s2[-2], phi_p2[-2]])

			d1_ = p1_ - p1__
			d2_ = p2_ - p2__

			# if np.linalg.norm(root[1] - phi_p2[-1]+delta_pp2[-1]) > 1e-6:
			# 	print(f"p1 = {p1}, p2 = {p2}")
			# 	print(f"p1_ = {p1_}, p2_ = {p2_}")
			# 	print(f"p1__ = {p1__}, p2__ = {p2__}")
			# 	print ("PROBLEM!")

			if (np.abs(dmu(root))>1e-14).any():
				# print(f"Bad root: phi1 = ({root[0], root[1]}), phi2 = ({root[2], phi_p2[-1]+delta_pp2[-1]})...", flush=True)
				delta_pp2.append(delta_pp2[-1]/1.1)
				continue
			elif np.linalg.norm(p1-p2) < 1e-6: 
				# print(f"Too close: phi1 = ({root[0], root[1]}), phi2 = ({root[2], phi_p2[-1]+delta_pp2[-1]})...", flush=True)
				delta_pp2.append(delta_pp2[-1]/1.1)
				continue
			# elif np.dot(d1/np.linalg.norm(d1), d1_/np.linalg.norm(d1_)) <= 0 or np.dot(d2/np.linalg.norm(d2), d2_/np.linalg.norm(d2_)) <= 0:
				# print(f"Making an about turn: phi1 = ({root[0], root[1]}), phi2 = ({root[2], phi_p2[-1]+delta_pp2[-1]})...", flush=True)
				# delta_pp2.append(delta_pp2[-1]/2)
				# continue


			# print(f"Found root: phi1 = ({root[0], root[1]}), phi2 = ({root[2], phi_p2[-1]+delta_pp2[-1]})...", flush=True)

			phi_s1.append(root[0])
			phi_p1.append(root[1])
			phi_s2.append(root[2])
			phi_p2.append(phi_p2[-1]+delta_pp2[-1])
			delta_pp2.append(delta_pp2[-1])
			condition = (phi_s1[-1] < 1e-12 or phi_p1[-1] < 1e-12 or 1-phi_s1[-1]-phi_p1[-1] < 1e-12 or delta_pp2[-1] < 1e-20)

		phi_s1 = np.array(phi_s1)
		phi_p1 = np.array(phi_p1)
		phi_s2 = np.array(phi_s2)
		phi_p2 = np.array(phi_p2)

		return phi_s1, phi_s2, phi_p1, phi_p2

	def find_solution_in_nbrhd_pert_in_s2(self, crit_point, along_normal=False, loglims=[-3,-10,100]):

		# print(f"crit point of interest = {crit_point}", flush=True)
		delta_ps2 = []
		phi_s1    = [crit_point[0]]
		phi_s2    = [crit_point[0]]
		phi_p1    = [crit_point[1]]
		phi_p2    = [crit_point[1]]

		good_root = False

		slope          = tangent.tangent2(self.vs, self.vc, self.vp, crit_point[0], crit_point[1], self.chi_pc, self.chi_ps, self.chi_sc, self.spinodal.root_up_s, self.spinodal.root_lo_s)
		if np.isnan(slope) or np.isinf(slope):
			if along_normal:
				tangent_vector = np.array([1,0], dtype=np.float64)
			else:
				tangent_vector = np.array([0,1], dtype=np.float64)

		else:
			if along_normal:
				if np.abs(slope) < 1e-12:
					tangent_vector = np.array([0, 1])
				else:
					tslope          = -1/slope
					tangent_vector = np.array([1, tslope])/np.sqrt(1+tslope**2)
			else:
				tangent_vector = np.array([1, slope])/np.sqrt(1+slope**2)
	
		print(f"Tangent vector = {tangent_vector}", flush=True)
		pert_scale = np.logspace(loglims[0], loglims[1], loglims[2])

		# print(pert_scale)

		for pert in pert_scale:
			def mu_init(phi_):
				eq1 = self.delta_mu_s(phi_[0], phi_[1], crit_point[0] + pert * tangent_vector[0], phi_[2])
				eq2 = self.delta_mu_p(phi_[0], phi_[1], crit_point[0] + pert * tangent_vector[0], phi_[2])
				eq3 = self.delta_mu_c(phi_[0], phi_[1], crit_point[0] + pert * tangent_vector[0], phi_[2])
				return [eq1, eq2, eq3]

			root = fsolve(mu_init, [crit_point[0] - pert*tangent_vector[0], crit_point[1] - pert*tangent_vector[1], crit_point[1] + pert*tangent_vector[1]], xtol=1e-16)
			p1   = np.array([root[0], root[1]])
			p2   = np.array([crit_point[0] + pert * tangent_vector[0], root[2]])

			d1   = p1 - np.array([crit_point[0], crit_point[1]])
			d2   = p2 - np.array([crit_point[0], crit_point[1]])
			d1   = d1/np.linalg.norm(d1)
			d2   = d1/np.linalg.norm(d2)

			if np.linalg.norm(d1-d2)<1e-3:
				# print("Directions are too close.")
				continue

			if root[0] > 1 or root[0] < 0 or root[1] > 1 or root[1] < 0 or root[2] > 1 or root[2] < 0:
				# print("Breaking out...")
				continue
			
			if (np.abs(mu_init(root))>1e-14).any():
				# print(f"Bad root: phi1 = ({root[0], root[1]}), phi2 = ({crit_point[0] + pert * tangent_vector[0], root[2]})...", flush=True)
				continue
			# elif np.linalg.norm(p1-p2) < 1e-3: 
			# 	print(f"Too close...")
			# 	continue
			else:
				# print(f"pert = {pert}")
				# print(f"Found root: phi1 = ({root[0], root[1]}), phi2 = ({crit_point[0] + pert * tangent_vector[0], root[2]})...", flush=True)
				if abs(pert*tangent_vector[0]) < 1e-20:
					continue
				good_root = True
				delta_ps2.append(pert*tangent_vector[0])
				phi_s1.append(root[0])
				phi_p1.append(root[1])
				phi_p2.append(root[2])
				phi_s2.append(crit_point[0]+pert*tangent_vector[0])

			# breaking out of pert loop
			if good_root:
				break

		# print(f"Broken out of pert loop!", flush=True)

		if not good_root:
			pass
			# print(f"No neighboring solution found...", flush=True)

		# print("I have found the initial root outside of the critical point. Time to go beyond.", flush=True)
		# print(f"phi_s1 = {phi_s1}", flush=True)
		# print(f"phi_p1 = {phi_p1}", flush=True)
		# print(f"phi_s2 = {phi_s2}", flush=True)
		# print(f"phi_p2 = {phi_p2}", flush=True)

		return phi_s1, phi_s2, phi_p1, phi_p2, delta_ps2

	def calc_perts_in_s2(self, phi1, phi2, delta_phis2):

		a1 = np.float64(self.d_delta_mu_s_dps1(phi1[0], phi1[1], phi2[0], phi2[1]))
		a2 = np.float64(self.d_delta_mu_s_dpp1(phi1[0], phi1[1], phi2[0], phi2[1]))
		a3 = np.float64(self.d_delta_mu_s_dpp2(phi1[0], phi1[1], phi2[0], phi2[1]))
		a4 = np.float64(-delta_phis2 * self.d_delta_mu_s_dps2(phi1[0], phi1[1], phi2[0], phi2[1]))

		b1 = np.float64(self.d_delta_mu_p_dps1(phi1[0], phi1[1], phi2[0], phi2[1]))
		b2 = np.float64(self.d_delta_mu_p_dpp1(phi1[0], phi1[1], phi2[0], phi2[1]))
		b3 = np.float64(self.d_delta_mu_p_dpp2(phi1[0], phi1[1], phi2[0], phi2[1]))
		b4 = np.float64(-delta_phis2 * self.d_delta_mu_p_dps2(phi1[0], phi1[1], phi2[0], phi2[1]))

		c1 = np.float64(self.d_delta_mu_c_dps1(phi1[0], phi1[1], phi2[0], phi2[1]))
		c2 = np.float64(self.d_delta_mu_c_dpp1(phi1[0], phi1[1], phi2[0], phi2[1]))
		c3 = np.float64(self.d_delta_mu_c_dpp2(phi1[0], phi1[1], phi2[0], phi2[1]))
		c4 = np.float64(-delta_phis2 * self.d_delta_mu_c_dps2(phi1[0], phi1[1], phi2[0], phi2[1]))

		D  = np.float64(np.linalg.det(np.array([[a1,a2,a3],[b1,b2,b3],[c1,c2,c3]], dtype=np.float64)))
		Dx = np.float64(np.linalg.det(np.array([[a4,a2,a3],[b4,b2,b3],[c4,c2,c3]], dtype=np.float64)))
		Dy = np.float64(np.linalg.det(np.array([[a1,a4,a3],[b1,b4,b3],[c1,c4,c3]], dtype=np.float64)))
		Dz = np.float64(np.linalg.det(np.array([[a1,a2,a4],[b1,b2,b4],[c1,c2,c4]], dtype=np.float64)))

		# print(f"a1 = {a1}, a2 = {a2}, a3 = {a3}, a4 = {a4}")
		# print(f"b1 = {b1}, b2 = {b2}, b3 = {b3}, b4 = {b4}")
		# print(f"c1 = {c1}, c2 = {c2}, b3 = {c3}, b4 = {c4}")
		# print(f"D = {D}, Dx = {Dx}, Dy = {Dy}, Dz = {Dz}")
		# print(f"dx = {Dx/D}, dy = {Dy/D}, dz = {Dz/D}")

		return Dx/D, Dy/D, Dz/D

	def binodal_run_in_s2(self, crit_point, along_normal=False):
		phi_s1, phi_s2, phi_p1, phi_p2, delta_ps2 = self.find_solution_in_nbrhd_pert_in_s2(crit_point, along_normal)
		if len(phi_s1) == 1:
			return np.array(phi_s1), np.array(phi_s2), np.array(phi_p1), np.array(phi_p2)

		# print(f"delta_ps2 = {delta_ps2}", flush=True)
		condition = (phi_s1[-1] < 1e-12 or phi_p1[-1] < 1e-12 or 1-phi_s1[-1]-phi_p1[-1] < 1e-12) and (phi_s2[-1] < 1e-12 or phi_p2[-1] < 1e-12 or 1-phi_s2[-1]-phi_p2[-1] < 1e-12)

		iterr  = 0
		max_it = 1e+5

		while not condition:
			iterr += 1
			if iterr > max_it:
				print("Completed iterations. Breaking out...", flush=True)
				break

			# print(f"@ i = {iterr}/{max_it}...", flush=True)

			phi1 = [phi_s1[-1], phi_p1[-1]]
			phi2 = [phi_s2[-1], phi_p2[-1]]
			delta_ps1, delta_pp1, delta_pp2 = self.calc_perts_in_s2(phi1, phi2, delta_ps2[-1])
			# print(f"delta_ps1 = {delta_ps1}, delta_pp1 = {delta_pp1}, delta_ps2 = {delta_ps2[-1]}, delta_pp2 = {delta_pp2}", flush=True)

			delta_1 = np.array([delta_ps1, delta_pp1])
			delta_2 = np.array([delta_ps2[-1], delta_pp2])

			delta_1 = delta_1/np.linalg.norm(delta_1)
			delta_2 = delta_2/np.linalg.norm(delta_2)

			if np.linalg.norm(delta_1 - delta_2) < 1e-3:
				# print("Moving in very similar directions.", flush=True)
				continue

			def dmu(phi_):
				eq1 = self.delta_mu_s(phi_[0], phi_[1], phi_s2[-1]+delta_ps2[-1], phi_[2])
				eq2 = self.delta_mu_p(phi_[0], phi_[1], phi_s2[-1]+delta_ps2[-1], phi_[2])
				eq3 = self.delta_mu_c(phi_[0], phi_[1], phi_s2[-1]+delta_ps2[-1], phi_[2])
				return [eq1, eq2, eq3]

			# print(f"Guess provided: phi1 = ({phi_s1[-1]+delta_ps1, phi_p1[-1]+delta_pp1}), phi2 = {phi_s2[-1]+delta_ps2[-1], phi_p2[-1]+delta_pp2}", flush=True)
			root = fsolve(dmu, [phi_s1[-1]+delta_ps1, phi_p1[-1]+delta_pp1, phi_p2[-1]+delta_pp2], xtol=1e-30)

			if root[0] > 1 or root[0] < 0 or root[1] > 1 or root[1] < 0 or root[2] > 1 or root[2] < 0 or (root[0]+root[1]) > 1 or (root[2]+phi_s2[-1]+delta_ps2[-1])>1:
				delta_ps2.append(delta_ps2[-1]/1.5)
				continue

			p1 = np.array([root[0], root[1]])
			p2 = np.array([phi_s2[-1]+delta_ps2[-1], root[2]])

			p1_ = np.array([phi_s1[-1], phi_p1[-1]])
			p2_ = np.array([phi_s2[-1], phi_p2[-1]])

			d1 = p1 - p1_ 
			d2 = p2 - p2_ 

			p1__ = np.array([phi_s1[-2], phi_p1[-2]])
			p2__ = np.array([phi_s2[-2], phi_p2[-2]])

			d1_ = p1_ - p1__
			d2_ = p2_ - p2__

			# if np.linalg.norm(root[1] - phi_p2[-1]+delta_pp2[-1]) > 1e-6:
			# 	print(f"p1 = {p1}, p2 = {p2}")
			# 	print(f"p1_ = {p1_}, p2_ = {p2_}")
			# 	print(f"p1__ = {p1__}, p2__ = {p2__}")
			# 	print ("PROBLEM!")

			if (np.abs(dmu(root))>1e-14).any():
				# print(f"Bad root: phi1 = ({root[0], root[1]}), phi2 = ({phi_s2[-1]+delta_ps2[-1], root[2]})...", flush=True)
				delta_ps2.append(delta_ps2[-1]/1.1)
				continue
			elif np.linalg.norm(p1-p2) < 1e-6: 
				# print(f"Too close: phi1 = ({root[0], root[1]}), phi2 = ({phi_s2[-1]+delta_ps2[-1], root[2]})...", flush=True)
				delta_ps2.append(delta_ps2[-1]/1.1)
				continue
			# elif np.dot(d1/np.linalg.norm(d1), d1_/np.linalg.norm(d1_)) <= 0 or np.dot(d2/np.linalg.norm(d2), d2_/np.linalg.norm(d2_)) <= 0:
				# print(f"Making an about turn: phi1 = ({root[0], root[1]}), phi2 = ({root[2], phi_p2[-1]+delta_pp2[-1]})...", flush=True)
				# delta_pp2.append(delta_pp2[-1]/2)
				# continue

			# print(f"Found root: phi1 = ({root[0], root[1]}), phi2 = ({phi_s2[-1]+delta_ps2[-1], root[2]})...", flush=True)

			phi_s1.append(root[0])
			phi_p1.append(root[1])
			phi_p2.append(root[2])
			phi_s2.append(phi_s2[-1]+delta_ps2[-1])
			delta_ps2.append(delta_ps2[-1])
			condition = (phi_s1[-1] < 1e-12 or phi_p1[-1] < 1e-12 or 1-phi_s1[-1]-phi_p1[-1] < 1e-12)

		phi_s1 = np.array(phi_s1)
		phi_p1 = np.array(phi_p1)
		phi_s2 = np.array(phi_s2)
		phi_p2 = np.array(phi_p2)

		return phi_s1, phi_s2, phi_p1, phi_p2

	def perform_sweep(self, island1, island2):
		# print(f"Performing a sweep to check for chemical potentials...", flush=True)
		chem_pot_s_upper = self.mu_s(island1[:,0], island1[:,1])
		chem_pot_p_upper = self.mu_p(island1[:,0], island1[:,1])
		chem_pot_c_upper = self.mu_c(island1[:,0], island1[:,1])

		masks = np.isinf(chem_pot_s_upper) | np.isnan(chem_pot_s_upper) | np.isinf(chem_pot_p_upper) |\
		np.isnan(chem_pot_p_upper) | np.isinf (chem_pot_c_upper) | np.isnan (chem_pot_c_upper)
		
		chem_pot_s_upper = chem_pot_s_upper [~masks]
		chem_pot_p_upper = chem_pot_p_upper [~masks]
		chem_pot_c_upper = chem_pot_c_upper [~masks]

		# get only the relevant phi_uppers
		island1 = island1[~masks]

		chem_pot_s_lower = self.mu_s(island2[:,0], island2[:,1])
		chem_pot_p_lower = self.mu_p(island2[:,0], island2[:,1])
		chem_pot_c_lower = self.mu_c(island2[:,0], island2[:,1])

		# create a mask for all the good lower chemical potentials
		masks = np.isinf(chem_pot_s_lower) | np.isnan(chem_pot_s_lower) | np.isinf(chem_pot_p_lower) |\
		np.isnan(chem_pot_p_lower) | np.isinf (chem_pot_c_lower) | np.isnan (chem_pot_c_lower)

		# get only the relevant phi_lowers
		chem_pot_s_lower = chem_pot_s_lower [~masks]
		chem_pot_p_lower = chem_pot_p_lower [~masks]
		chem_pot_c_lower = chem_pot_c_lower [~masks]

		# get only the relevant phi_lowers
		island2 = island2[~masks]

		# get an array of these values
		mu_upper   = np.array ([chem_pot_s_upper, chem_pot_p_upper, chem_pot_c_upper]).T
		mu_lower   = np.array ([chem_pot_s_lower, chem_pot_p_lower, chem_pot_c_lower]).T

		# calculate all the distances
		distances       = np.linalg.norm (mu_upper[:, np.newaxis] - mu_lower, axis=2)
		print (f"Memory size of distances is {distances.nbytes/1e+9} gigabytes.", flush=True)
		closest_indices = np.argmin(distances, axis=1)

		# partition, partition... 
		island2         = island2[closest_indices]
		mu_lower        = mu_lower[closest_indices]
		min_distances   = distances[np.arange(len(mu_upper)), closest_indices]

		return [island1, island2, min_distances]

	def binodal_finder(self, island1, island2, hull1, hull2):

		sol1 = np.empty((0,3))
		sol2 = np.empty((0,3))

		for idx, i1 in enumerate(island1):
			def mu_equations(phi):
				eq1 = self.mu_s(phi[0], island1[idx,1]) - self.mu_s(phi[1], phi[2])
				eq2 = self.mu_p(phi[0], island1[idx,1]) - self.mu_p(phi[1], phi[2])
				eq3 = self.mu_c(phi[0], island1[idx,1]) - self.mu_c(phi[1], phi[2])
				return [eq1, eq2, eq3]

			root = fsolve(mu_equations, [island1[idx,0], island2[idx,0], island2[idx,1]])
			p1   = np.array([root[0], island1[idx,1], 1-root[0]-island1[idx,1]])
			p2   = np.array([root[1], root[2], 1-root[1]-root[2]])		

			if (np.abs(np.array(mu_equations(root)))>1e-14).any():
				continue
			else:
				if (ternary.stab_crit(p1[0], p1[1], self.vs, self.vc, self.vp, self.chi_ps, self.chi_pc, self.chi_sc)<0) or (ternary.stab_crit(p2[0], p2[1], self.vs, self.vc, self.vp, self.chi_ps, self.chi_pc, self.chi_sc)<0):
					continue 

				elif np.isnan(ternary.stab_crit(p1[0], p1[1], self.vs, self.vc, self.vp, self.chi_ps, self.chi_pc, self.chi_sc)) or np.isnan(ternary.stab_crit(p2[0], p2[1], self.vs, self.vc, self.vp, self.chi_ps, self.chi_pc, self.chi_sc)):
					continue 
				
				elif np.isinf(ternary.stab_crit(p1[0], p1[1], self.vs, self.vc, self.vp, self.chi_ps, self.chi_pc, self.chi_sc)) or np.isinf(ternary.stab_crit(p2[0], p2[1], self.vs, self.vc, self.vp, self.chi_ps, self.chi_pc, self.chi_sc)):
					continue 

				elif np.linalg.norm(p1-p2) < 1e-4:
					continue

				elif (not hull1.contains_point(p1[0:2])) or (not hull2.contains_point(p2[0:2])):
					continue 

				elif (hull1.contains_point(p1[0:2]) and hull1.contains_point(p2[0:2])) or (hull2.contains_point(p1[0:2]) and hull2.contains_point(p2[0:2])):
					continue

				else:
					sol1 = np.vstack((sol1, p1))
					sol2 = np.vstack((sol2, p2))			

			def mu_equations(phi):
				eq1 = self.mu_s(island1[idx,0], phi[0]) - self.mu_s(phi[1], phi[2])
				eq2 = self.mu_p(island1[idx,0], phi[0]) - self.mu_p(phi[1], phi[2])
				eq3 = self.mu_c(island1[idx,0], phi[0]) - self.mu_c(phi[1], phi[2])
				return [eq1, eq2, eq3]

			root = fsolve(mu_equations, [island1[idx,1], island2[idx,0], island2[idx,1]])
			p1   = np.array([island1[idx,0], root[0], 1-island1[idx,0]-root[0]])
			p2   = np.array([root[1], root[2], 1-root[1]-root[2]])

			if (np.abs(np.array(mu_equations(root)))>1e-14).any():
				continue
			else:
				if (ternary.stab_crit(p1[0], p1[1], self.vs, self.vc, self.vp, self.chi_ps, self.chi_pc, self.chi_sc)<0) or (ternary.stab_crit(p2[0], p2[1], self.vs, self.vc, self.vp, self.chi_ps, self.chi_pc, self.chi_sc)<0):
					continue 

				elif np.isnan(ternary.stab_crit(p1[0], p1[1], self.vs, self.vc, self.vp, self.chi_ps, self.chi_pc, self.chi_sc)) or np.isnan(ternary.stab_crit(p2[0], p2[1], self.vs, self.vc, self.vp, self.chi_ps, self.chi_pc, self.chi_sc)):
					continue 
				
				elif np.isinf(ternary.stab_crit(p1[0], p1[1], self.vs, self.vc, self.vp, self.chi_ps, self.chi_pc, self.chi_sc)) or np.isinf(ternary.stab_crit(p2[0], p2[1], self.vs, self.vc, self.vp, self.chi_ps, self.chi_pc, self.chi_sc)):
					continue

				elif np.linalg.norm(p1-p2) < 1e-4:
					continue

				elif (not hull1.contains_point(p1[0:2])) or (not hull2.contains_point(p2[0:2])):
					continue 

				elif (hull1.contains_point(p1[0:2]) and hull1.contains_point(p2[0:2])) or (hull2.contains_point(p1[0:2]) and hull2.contains_point(p2[0:2])):
					continue

				else:
					sol1 = np.vstack((sol1, p1))
					sol2 = np.vstack((sol2, p2))
		

		return sol1, sol2 

	def binodal_finder_(self, island1, island2, hull):

		sol1 = np.empty((0,3))
		sol2 = np.empty((0,3))

		for idx, i1 in enumerate(island1):
			def mu_equations(phi):
				eq1 = self.mu_s(phi[0], island1[idx,1]) - self.mu_s(phi[1], phi[2])
				eq2 = self.mu_p(phi[0], island1[idx,1]) - self.mu_p(phi[1], phi[2])
				eq3 = self.mu_c(phi[0], island1[idx,1]) - self.mu_c(phi[1], phi[2])
				return [eq1, eq2, eq3]

			root = fsolve(mu_equations, [island1[idx,0], island2[idx,0], island2[idx,1]])
			p1   = np.array([root[0], island1[idx,1], 1-root[0]-island1[idx,1]])
			p2   = np.array([root[1], root[2], 1-root[1]-root[2]])		

			if (np.abs(np.array(mu_equations(root)))>1e-14).any():
				continue
			else:
				if (ternary.stab_crit(p1[0], p1[1], self.vs, self.vc, self.vp, self.chi_ps, self.chi_pc, self.chi_sc)<0) or (ternary.stab_crit(p2[0], p2[1], self.vs, self.vc, self.vp, self.chi_ps, self.chi_pc, self.chi_sc)<0):
					continue 

				elif np.isnan(ternary.stab_crit(p1[0], p1[1], self.vs, self.vc, self.vp, self.chi_ps, self.chi_pc, self.chi_sc)) or np.isnan(ternary.stab_crit(p2[0], p2[1], self.vs, self.vc, self.vp, self.chi_ps, self.chi_pc, self.chi_sc)):
					continue 
				
				elif np.isinf(ternary.stab_crit(p1[0], p1[1], self.vs, self.vc, self.vp, self.chi_ps, self.chi_pc, self.chi_sc)) or np.isinf(ternary.stab_crit(p2[0], p2[1], self.vs, self.vc, self.vp, self.chi_ps, self.chi_pc, self.chi_sc)):
					continue 

				elif np.linalg.norm(p1-p2) < 1e-4:
					continue

				elif not(hull.contains_point(p1[0:2])) or not(hull.contains_point(p2[0:2])):
					continue 

				else:
					sol1 = np.vstack((sol1, p1))
					sol2 = np.vstack((sol2, p2))

			def mu_equations(phi):
				eq1 = self.mu_s(island1[idx,0], phi[0]) - self.mu_s(phi[1], phi[2])
				eq2 = self.mu_p(island1[idx,0], phi[0]) - self.mu_p(phi[1], phi[2])
				eq3 = self.mu_c(island1[idx,0], phi[0]) - self.mu_c(phi[1], phi[2])
				return [eq1, eq2, eq3]

			root = fsolve(mu_equations, [island1[idx,1], island2[idx,0], island2[idx,1]])
			p1   = np.array([island1[idx,0], root[0], 1-island1[idx,0]-root[0]])
			p2   = np.array([root[1], root[2], 1-root[1]-root[2]])

			if (np.abs(np.array(mu_equations(root)))>1e-14).any():
				continue
			else:
				if (ternary.stab_crit(p1[0], p1[1], self.vs, self.vc, self.vp, self.chi_ps, self.chi_pc, self.chi_sc)<0) or (ternary.stab_crit(p2[0], p2[1], self.vs, self.vc, self.vp, self.chi_ps, self.chi_pc, self.chi_sc)<0):
					continue 

				elif np.isnan(ternary.stab_crit(p1[0], p1[1], self.vs, self.vc, self.vp, self.chi_ps, self.chi_pc, self.chi_sc)) or np.isnan(ternary.stab_crit(p2[0], p2[1], self.vs, self.vc, self.vp, self.chi_ps, self.chi_pc, self.chi_sc)):
					continue 
				
				elif np.isinf(ternary.stab_crit(p1[0], p1[1], self.vs, self.vc, self.vp, self.chi_ps, self.chi_pc, self.chi_sc)) or np.isinf(ternary.stab_crit(p2[0], p2[1], self.vs, self.vc, self.vp, self.chi_ps, self.chi_pc, self.chi_sc)):
					continue

				elif np.linalg.norm(p1-p2) < 1e-4:
					continue
				
				elif not(hull.contains_point(p1[0:2])) or not(hull.contains_point(p2[0:2])):
					continue 

				else:
					sol1 = np.vstack((sol1, p1))
					sol2 = np.vstack((sol2, p2))
		

		return sol1, sol2 

	def binodal_finder__(self, island1, island2):

		sol1 = np.empty((0,3))
		sol2 = np.empty((0,3))

		for idx, i1 in enumerate(island1):
			def mu_equations(phi):
				eq1 = self.mu_s(phi[0], island1[idx,1]) - self.mu_s(phi[1], phi[2])
				eq2 = self.mu_p(phi[0], island1[idx,1]) - self.mu_p(phi[1], phi[2])
				eq3 = self.mu_c(phi[0], island1[idx,1]) - self.mu_c(phi[1], phi[2])
				return [eq1, eq2, eq3]

			root = fsolve(mu_equations, [island1[idx,0], island2[idx,0], island2[idx,1]])
			p1   = np.array([root[0], island1[idx,1], 1-root[0]-island1[idx,1]])
			p2   = np.array([root[1], root[2], 1-root[1]-root[2]])		

			if (np.abs(np.array(mu_equations(root)))>1e-14).any():
				continue
			else:
				if (ternary.stab_crit(p1[0], p1[1], self.vs, self.vc, self.vp, self.chi_ps, self.chi_pc, self.chi_sc)<0) or (ternary.stab_crit(p2[0], p2[1], self.vs, self.vc, self.vp, self.chi_ps, self.chi_pc, self.chi_sc)<0):
					continue 

				elif np.isnan(ternary.stab_crit(p1[0], p1[1], self.vs, self.vc, self.vp, self.chi_ps, self.chi_pc, self.chi_sc)) or np.isnan(ternary.stab_crit(p2[0], p2[1], self.vs, self.vc, self.vp, self.chi_ps, self.chi_pc, self.chi_sc)):
					continue 
				
				elif np.isinf(ternary.stab_crit(p1[0], p1[1], self.vs, self.vc, self.vp, self.chi_ps, self.chi_pc, self.chi_sc)) or np.isinf(ternary.stab_crit(p2[0], p2[1], self.vs, self.vc, self.vp, self.chi_ps, self.chi_pc, self.chi_sc)):
					continue 

				elif np.linalg.norm(p1-p2) < 1e-4:
					continue

				else:
					sol1 = np.vstack((sol1, p1))
					sol2 = np.vstack((sol2, p2))			

			def mu_equations(phi):
				eq1 = self.mu_s(island1[idx,0], phi[0]) - self.mu_s(phi[1], phi[2])
				eq2 = self.mu_p(island1[idx,0], phi[0]) - self.mu_p(phi[1], phi[2])
				eq3 = self.mu_c(island1[idx,0], phi[0]) - self.mu_c(phi[1], phi[2])
				return [eq1, eq2, eq3]

			root = fsolve(mu_equations, [island1[idx,1], island2[idx,0], island2[idx,1]])
			p1   = np.array([island1[idx,0], root[0], 1-island1[idx,0]-root[0]])
			p2   = np.array([root[1], root[2], 1-root[1]-root[2]])

			if (np.abs(np.array(mu_equations(root)))>1e-14).any():
				continue
			else:
				if (ternary.stab_crit(p1[0], p1[1], self.vs, self.vc, self.vp, self.chi_ps, self.chi_pc, self.chi_sc)<0) or (ternary.stab_crit(p2[0], p2[1], self.vs, self.vc, self.vp, self.chi_ps, self.chi_pc, self.chi_sc)<0):
					continue 

				elif np.isnan(ternary.stab_crit(p1[0], p1[1], self.vs, self.vc, self.vp, self.chi_ps, self.chi_pc, self.chi_sc)) or np.isnan(ternary.stab_crit(p2[0], p2[1], self.vs, self.vc, self.vp, self.chi_ps, self.chi_pc, self.chi_sc)):
					continue 
				
				elif np.isinf(ternary.stab_crit(p1[0], p1[1], self.vs, self.vc, self.vp, self.chi_ps, self.chi_pc, self.chi_sc)) or np.isinf(ternary.stab_crit(p2[0], p2[1], self.vs, self.vc, self.vp, self.chi_ps, self.chi_pc, self.chi_sc)):
					continue

				elif np.linalg.norm(p1-p2) < 1e-4:
					continue

				else:
					sol1 = np.vstack((sol1, p1))
					sol2 = np.vstack((sol2, p2))
		

		return sol1, sol2 

	def solve_within(self, side1, side2, center, central_axis):

		b1 = np.empty((0,3))
		b2 = np.empty((0,3))

		for idx, pt in enumerate(side1):
			# print(f"pt = {pt}", flush=True)
			def mu_equations(phi):
				eq1 = self.mu_s(side1[idx,0], phi[0]) - self.mu_s(phi[1], phi[2])
				eq2 = self.mu_p(side1[idx,0], phi[0]) - self.mu_p(phi[1], phi[2])
				eq3 = self.mu_c(side1[idx,0], phi[0]) - self.mu_c(phi[1], phi[2])
				# print(f"eq1 = {eq1}, eq2 = {eq2}, eq3 = {eq3}", flush=True)
				return [eq1, eq2, eq3]

			for tidx, tpt in enumerate(side2):
				root = fsolve(mu_equations, [pt[1], tpt[0], tpt[1]])
				if (np.abs(np.array(mu_equations(root)))>1e-14).any():
					continue
				else:
					p1 = np.array([pt[0], root[0], 1-root[0]-pt[0]])
					p2 = np.array([root[1], root[2], 1-root[1]-root[2]])

					if np.isnan(ternary.stab_crit (p1[0], p1[1], self.vs, self.vc, self.vp, self.chi_ps, self.chi_pc, self.chi_sc)) or np.isnan(ternary.stab_crit (p2[0], p2[1], self.vs, self.vc, self.vp, self.chi_ps, self.chi_pc, self.chi_sc)):
						continue

					elif np.isinf(ternary.stab_crit (p1[0], p1[1], self.vs, self.vc, self.vp, self.chi_ps, self.chi_pc, self.chi_sc)) or np.isinf(ternary.stab_crit (p2[0], p2[1], self.vs, self.vc, self.vp, self.chi_ps, self.chi_pc, self.chi_sc)):
						continue

					elif np.linalg.norm(p1[0:2]-p2[0:2]) < 1e-6:
						continue

					else:
						if np.sign(np.cross(central_axis, p1[0:2]-center[0:2])) == np.sign(np.cross(central_axis, p2[0:2]-center[0:2])):
							continue
						elif np.cross(central_axis, p1[0:2]-center[0:2]) >=0:
							b1 = np.vstack((b1, p1))
							b2 = np.vstack((b2, p2))
						else:
							b1 = np.vstack((b1, p2))
							b2 = np.vstack((b2, p1))
						break

		return b1, b2

# end of class sym_mu_ps

class sym_mu_sc:
	def __init__(self, inputs, spinodal):
		self.chi_sc   = inputs["chi_sc"]
		self.chi_ps   = inputs["chi_ps"]
		self.chi_pc   = inputs["chi_pc"]
		self.vs       = inputs["vs"]
		self.vc       = inputs["vc"]
		self.vp       = inputs["vp"]
		self.spinodal = spinodal
		self.setup()
		return

	def setup(self):

		phi_s1, phi_c1, phi_s2, phi_c2 = sym.symbols('phi_s1 phi_c1 phi_s2 phi_c2')

		mu_s1 = sym.log(phi_s1)          + 1 - phi_s1            - self.vs/self.vp * (1-phi_s1-phi_c1) - self.vs/self.vc * phi_c1            + self.vs * ((1-phi_s1-phi_c1)**2 * self.chi_ps + phi_c1**2 * self.chi_sc            + (1-phi_s1-phi_c1) * phi_c1            * (self.chi_ps + self.chi_sc - self.chi_pc))
		mu_p1 = sym.log(1-phi_s1-phi_c1) + 1 - (1-phi_s1-phi_c1) - self.vp/self.vs * phi_s1            - self.vp/self.vc * phi_c1            + self.vp * (phi_s1**2 * self.chi_ps            + phi_c1**2 * self.chi_pc            + phi_s1            * phi_c1            * (self.chi_ps + self.chi_pc - self.chi_sc))
		mu_c1 = sym.log(phi_c1)          + 1 - phi_c1            - self.vc/self.vs * phi_s1            - self.vc/self.vp * (1-phi_s1-phi_c1) + self.vc * (phi_s1**2 * self.chi_sc            + (1-phi_s1-phi_c1)**2 * self.chi_pc + phi_s1            * (1-phi_s1-phi_c1) * (self.chi_sc + self.chi_pc - self.chi_ps))

		mu_s2 = sym.log(phi_s2)          + 1 - phi_s2            - self.vs/self.vp * (1-phi_s2-phi_c2) - self.vs/self.vc * phi_c2            + self.vs * ((1-phi_s2-phi_c2)**2 * self.chi_ps + phi_c2**2 * self.chi_sc            + (1-phi_s2-phi_c2) * phi_c2            * (self.chi_ps + self.chi_sc - self.chi_pc))
		mu_p2 = sym.log(1-phi_s2-phi_c2) + 1 - (1-phi_s2-phi_c2) - self.vp/self.vs * phi_s2            - self.vp/self.vc * phi_c2            + self.vp * (phi_s2**2            * self.chi_ps + phi_c2**2 * self.chi_pc            + phi_s2            * phi_c2            * (self.chi_ps + self.chi_pc - self.chi_sc))
		mu_c2 = sym.log(phi_c2)          + 1 - phi_c2            - self.vc/self.vs * phi_s2            - self.vc/self.vp * (1-phi_s2-phi_c2) + self.vc * (phi_s2**2            * self.chi_sc + (1-phi_s2-phi_c2)**2 * self.chi_pc + phi_s2            * (1-phi_s2-phi_c2) * (self.chi_sc + self.chi_pc - self.chi_ps))

		delta_mu_s = mu_s1 - mu_s2
		delta_mu_p = mu_p1 - mu_p2
		delta_mu_c = mu_c1 - mu_c2

		# lambdify the function
		L_dmu_s = sym.lambdify([phi_s1, phi_c1, phi_s2, phi_c2], delta_mu_s)
		L_dmu_p = sym.lambdify([phi_s1, phi_c1, phi_s2, phi_c2], delta_mu_p)
		L_dmu_c = sym.lambdify([phi_s1, phi_c1, phi_s2, phi_c2], delta_mu_c)

		# do the differentiation of mu_s 
		d_delta_mu_s_dps1 = sym.diff(delta_mu_s, phi_s1)
		d_delta_mu_s_dps2 = sym.diff(delta_mu_s, phi_s2)
		d_delta_mu_s_dpc1 = sym.diff(delta_mu_s, phi_c1)
		d_delta_mu_s_dpc2 = sym.diff(delta_mu_s, phi_c2)

		# lambdify the function
		L_d_delta_mu_s_dps1 = sym.lambdify([phi_s1, phi_c1, phi_s2, phi_c2], d_delta_mu_s_dps1)
		L_d_delta_mu_s_dps2 = sym.lambdify([phi_s1, phi_c1, phi_s2, phi_c2], d_delta_mu_s_dps2)
		L_d_delta_mu_s_dpc1 = sym.lambdify([phi_s1, phi_c1, phi_s2, phi_c2], d_delta_mu_s_dpc1)
		L_d_delta_mu_s_dpc2 = sym.lambdify([phi_s1, phi_c1, phi_s2, phi_c2], d_delta_mu_s_dpc2)

		# do the differentiation of mu_p
		d_delta_mu_p_dps1 = sym.diff(delta_mu_p, phi_s1)
		d_delta_mu_p_dps2 = sym.diff(delta_mu_p, phi_s2)
		d_delta_mu_p_dpc1 = sym.diff(delta_mu_p, phi_c1)
		d_delta_mu_p_dpc2 = sym.diff(delta_mu_p, phi_c2)

		# lambdify the function 
		L_d_delta_mu_p_dps1 = sym.lambdify([phi_s1, phi_c1, phi_s2, phi_c2], d_delta_mu_p_dps1)
		L_d_delta_mu_p_dps2 = sym.lambdify([phi_s1, phi_c1, phi_s2, phi_c2], d_delta_mu_p_dps2)
		L_d_delta_mu_p_dpc1 = sym.lambdify([phi_s1, phi_c1, phi_s2, phi_c2], d_delta_mu_p_dpc1)
		L_d_delta_mu_p_dpc2 = sym.lambdify([phi_s1, phi_c1, phi_s2, phi_c2], d_delta_mu_p_dpc2)

		# do the differentiation of mu_c
		d_delta_mu_c_dps1   = sym.diff(delta_mu_c, phi_s1)
		d_delta_mu_c_dps2   = sym.diff(delta_mu_c, phi_s2)
		d_delta_mu_c_dpc1   = sym.diff(delta_mu_c, phi_c1)
		d_delta_mu_c_dpc2   = sym.diff(delta_mu_c, phi_c2)

		# lambdify the function 
		L_d_delta_mu_c_dps1 = sym.lambdify([phi_s1, phi_c1, phi_s2, phi_c2], d_delta_mu_c_dps1)
		L_d_delta_mu_c_dps2 = sym.lambdify([phi_s1, phi_c1, phi_s2, phi_c2], d_delta_mu_c_dps2)
		L_d_delta_mu_c_dpc1 = sym.lambdify([phi_s1, phi_c1, phi_s2, phi_c2], d_delta_mu_c_dpc1)
		L_d_delta_mu_c_dpc2 = sym.lambdify([phi_s1, phi_c1, phi_s2, phi_c2], d_delta_mu_c_dpc2)

		# get the new function
		self.delta_mu_s = lambda s1, c1, s2, c2: L_dmu_s(s1, c1, s2, c2)
		self.delta_mu_p = lambda s1, c1, s2, c2: L_dmu_p(s1, c1, s2, c2)
		self.delta_mu_c = lambda s1, c1, s2, c2: L_dmu_c(s1, c1, s2, c2)

		# get the new function
		self.d_delta_mu_s_dps1 = lambda s1, c1, s2, c2: L_d_delta_mu_s_dps1(s1, c1, s2, c2)
		self.d_delta_mu_s_dpc1 = lambda s1, c1, s2, c2: L_d_delta_mu_s_dpc1(s1, c1, s2, c2)
		self.d_delta_mu_s_dps2 = lambda s1, c1, s2, c2: L_d_delta_mu_s_dps2(s1, c1, s2, c2)
		self.d_delta_mu_s_dpc2 = lambda s1, c1, s2, c2: L_d_delta_mu_s_dpc2(s1, c1, s2, c2)

		# get the new function
		self.d_delta_mu_p_dps1 = lambda s1, c1, s2, c2: L_d_delta_mu_p_dps1(s1, c1, s2, c2)
		self.d_delta_mu_p_dpc1 = lambda s1, c1, s2, c2: L_d_delta_mu_p_dpc1(s1, c1, s2, c2)
		self.d_delta_mu_p_dps2 = lambda s1, c1, s2, c2: L_d_delta_mu_p_dps2(s1, c1, s2, c2)
		self.d_delta_mu_p_dpc2 = lambda s1, c1, s2, c2: L_d_delta_mu_p_dpc2(s1, c1, s2, c2)

		# get the new function
		self.d_delta_mu_c_dps1 = lambda s1, c1, s2, c2: L_d_delta_mu_c_dps1(s1, c1, s2, c2)
		self.d_delta_mu_c_dpc1 = lambda s1, c1, s2, c2: L_d_delta_mu_c_dpc1(s1, c1, s2, c2)
		self.d_delta_mu_c_dps2 = lambda s1, c1, s2, c2: L_d_delta_mu_c_dps2(s1, c1, s2, c2)
		self.d_delta_mu_c_dpc2 = lambda s1, c1, s2, c2: L_d_delta_mu_c_dpc2(s1, c1, s2, c2)

		return

	def find_solution_in_nbrhd_pert_in_c2(self, crit_point, along_normal=False, loglims=[-3,-10,100]):

		# print(f"crit point of interest = {crit_point}", flush=True)
		delta_pc2 = []
		phi_s1    = [crit_point[0]]
		phi_s2    = [crit_point[0]]
		phi_c1    = [crit_point[2]]
		phi_c2    = [crit_point[2]]

		good_root = False

		slope          = tangent.tangent2(self.vs, self.vc, self.vp, crit_point[0], crit_point[1], self.chi_pc, self.chi_ps, self.chi_sc, self.spinodal.root_up_s, self.spinodal.root_lo_s)
		if np.isinf(slope) or np.isnan(slope):
			tangent_vector = np.array([0,1])
		else:
			tangent_vector = np.array([1, -(slope+1)])/np.sqrt(1+(-(slope+1))**2)
		if along_normal:
			if np.isinf(slope) or np.isnan(slope) or abs(slope) < 1e-6:
				tangent_vector = np.array([0, 1])
			else:
				slope = -1/slope
				tangent_vector = np.array([1, -(slope+1)])/np.sqrt(1+(-(slope+1))**2)

		print(f"Tangent vector = {tangent_vector}", flush=True)
		# print(f"slope = {slope}")

		pert_scale = np.logspace(loglims[0], loglims[1], loglims[2])

		# print(pert_scale)

		for pert in pert_scale:
			def mu_init(phi_):
				eq1 = self.delta_mu_s(phi_[0], phi_[1], phi_[2], crit_point[2] + pert * tangent_vector[1])
				eq2 = self.delta_mu_p(phi_[0], phi_[1], phi_[2], crit_point[2] + pert * tangent_vector[1])
				eq3 = self.delta_mu_c(phi_[0], phi_[1], phi_[2], crit_point[2] + pert * tangent_vector[1])
				return [eq1, eq2, eq3]

			root = fsolve(mu_init, [crit_point[0] - pert*tangent_vector[0], crit_point[2] - pert*tangent_vector[1], crit_point[0] + pert*tangent_vector[0]], xtol=1e-16)
			p1   = np.array([root[0], root[1]])
			p2   = np.array([root[2], crit_point[2] + pert * tangent_vector[1]])

			d1   = p1 - np.array([crit_point[0], crit_point[2]])
			d2   = p2 - np.array([crit_point[0], crit_point[2]])
			d1   = d1/np.linalg.norm(d1)
			d2   = d1/np.linalg.norm(d2)

			if np.linalg.norm(d1-d2)<1e-3:
				# print("Directions are too close.")
				continue

			if root[0] > 1 or root[0] < 0 or root[1] > 1 or root[1] < 0 or root[2] > 1 or root[2] < 0:
				# print("Breaking out...")
				continue
			
			if (np.abs(mu_init(root))>1e-14).any():
				# print(f"Bad root: phi1 = ({root[0], root[1]}), phi2 = ({root[2], crit_point[2] + pert * tangent_vector[1]})...", flush=True)
				continue

			else:
				# print(f"pert = {pert}")
				# print(f"Found root: phi1 = ({root[0], root[1]}), phi2 = ({root[2], crit_point[2] + pert * tangent_vector[1]})...", flush=True)
				good_root = True
				if abs(pert*tangent_vector[1]) < 1e-20:
					continue
				delta_pc2.append(pert*tangent_vector[1])
				phi_s1.append(root[0])
				phi_c1.append(root[1])
				phi_s2.append(root[2])
				phi_c2.append(crit_point[2]+pert*tangent_vector[1])

			# breaking out of pert loop
			if good_root:
				break

		# print(f"Broken out of pert loop!", flush=True)

		if not good_root:
			# print(f"No neighboring solution found...", flush=True)
			exit()

		# print("I have found the initial root outside of the critical point. Time to go beyond.", flush=True)
		# print(f"phi_s1 = {phi_s1}", flush=True)
		# print(f"phi_p1 = {phi_c1}", flush=True)
		# print(f"phi_s2 = {phi_s2}", flush=True)
		# print(f"phi_p2 = {phi_c2}", flush=True)

		# exit() 

		return phi_s1, phi_s2, phi_c1, phi_c2, delta_pc2

	def calc_perts_in_c2(self, phi1, phi2, delta_phic2):

		a1 = np.float64(self.d_delta_mu_s_dps1(phi1[0], phi1[1], phi2[0], phi2[1]))
		a2 = np.float64(self.d_delta_mu_s_dpc1(phi1[0], phi1[1], phi2[0], phi2[1]))
		a3 = np.float64(self.d_delta_mu_s_dps2(phi1[0], phi1[1], phi2[0], phi2[1]))
		a4 = np.float64(-delta_phic2 * self.d_delta_mu_s_dpc2(phi1[0], phi1[1], phi2[0], phi2[1]))

		b1 = np.float64(self.d_delta_mu_p_dps1(phi1[0], phi1[1], phi2[0], phi2[1]))
		b2 = np.float64(self.d_delta_mu_p_dpc1(phi1[0], phi1[1], phi2[0], phi2[1]))
		b3 = np.float64(self.d_delta_mu_p_dps2(phi1[0], phi1[1], phi2[0], phi2[1]))
		b4 = np.float64(-delta_phic2 * self.d_delta_mu_p_dpc2(phi1[0], phi1[1], phi2[0], phi2[1]))

		c1 = np.float64(self.d_delta_mu_c_dps1(phi1[0], phi1[1], phi2[0], phi2[1]))
		c2 = np.float64(self.d_delta_mu_c_dpc1(phi1[0], phi1[1], phi2[0], phi2[1]))
		c3 = np.float64(self.d_delta_mu_c_dps2(phi1[0], phi1[1], phi2[0], phi2[1]))
		c4 = np.float64(-delta_phic2 * self.d_delta_mu_c_dpc2(phi1[0], phi1[1], phi2[0], phi2[1]))

		D  = np.float64(np.linalg.det(np.array([[a1,a2,a3],[b1,b2,b3],[c1,c2,c3]], dtype=np.float64)))
		Dx = np.float64(np.linalg.det(np.array([[a4,a2,a3],[b4,b2,b3],[c4,c2,c3]], dtype=np.float64)))
		Dy = np.float64(np.linalg.det(np.array([[a1,a4,a3],[b1,b4,b3],[c1,c4,c3]], dtype=np.float64)))
		Dz = np.float64(np.linalg.det(np.array([[a1,a2,a4],[b1,b2,b4],[c1,c2,c4]], dtype=np.float64)))

		# print(f"a1 = {a1}, a2 = {a2}, a3 = {a3}, a4 = {a4}")
		# print(f"b1 = {b1}, b2 = {b2}, b3 = {b3}, b4 = {b4}")
		# print(f"c1 = {c1}, c2 = {c2}, b3 = {c3}, b4 = {c4}")
		# print(f"D = {D}, Dx = {Dx}, Dy = {Dy}, Dz = {Dz}")
		# print(f"dx = {Dx/D}, dy = {Dy/D}, dz = {Dz/D}")

		return Dx/D, Dy/D, Dz/D

	def binodal_run_in_c2(self, crit_point, along_normal=False):

		phi_s1, phi_s2, phi_c1, phi_c2, delta_pc2 = self.find_solution_in_nbrhd_pert_in_c2(crit_point, along_normal)
		if len(phi_s1) == 1:
			return np.array(phi_s1), np.array(phi_s2), np.array(phi_c1), np.array(phi_c2)

		# print(f"delta_pc2 = {delta_pc2}", flush=True)
		condition = (phi_s1[-1] < 1e-12 or phi_c1[-1] < 1e-12 or 1-phi_s1[-1]-phi_c1[-1] < 1e-12) and (phi_s2[-1] < 1e-12 or phi_c2[-1] < 1e-12 or 1-phi_s2[-1]-phi_c2[-1] < 1e-12)

		# for i in range(ncycles):
		iterr  = 0 
		max_it = 1e+5

		while not condition:
			iterr += 1
			if iterr > max_it:
				# print("Completed iterations. Breaking out...", flush=True)
				break
			# preprocess...
			# preprocessor = np.array(delta_pp2[-100:])
			# if (np.abs(preprocessor) < 1e-16).all():
			# 	pass
			# delta_pp2.append(1e-6)

			# print(f"@ i = {iterr}/{max_it}...", flush=True)

			phi1 = [phi_s1[-1], phi_c1[-1]]
			phi2 = [phi_s2[-1], phi_c2[-1]]
			delta_ps1, delta_pc1, delta_ps2 = self.calc_perts_in_c2(phi1, phi2, delta_pc2[-1])
			# print(f"delta_ps1 = {delta_ps1}, delta_pc1 = {delta_pc1}, delta_ps2 = {delta_ps2}, delta_pc2 = {delta_pc2[-1]}", flush=True)

			delta_1 = np.array([delta_ps1, delta_pc1])
			delta_2 = np.array([delta_ps2, delta_pc2[-1]])

			delta_1 = delta_1/np.linalg.norm(delta_1)
			delta_2 = delta_2/np.linalg.norm(delta_2)

			if np.linalg.norm(delta_1 - delta_2) < 1e-3:
				# print("Moving in very similar directions.", flush=True)
				continue

			def dmu(phi_):
				eq1 = self.delta_mu_s(phi_[0], phi_[1], phi_[2], phi_c2[-1]+delta_pc2[-1])
				eq2 = self.delta_mu_p(phi_[0], phi_[1], phi_[2], phi_c2[-1]+delta_pc2[-1])
				eq3 = self.delta_mu_c(phi_[0], phi_[1], phi_[2], phi_c2[-1]+delta_pc2[-1])
				return [eq1, eq2, eq3]

			# print(f"Guess provided: phi1 = ({phi_s1[-1]+delta_ps1, phi_c1[-1]+delta_pc1}), phi2 = {phi_s2[-1]+delta_ps2, phi_c2[-1]+delta_pc2[-1]}", flush=True)
			root = fsolve(dmu, [phi_s1[-1]+delta_ps1, phi_c1[-1]+delta_pc1, phi_s2[-1]+delta_ps2], xtol=1e-30)

			if root[0] > 1 or root[0] < 0 or root[1] > 1 or root[1] < 0 or root[2] > 1 or root[2] < 0 or root[0]+root[1]>1 or root[2]+phi_c2[-1]+delta_pc2[-1]>1:
				delta_pc2.append(delta_pc2[-1]/1.1)
				continue

			p1 = np.array([root[0], root[1]])
			p2 = np.array([root[2], phi_c2[-1]+delta_pc2[-1]])

			p1_ = np.array([phi_s1[-1], phi_c1[-1]])
			p2_ = np.array([phi_s2[-1], phi_c2[-1]])

			d1 = p1 - p1_ 
			d2 = p2 - p2_ 

			p1__ = np.array([phi_s1[-2], phi_c1[-2]])
			p2__ = np.array([phi_s2[-2], phi_c2[-2]])

			d1_ = p1_ - p1__
			d2_ = p2_ - p2__


			# if np.linalg.norm(root[1] - phi_c2[-1]+delta_pc2[-1]) > 1e-6:
			# 	print(f"p1 = {p1}, p2 = {p2}")
			# 	print(f"p1_ = {p1_}, p2_ = {p2_}")
			# 	print(f"p1__ = {p1__}, p2__ = {p2__}")
			# 	print ("PROBLEM!")

			if (np.abs(dmu(root))>1e-14).any():
				# print(f"Bad root: phi1 = ({root[0], root[1]}), phi2 = ({root[2], phi_c2[-1]+delta_pc2[-1]})...", flush=True)
				delta_pc2.append(delta_pc2[-1]/1.1)
				continue
			elif np.linalg.norm(p1-p2) < 1e-6: 
				# print(f"Too close: phi1 = ({root[0], root[1]}), phi2 = ({root[2], phi_c2[-1]+delta_pc2[-1]})...", flush=True)
				delta_pc2.append(delta_pc2[-1]/1.1)
				continue
			# elif np.dot(d1/np.linalg.norm(d1), d1_/np.linalg.norm(d1_)) <= 0 or np.dot(d2/np.linalg.norm(d2), d2_/np.linalg.norm(d2_)) <= 0:
				# print(f"Making an about turn: phi1 = ({root[0], root[1]}), phi2 = ({root[2], phi_p2[-1]+delta_pp2[-1]})...", flush=True)
				# delta_pp2.append(delta_pp2[-1]/2)
				# continue


			# print(f"Found root: phi1 = ({root[0], root[1]}), phi2 = ({root[2], phi_c2[-1]+delta_pc2[-1]})...", flush=True)

			phi_s1.append(root[0])
			phi_c1.append(root[1])
			phi_s2.append(root[2])
			phi_c2.append(phi_c2[-1]+delta_pc2[-1])
			delta_pc2.append(delta_pc2[-1])
			condition = (phi_s1[-1] < 1e-12 or phi_c1[-1] < 1e-12 or 1-phi_s1[-1]-phi_c1[-1] < 1e-12)

		phi_s1 = np.array(phi_s1)
		phi_c1 = np.array(phi_c1)
		phi_s2 = np.array(phi_s2)
		phi_c2 = np.array(phi_c2)

		return phi_s1, phi_s2, phi_c1, phi_c2

	def find_solution_in_nbrhd_pert_in_s2(self, crit_point, along_normal=False, loglims=[-3,-10,100]):

		# print(f"crit point of interest = {crit_point}", flush=True)
		delta_ps2 = []
		phi_s1    = [crit_point[0]]
		phi_s2    = [crit_point[0]]
		phi_c1    = [crit_point[2]]
		phi_c2    = [crit_point[2]]

		good_root = False

		slope          = tangent.tangent2(self.vs, self.vc, self.vp, crit_point[0], crit_point[1], self.chi_pc, self.chi_ps, self.chi_sc, self.spinodal.root_up_s, self.spinodal.root_lo_s)
		if np.isinf(slope) or np.isnan(slope):
			tangent_vector = np.array([0,1])
		else:
			tangent_vector = np.array([1, -(slope+1)])/np.sqrt(1+(-(slope+1))**2)
		if along_normal:
			if np.isinf(slope) or np.isnan(slope) or abs(slope) < 1e-6 :
				tangent_vector = np.array([0, 1])
			else:
				slope = -1/slope
				tangent_vector = np.array([1, -(slope+1)])/np.sqrt(1+(-(slope+1))**2)

		print(f"Tangent vector = {tangent_vector}", flush=True)
		# print(f"slope = {slope}")

		pert_scale = np.logspace(loglims[0], loglims[1], loglims[2])

		for pert in pert_scale:
			def mu_init(phi_):
				eq1 = self.delta_mu_s(phi_[0], phi_[1], crit_point[0] + pert * tangent_vector[0], phi_[2])
				eq2 = self.delta_mu_p(phi_[0], phi_[1], crit_point[0] + pert * tangent_vector[0], phi_[2])
				eq3 = self.delta_mu_c(phi_[0], phi_[1], crit_point[0] + pert * tangent_vector[0], phi_[2])
				return [eq1, eq2, eq3]

			root = fsolve(mu_init, [crit_point[0] - pert*tangent_vector[0], crit_point[2] - pert*tangent_vector[1], crit_point[2] + pert*tangent_vector[1]], xtol=1e-16)
			p1   = np.array([root[0], root[1]])
			p2   = np.array([crit_point[0] + pert * tangent_vector[0], root[2]])

			d1   = p1 - np.array([crit_point[0], crit_point[2]])
			d2   = p2 - np.array([crit_point[0], crit_point[2]])
			d1   = d1/np.linalg.norm(d1)
			d2   = d1/np.linalg.norm(d2)

			if np.linalg.norm(d1-d2)<1e-3:
				# print("Directions are too close.")
				continue

			if root[0] > 1 or root[0] < 0 or root[1] > 1 or root[1] < 0 or root[2] > 1 or root[2] < 0:
				# print("Breaking out...")
				continue
			
			if (np.abs(mu_init(root))>1e-14).any():
				# print(f"Bad root: phi1 = ({root[0], root[1]}), phi2 = ({crit_point[0] + pert * tangent_vector[0], root[2]})...", flush=True)
				continue
			# elif np.linalg.norm(p1-p2) < 1e-3: 
			# 	print(f"Too close...")
			# 	continue
			else:
				# print(f"pert = {pert}")
				# print(f"Found root: phi1 = ({root[0], root[1]}), phi2 = ({root[2], crit_point[2] + pert * tangent_vector[1]})...", flush=True)
				if abs(pert*tangent_vector[0]) < 1e-20:
					continue
				good_root = True
				delta_ps2.append(pert*tangent_vector[0])
				phi_s1.append(root[0])
				phi_c1.append(root[1])
				phi_c2.append(root[2])
				phi_s2.append(crit_point[0]+pert*tangent_vector[0])

			# breaking out of pert loop
			if good_root:
				break

		# print(f"Broken out of pert loop!", flush=True)

		if not good_root:
			pass
			# print(f"No neighboring solution found...", flush=True)

		# print("I have found the initial root outside of the critical point. Time to go beyond.", flush=True)
		# print(f"phi_s1 = {phi_s1}", flush=True)
		# print(f"phi_p1 = {phi_c1}", flush=True)
		# print(f"phi_s2 = {phi_s2}", flush=True)
		# print(f"phi_p2 = {phi_c2}", flush=True)

		# exit() 

		return phi_s1, phi_s2, phi_c1, phi_c2, delta_ps2

	def calc_perts_in_s2(self, phi1, phi2, delta_phis2):

		a1 = np.float64(self.d_delta_mu_s_dps1(phi1[0], phi1[1], phi2[0], phi2[1]))
		a2 = np.float64(self.d_delta_mu_s_dpc1(phi1[0], phi1[1], phi2[0], phi2[1]))
		a3 = np.float64(self.d_delta_mu_s_dpc2(phi1[0], phi1[1], phi2[0], phi2[1]))
		a4 = np.float64(-delta_phis2 * self.d_delta_mu_s_dps2(phi1[0], phi1[1], phi2[0], phi2[1]))

		b1 = np.float64(self.d_delta_mu_p_dps1(phi1[0], phi1[1], phi2[0], phi2[1]))
		b2 = np.float64(self.d_delta_mu_p_dpc1(phi1[0], phi1[1], phi2[0], phi2[1]))
		b3 = np.float64(self.d_delta_mu_p_dpc2(phi1[0], phi1[1], phi2[0], phi2[1]))
		b4 = np.float64(-delta_phis2 * self.d_delta_mu_p_dps2(phi1[0], phi1[1], phi2[0], phi2[1]))

		c1 = np.float64(self.d_delta_mu_c_dps1(phi1[0], phi1[1], phi2[0], phi2[1]))
		c2 = np.float64(self.d_delta_mu_c_dpc1(phi1[0], phi1[1], phi2[0], phi2[1]))
		c3 = np.float64(self.d_delta_mu_c_dpc2(phi1[0], phi1[1], phi2[0], phi2[1]))
		c4 = np.float64(-delta_phis2 * self.d_delta_mu_c_dps2(phi1[0], phi1[1], phi2[0], phi2[1]))

		D  = np.float64(np.linalg.det(np.array([[a1,a2,a3],[b1,b2,b3],[c1,c2,c3]], dtype=np.float64)))
		Dx = np.float64(np.linalg.det(np.array([[a4,a2,a3],[b4,b2,b3],[c4,c2,c3]], dtype=np.float64)))
		Dy = np.float64(np.linalg.det(np.array([[a1,a4,a3],[b1,b4,b3],[c1,c4,c3]], dtype=np.float64)))
		Dz = np.float64(np.linalg.det(np.array([[a1,a2,a4],[b1,b2,b4],[c1,c2,c4]], dtype=np.float64)))

		# print(f"a1 = {a1}, a2 = {a2}, a3 = {a3}, a4 = {a4}")
		# print(f"b1 = {b1}, b2 = {b2}, b3 = {b3}, b4 = {b4}")
		# print(f"c1 = {c1}, c2 = {c2}, b3 = {c3}, b4 = {c4}")
		# print(f"D = {D}, Dx = {Dx}, Dy = {Dy}, Dz = {Dz}")
		# print(f"dx = {Dx/D}, dy = {Dy/D}, dz = {Dz/D}")

		return Dx/D, Dy/D, Dz/D

	def binodal_run_in_s2(self, crit_point, along_normal=False):
		phi_s1, phi_s2, phi_c1, phi_c2, delta_ps2 = self.find_solution_in_nbrhd_pert_in_s2(crit_point, along_normal)
		if len(phi_s1) == 1:
			return np.array(phi_s1), np.array(phi_s2), np.array(phi_c1), np.array(phi_c2)

		# print(f"delta_pc2 = {delta_ps2}", flush=True)
		condition = (phi_s1[-1] < 1e-12 or phi_c1[-1] < 1e-12 or 1-phi_s1[-1]-phi_c1[-1] < 1e-12) and (phi_s2[-1] < 1e-12 or phi_c2[-1] < 1e-12 or 1-phi_s2[-1]-phi_c2[-1] < 1e-12)

		# for i in range(ncycles):
		iterr  = 0 
		max_it = 1e+5

		while not condition:
			iterr += 1
			if iterr > max_it:
				# print("Completed iterations. Breaking out...", flush=True)
				break

			# print(f"@ i = {iterr}/{max_it}...", flush=True)

			phi1 = [phi_s1[-1], phi_c1[-1]]
			phi2 = [phi_s2[-1], phi_c2[-1]]
			delta_ps1, delta_pc1, delta_pc2 = self.calc_perts_in_s2(phi1, phi2, delta_ps2[-1])
			# print(f"delta_ps1 = {delta_ps1}, delta_pc1 = {delta_pc1}, delta_ps2 = {delta_ps2[-1]}, delta_pc2 = {delta_pc2}", flush=True)

			delta_1 = np.array([delta_ps1, delta_pc1])
			delta_2 = np.array([delta_ps2[-1], delta_pc2])

			delta_1 = delta_1/np.linalg.norm(delta_1)
			delta_2 = delta_2/np.linalg.norm(delta_2)

			if np.linalg.norm(delta_1 - delta_2) < 1e-3:
				# print("Moving in very similar directions.", flush=True)
				continue

			def dmu(phi_):
				eq1 = self.delta_mu_s(phi_[0], phi_[1], phi_s2[-1]+delta_ps2[-1], phi_[2])
				eq2 = self.delta_mu_p(phi_[0], phi_[1], phi_s2[-1]+delta_ps2[-1], phi_[2])
				eq3 = self.delta_mu_c(phi_[0], phi_[1], phi_s2[-1]+delta_ps2[-1], phi_[2])
				return [eq1, eq2, eq3]

			# print(f"Guess provided: phi1 = ({phi_s1[-1]+delta_ps1, phi_c1[-1]+delta_pc1}), phi2 = {phi_s2[-1]+delta_ps2[-1], phi_c2[-1]+delta_pc2}", flush=True)
			root = fsolve(dmu, [phi_s1[-1]+delta_ps1, phi_c1[-1]+delta_pc1, phi_c2[-1]+delta_pc2], xtol=1e-30)

			if root[0] > 1 or root[0] < 0 or root[1] > 1 or root[1] < 0 or root[2] > 1 or root[2] < 0 or root[0]+root[1]>1 or (root[2]+phi_s2[-1]+delta_ps2[-1])>1:
				delta_ps2.append(delta_ps2[-1]/1.1)
				continue
				# break

			p1 = np.array([root[0], root[1]])
			p2 = np.array([phi_s2[-1]+delta_ps2[-1], root[2]])

			p1_ = np.array([phi_s1[-1], phi_c1[-1]])
			p2_ = np.array([phi_s2[-1], phi_c2[-1]])

			d1 = p1 - p1_ 
			d2 = p2 - p2_ 

			p1__ = np.array([phi_s1[-2], phi_c1[-2]])
			p2__ = np.array([phi_s2[-2], phi_c2[-2]])

			d1_ = p1_ - p1__
			d2_ = p2_ - p2__


			# if np.linalg.norm(root[1] - phi_s2[-1]+delta_ps2[-1]) > 1e-6:
			# 	print(f"p1 = {p1}, p2 = {p2}")
			# 	print(f"p1_ = {p1_}, p2_ = {p2_}")
			# 	print(f"p1__ = {p1__}, p2__ = {p2__}")
			# 	print ("PROBLEM!")

			if (np.abs(dmu(root))>1e-14).any():
				# print(f"Bad root: phi1 = ({root[0], root[1]}), phi2 = ({phi_s2[-1]+delta_ps2[-1], root[2]})...", flush=True)
				delta_ps2.append(delta_ps2[-1]/1.1)
				continue
			elif np.linalg.norm(p1-p2) < 1e-6: 
				# print(f"Too close: phi1 = ({root[0], root[1]}), phi2 = ({phi_s2[-1]+delta_ps2[-1], root[2]})...", flush=True)
				delta_ps2.append(delta_ps2[-1]/1.1)
				continue
			# elif np.dot(d1/np.linalg.norm(d1), d1_/np.linalg.norm(d1_)) <= 0 or np.dot(d2/np.linalg.norm(d2), d2_/np.linalg.norm(d2_)) <= 0:
				# print(f"Making an about turn: phi1 = ({root[0], root[1]}), phi2 = ({root[2], phi_p2[-1]+delta_pp2[-1]})...", flush=True)
				# delta_pp2.append(delta_pp2[-1]/2)
				# continue


			# print(f"Found root: phi1 = ({root[0], root[1]}), phi2 = ({phi_s2[-1]+delta_ps2[-1], root[2]})...", flush=True)

			phi_s1.append(root[0])
			phi_c1.append(root[1])
			phi_c2.append(root[2])
			phi_s2.append(phi_s2[-1]+delta_ps2[-1])
			delta_ps2.append(delta_ps2[-1])
			condition = (phi_s1[-1] < 1e-12 or phi_c1[-1] < 1e-12 or 1-phi_s1[-1]-phi_c1[-1] < 1e-12)

		phi_s1 = np.array(phi_s1)
		phi_c1 = np.array(phi_c1)
		phi_s2 = np.array(phi_s2)
		phi_c2 = np.array(phi_c2)

		return phi_s1, phi_s2, phi_c1, phi_c2

# end of class sym_mu_sc

class sym_mu_pc:
	def __init__(self, inputs, spinodal):
		self.chi_sc   = inputs["chi_sc"]
		self.chi_ps   = inputs["chi_ps"]
		self.chi_pc   = inputs["chi_pc"]
		self.vs       = inputs["vs"]
		self.vc       = inputs["vc"]
		self.vp       = inputs["vp"]
		self.spinodal = spinodal
		self.setup()
		return

	def setup(self):

		phi_p1, phi_c1, phi_p2, phi_c2 = sym.symbols('phi_p1 phi_c1 phi_p2 phi_c2')

		mu_s1 = sym.log(1-phi_p1-phi_c1) + 1 - (1-phi_p1-phi_c1) - self.vs/self.vp * phi_p1            - self.vs/self.vc * phi_c1 + self.vs * (phi_p1**2            * self.chi_ps + phi_c1**2 * self.chi_sc + phi_p1            * phi_c1 * (self.chi_ps + self.chi_sc - self.chi_pc))
		mu_p1 = sym.log(phi_p1)          + 1 - phi_p1            - self.vp/self.vs * (1-phi_p1-phi_c1) - self.vp/self.vc * phi_c1 + self.vp * ((1-phi_p1-phi_c1)**2 * self.chi_ps + phi_c1**2 * self.chi_pc + (1-phi_p1-phi_c1) * phi_c1 * (self.chi_ps + self.chi_pc - self.chi_sc))
		mu_c1 = sym.log(phi_c1)          + 1 - phi_c1            - self.vc/self.vs * (1-phi_p1-phi_c1) - self.vc/self.vp * phi_p1 + self.vc * ((1-phi_p1-phi_c1)**2 * self.chi_sc + phi_p1**2 * self.chi_pc + (1-phi_p1-phi_c1) * phi_p1 * (self.chi_sc + self.chi_pc - self.chi_ps))

		mu_s2 = sym.log(1-phi_p2-phi_c2) + 1 - (1-phi_p2-phi_c2) - self.vs/self.vp * phi_p2            - self.vs/self.vc * phi_c2 + self.vs * (phi_p2**2            * self.chi_ps + phi_c2**2 * self.chi_sc + phi_p2            * phi_c2 * (self.chi_ps + self.chi_sc - self.chi_pc))
		mu_p2 = sym.log(phi_p2)          + 1 - phi_p2            - self.vp/self.vs * (1-phi_p2-phi_c2) - self.vp/self.vc * phi_c2 + self.vp * ((1-phi_p2-phi_c2)**2 * self.chi_ps + phi_c2**2 * self.chi_pc + (1-phi_p2-phi_c2) * phi_c2 * (self.chi_ps + self.chi_pc - self.chi_sc))
		mu_c2 = sym.log(phi_c2)          + 1 - phi_c2            - self.vc/self.vs * (1-phi_p2-phi_c2) - self.vc/self.vp * phi_p2 + self.vc * ((1-phi_p2-phi_c2)**2 * self.chi_sc + phi_p2**2 * self.chi_pc + (1-phi_p2-phi_c2) * phi_p2 * (self.chi_sc + self.chi_pc - self.chi_ps))

		delta_mu_s = mu_s1 - mu_s2
		delta_mu_p = mu_p1 - mu_p2
		delta_mu_c = mu_c1 - mu_c2

		# lambdify the function
		L_dmu_s = sym.lambdify([phi_p1, phi_c1, phi_p2, phi_c2], delta_mu_s)
		L_dmu_p = sym.lambdify([phi_p1, phi_c1, phi_p2, phi_c2], delta_mu_p)
		L_dmu_c = sym.lambdify([phi_p1, phi_c1, phi_p2, phi_c2], delta_mu_c)

		# do the differentiation of mu_s 
		d_delta_mu_s_dpp1 = sym.diff(delta_mu_s, phi_p1)
		d_delta_mu_s_dpp2 = sym.diff(delta_mu_s, phi_p2)
		d_delta_mu_s_dpc1 = sym.diff(delta_mu_s, phi_c1)
		d_delta_mu_s_dpc2 = sym.diff(delta_mu_s, phi_c2)

		# lambdify the function
		L_d_delta_mu_s_dpp1 = sym.lambdify([phi_p1, phi_c1, phi_p2, phi_c2], d_delta_mu_s_dpp1)
		L_d_delta_mu_s_dpp2 = sym.lambdify([phi_p1, phi_c1, phi_p2, phi_c2], d_delta_mu_s_dpp2)
		L_d_delta_mu_s_dpc1 = sym.lambdify([phi_p1, phi_c1, phi_p2, phi_c2], d_delta_mu_s_dpc1)
		L_d_delta_mu_s_dpc2 = sym.lambdify([phi_p1, phi_c1, phi_p2, phi_c2], d_delta_mu_s_dpc2)

		# do the differentiation of mu_p
		d_delta_mu_p_dpp1 = sym.diff(delta_mu_p, phi_p1)
		d_delta_mu_p_dpp2 = sym.diff(delta_mu_p, phi_p2)
		d_delta_mu_p_dpc1 = sym.diff(delta_mu_p, phi_c1)
		d_delta_mu_p_dpc2 = sym.diff(delta_mu_p, phi_c2)

		# lambdify the function 
		L_d_delta_mu_p_dpp1 = sym.lambdify([phi_p1, phi_c1, phi_p2, phi_c2], d_delta_mu_p_dpp1)
		L_d_delta_mu_p_dpp2 = sym.lambdify([phi_p1, phi_c1, phi_p2, phi_c2], d_delta_mu_p_dpp2)
		L_d_delta_mu_p_dpc1 = sym.lambdify([phi_p1, phi_c1, phi_p2, phi_c2], d_delta_mu_p_dpc1)
		L_d_delta_mu_p_dpc2 = sym.lambdify([phi_p1, phi_c1, phi_p2, phi_c2], d_delta_mu_p_dpc2)

		# do the differentiation of mu_c
		d_delta_mu_c_dpp1   = sym.diff(delta_mu_c, phi_p1)
		d_delta_mu_c_dpp2   = sym.diff(delta_mu_c, phi_p2)
		d_delta_mu_c_dpc1   = sym.diff(delta_mu_c, phi_c1)
		d_delta_mu_c_dpc2   = sym.diff(delta_mu_c, phi_c2)

		# lambdify the function 
		L_d_delta_mu_c_dpp1 = sym.lambdify([phi_p1, phi_c1, phi_p2, phi_c2], d_delta_mu_c_dpp1)
		L_d_delta_mu_c_dpp2 = sym.lambdify([phi_p1, phi_c1, phi_p2, phi_c2], d_delta_mu_c_dpp2)
		L_d_delta_mu_c_dpc1 = sym.lambdify([phi_p1, phi_c1, phi_p2, phi_c2], d_delta_mu_c_dpc1)
		L_d_delta_mu_c_dpc2 = sym.lambdify([phi_p1, phi_c1, phi_p2, phi_c2], d_delta_mu_c_dpc2)

		# get the new function
		self.delta_mu_s = lambda p1, c1, p2, c2: L_dmu_s(p1, c1, p2, c2)
		self.delta_mu_p = lambda p1, c1, p2, c2: L_dmu_p(p1, c1, p2, c2)
		self.delta_mu_c = lambda p1, c1, p2, c2: L_dmu_c(p1, c1, p2, c2)

		# get the new function
		self.d_delta_mu_s_dpp1 = lambda p1, c1, p2, c2: L_d_delta_mu_s_dpp1(p1, c1, p2, c2)
		self.d_delta_mu_s_dpc1 = lambda p1, c1, p2, c2: L_d_delta_mu_s_dpc1(p1, c1, p2, c2)
		self.d_delta_mu_s_dpp2 = lambda p1, c1, p2, c2: L_d_delta_mu_s_dpp2(p1, c1, p2, c2)
		self.d_delta_mu_s_dpc2 = lambda p1, c1, p2, c2: L_d_delta_mu_s_dpc2(p1, c1, p2, c2)

		# get the new function
		self.d_delta_mu_p_dpp1 = lambda p1, c1, p2, c2: L_d_delta_mu_p_dpp1(p1, c1, p2, c2)
		self.d_delta_mu_p_dpc1 = lambda p1, c1, p2, c2: L_d_delta_mu_p_dpc1(p1, c1, p2, c2)
		self.d_delta_mu_p_dpp2 = lambda p1, c1, p2, c2: L_d_delta_mu_p_dpp2(p1, c1, p2, c2)
		self.d_delta_mu_p_dpc2 = lambda p1, c1, p2, c2: L_d_delta_mu_p_dpc2(p1, c1, p2, c2)

		# get the new function
		self.d_delta_mu_c_dpp1 = lambda p1, c1, p2, c2: L_d_delta_mu_c_dpp1(p1, c1, p2, c2)
		self.d_delta_mu_c_dpc1 = lambda p1, c1, p2, c2: L_d_delta_mu_c_dpc1(p1, c1, p2, c2)
		self.d_delta_mu_c_dpp2 = lambda p1, c1, p2, c2: L_d_delta_mu_c_dpp2(p1, c1, p2, c2)
		self.d_delta_mu_c_dpc2 = lambda p1, c1, p2, c2: L_d_delta_mu_c_dpc2(p1, c1, p2, c2)

		return

	def find_solution_in_nbrhd_pert_in_c2(self, crit_point, along_normal=False, loglims=[-3,-10,100]):

		# print(f"crit point of interest = {crit_point}", flush=True)
		delta_pc2 = []
		phi_p1    = [crit_point[1]]
		phi_p2    = [crit_point[1]]
		phi_c1    = [crit_point[2]]
		phi_c2    = [crit_point[2]]

		good_root = False

		slope          = -tangent.tangent2(self.vs, self.vc, self.vp, crit_point[0], crit_point[1], self.chi_pc, self.chi_ps, self.chi_sc, self.spinodal.root_up_s, self.spinodal.root_lo_s)
		if np.isinf(slope) or np.isnan(slope):
			tangent_vector = np.array([1,0])
		else:
			tangent_vector = np.array([slope, 1])/np.sqrt(1+(slope)**2)
		if along_normal:
			if np.isinf(slope) or np.isnan(slope) or abs(slope) < 1e-6:
				tangent_vector = np.array([1,0])
			else:
				slope = -1/slope
				tangent_vector = np.array([slope, 1])/np.sqrt(1+(slope)**2)
		
		print(f"Tangent vector = {tangent_vector}", flush=True)
		pert_scale = np.logspace(loglims[0], loglims[1], loglims[2])

		# print(pert_scale)

		for pert in pert_scale:
			def mu_init(phi_):
				eq1 = self.delta_mu_s(phi_[0], phi_[1], phi_[2], crit_point[2] + pert * tangent_vector[1])
				eq2 = self.delta_mu_p(phi_[0], phi_[1], phi_[2], crit_point[2] + pert * tangent_vector[1])
				eq3 = self.delta_mu_c(phi_[0], phi_[1], phi_[2], crit_point[2] + pert * tangent_vector[1])
				return [eq1, eq2, eq3]

			root = fsolve(mu_init, [crit_point[1] - pert * tangent_vector[0], crit_point[2] - pert * tangent_vector[1], crit_point[0] + pert * tangent_vector[0]], xtol=1e-16)
			p1   = np.array([root[0], root[1]])
			p2   = np.array([root[2], crit_point[2] + pert * tangent_vector[1]])

			d1   = p1 - np.array([crit_point[1], crit_point[2]])
			d2   = p2 - np.array([crit_point[1], crit_point[2]])
			d1   = d1/np.linalg.norm(d1)
			d2   = d1/np.linalg.norm(d2)

			if np.linalg.norm(d1-d2)<1e-3:
				# print("Directions are too close.")
				continue


			if root[0] > 1 or root[0] < 0 or root[1] > 1 or root[1] < 0 or root[2] > 1 or root[2] < 0:
				# print("Breaking out...")
				continue
			
			if (np.abs(mu_init(root))>1e-14).any():
				# print(f"Bad root: phi1 = ({root[0], root[1]}), phi2 = ({root[2], crit_point[2] + pert * tangent_vector[1]})...", flush=True)
				continue
			# elif np.linalg.norm(p1-p2) < 1e-3: 
			# 	print(f"Too close...")
			# 	continue
			else:
				# print(f"pert = {pert}")
				# print(f"Found root: phi1 = ({root[0], root[1]}), phi2 = ({root[2], crit_point[2] + pert * tangent_vector[1]})...", flush=True)
				if abs(pert*tangent_vector[1]) < 1e-20:
					continue
				good_root = True
				delta_pc2.append(pert*tangent_vector[1])
				phi_p1.append(root[0])
				phi_c1.append(root[1])
				phi_p2.append(root[2])
				phi_c2.append(crit_point[2]+pert*tangent_vector[1])

			# breaking out of pert loop
			if good_root:
				break

		# print(f"Broken out of pert loop!", flush=True)

		if not good_root:
			pass
			# print(f"No neighboring solution found...", flush=True)

		# print("I have found the initial root outside of the critical point. Time to go beyond.", flush=True)
		# print(f"phi_p1 = {phi_p1}", flush=True)
		# print(f"phi_c1 = {phi_c1}", flush=True)
		# print(f"phi_p2 = {phi_p2}", flush=True)
		# print(f"phi_c2 = {phi_c2}", flush=True)

		return phi_p1, phi_p2, phi_c1, phi_c2, delta_pc2

	def calc_perts_in_c2(self, phi1, phi2, delta_phic2):

		a1 = np.float64(self.d_delta_mu_s_dpp1(phi1[0], phi1[1], phi2[0], phi2[1]))
		a2 = np.float64(self.d_delta_mu_s_dpc1(phi1[0], phi1[1], phi2[0], phi2[1]))
		a3 = np.float64(self.d_delta_mu_s_dpp2(phi1[0], phi1[1], phi2[0], phi2[1]))
		a4 = np.float64(-delta_phic2 * self.d_delta_mu_s_dpc2(phi1[0], phi1[1], phi2[0], phi2[1]))

		b1 = np.float64(self.d_delta_mu_p_dpp1(phi1[0], phi1[1], phi2[0], phi2[1]))
		b2 = np.float64(self.d_delta_mu_p_dpc1(phi1[0], phi1[1], phi2[0], phi2[1]))
		b3 = np.float64(self.d_delta_mu_p_dpp2(phi1[0], phi1[1], phi2[0], phi2[1]))
		b4 = np.float64(-delta_phic2 * self.d_delta_mu_p_dpc2(phi1[0], phi1[1], phi2[0], phi2[1]))

		c1 = np.float64(self.d_delta_mu_c_dpp1(phi1[0], phi1[1], phi2[0], phi2[1]))
		c2 = np.float64(self.d_delta_mu_c_dpc1(phi1[0], phi1[1], phi2[0], phi2[1]))
		c3 = np.float64(self.d_delta_mu_c_dpp2(phi1[0], phi1[1], phi2[0], phi2[1]))
		c4 = np.float64(-delta_phic2 * self.d_delta_mu_c_dpc2(phi1[0], phi1[1], phi2[0], phi2[1]))

		D  = np.float64(np.linalg.det(np.array([[a1,a2,a3],[b1,b2,b3],[c1,c2,c3]], dtype=np.float64)))
		Dx = np.float64(np.linalg.det(np.array([[a4,a2,a3],[b4,b2,b3],[c4,c2,c3]], dtype=np.float64)))
		Dy = np.float64(np.linalg.det(np.array([[a1,a4,a3],[b1,b4,b3],[c1,c4,c3]], dtype=np.float64)))
		Dz = np.float64(np.linalg.det(np.array([[a1,a2,a4],[b1,b2,b4],[c1,c2,c4]], dtype=np.float64)))

		# print(f"a1 = {a1}, a2 = {a2}, a3 = {a3}, a4 = {a4}")
		# print(f"b1 = {b1}, b2 = {b2}, b3 = {b3}, b4 = {b4}")
		# print(f"c1 = {c1}, c2 = {c2}, b3 = {c3}, b4 = {c4}")
		# print(f"D = {D}, Dx = {Dx}, Dy = {Dy}, Dz = {Dz}")
		# print(f"dx = {Dx/D}, dy = {Dy/D}, dz = {Dz/D}")

		return Dx/D, Dy/D, Dz/D

	def binodal_run_in_c2(self, crit_point, along_normal=False):
		phi_p1, phi_p2, phi_c1, phi_c2, delta_pc2 = self.find_solution_in_nbrhd_pert_in_c2(crit_point, along_normal)
		if len(phi_p1) == 1:
			return np.array(phi_p1), np.array(phi_p2), np.array(phi_c1), np.array(phi_c2)

		# print(f"delta_pc2 = {delta_pc2}", flush=True)
		condition = (phi_p1[-1] < 1e-12 or phi_c1[-1] < 1e-12 or 1-phi_p1[-1]-phi_c1[-1] < 1e-12) and (phi_p2[-1] < 1e-12 or phi_c2[-1] < 1e-12 or 1-phi_p2[-1]-phi_c2[-1] < 1e-12)

		# for i in range(ncycles):
		iterr  = 0 
		max_it = 1e+5

		while not condition:
			iterr += 1
			if iterr > max_it:
				# print("Completed iterations. Breaking out...", flush=True)
				break
			# preprocess...
			# preprocessor = np.array(delta_pp2[-100:])
			# if (np.abs(preprocessor) < 1e-16).all():
			# 	pass
			# delta_pp2.append(1e-6)

			# print(f"@ i = {iterr}/{max_it}...", flush=True)

			phi1 = [phi_p1[-1], phi_c1[-1]]
			phi2 = [phi_p2[-1], phi_c2[-1]]
			delta_pp1, delta_pc1, delta_pp2 = self.calc_perts_in_c2(phi1, phi2, delta_pc2[-1])
			# print(f"delta_pp1 = {delta_pp1}, delta_pc1 = {delta_pc1}, delta_pp2 = {delta_pp2}, delta_pc2 = {delta_pc2[-1]}", flush=True)

			delta_1 = np.array([delta_pp1, delta_pc1])
			delta_2 = np.array([delta_pp2, delta_pc2[-1]])

			delta_1 = delta_1/np.linalg.norm(delta_1)
			delta_2 = delta_2/np.linalg.norm(delta_2)

			if np.linalg.norm(delta_1 - delta_2) < 1e-3:
				# print("Moving in very similar directions.", flush=True)
				continue

			def dmu(phi_):
				eq1 = self.delta_mu_s(phi_[0], phi_[1], phi_[2], phi_c2[-1]+delta_pc2[-1])
				eq2 = self.delta_mu_p(phi_[0], phi_[1], phi_[2], phi_c2[-1]+delta_pc2[-1])
				eq3 = self.delta_mu_c(phi_[0], phi_[1], phi_[2], phi_c2[-1]+delta_pc2[-1])
				return [eq1, eq2, eq3]

			# print(f"Guess provided: phi1 = ({phi_p1[-1]+delta_pp1, phi_c1[-1]+delta_pc1}), phi2 = {phi_p2[-1]+delta_pp2, phi_c2[-1]+delta_pc2[-1]}", flush=True)
			root = fsolve(dmu, [phi_p1[-1]+delta_pp1, phi_c1[-1]+delta_pc1, phi_p2[-1]+delta_pp2], xtol=1e-30)

			if root[0] > 1 or root[0] < 0 or root[1] > 1 or root[1] < 0 or root[2] > 1 or root[2] < 0 or (root[0]+root[1])>1 or root[2]+phi_c2[-1]+delta_pc2[-1]>1:
				delta_pc2.append(delta_pc2[-1]/1.1)
				continue

			p1 = np.array([root[0], root[1]])
			p2 = np.array([root[2], phi_c2[-1]+delta_pc2[-1]])

			p1_ = np.array([phi_p1[-1], phi_c1[-1]])
			p2_ = np.array([phi_p2[-1], phi_c2[-1]])

			d1 = p1 - p1_ 
			d2 = p2 - p2_ 

			p1__ = np.array([phi_p1[-2], phi_c1[-2]])
			p2__ = np.array([phi_p2[-2], phi_c2[-2]])

			d1_ = p1_ - p1__
			d2_ = p2_ - p2__


			# if np.linalg.norm(root[1] - phi_c2[-1]+delta_pc2[-1]) > 1e-6:
			# 	print(f"p1 = {p1}, p2 = {p2}")
			# 	print(f"p1_ = {p1_}, p2_ = {p2_}")
			# 	print(f"p1__ = {p1__}, p2__ = {p2__}")
			# 	print ("PROBLEM!")

			if (np.abs(dmu(root))>1e-14).any():
				# print(f"Bad root: phi1 = ({root[0], root[1]}), phi2 = ({root[2], phi_c2[-1]+delta_pc2[-1]})...", flush=True)
				delta_pc2.append(delta_pc2[-1]/1.1)
				continue
			elif np.linalg.norm(p1-p2) < 1e-6: 
				# print(f"Too close: phi1 = ({root[0], root[1]}), phi2 = ({root[2], phi_c2[-1]+delta_pc2[-1]})...", flush=True)
				delta_pc2.append(delta_pc2[-1]/1.1)
				continue
			# elif np.dot(d1/np.linalg.norm(d1), d1_/np.linalg.norm(d1_)) <= 0 or np.dot(d2/np.linalg.norm(d2), d2_/np.linalg.norm(d2_)) <= 0:
				# print(f"Making an about turn: phi1 = ({root[0], root[1]}), phi2 = ({root[2], phi_p2[-1]+delta_pp2[-1]})...", flush=True)
				# delta_pp2.append(delta_pp2[-1]/2)
				# continue


			# print(f"Found root: phi1 = ({root[0], root[1]}), phi2 = ({root[2], phi_c2[-1]+delta_pc2[-1]})...", flush=True)

			phi_p1.append(root[0])
			phi_c1.append(root[1])
			phi_p2.append(root[2])
			phi_c2.append(phi_c2[-1]+delta_pc2[-1])
			delta_pc2.append(delta_pc2[-1])
			condition = (phi_p1[-1] < 1e-12 or phi_c1[-1] < 1e-12 or 1-phi_p1[-1]-phi_c1[-1] < 1e-12)

		phi_p1 = np.array(phi_p1)
		phi_c1 = np.array(phi_c1)
		phi_p2 = np.array(phi_p2)
		phi_c2 = np.array(phi_c2)

		return phi_p1, phi_p2, phi_c1, phi_c2

	def find_solution_in_nbrhd_pert_in_p2(self, crit_point, along_normal=False, loglims=[-3,-10,100]):

		# print(f"crit point of interest = {crit_point}", flush=True)
		delta_pp2 = []
		phi_p1    = [crit_point[1]]
		phi_p2    = [crit_point[1]]
		phi_c1    = [crit_point[2]]
		phi_c2    = [crit_point[2]]

		good_root = False

		slope          = -tangent.tangent2(self.vs, self.vc, self.vp, crit_point[0], crit_point[1], self.chi_pc, self.chi_ps, self.chi_sc, self.spinodal.root_up_s, self.spinodal.root_lo_s)

		if np.isinf(slope) or np.isnan(slope):
			tangent_vector = np.array([1,0])
		else:
			tangent_vector = np.array([slope, 1])/np.sqrt(1+(slope)**2)
		if along_normal:
			if np.isinf(slope) or np.isnan(slope) or abs(slope) < 1e-6:
				tangent_vector = np.array([1,0])
			else:
				slope = -1/slope
				tangent_vector = np.array([slope, 1])/np.sqrt(1+(slope)**2)

		print(f"Tangent vector = {tangent_vector}", flush=True)
		pert_scale = np.logspace(loglims[0], loglims[1], loglims[2])

		# print(pert_scale)

		for pert in pert_scale:
			def mu_init(phi_):
				eq1 = self.delta_mu_s(phi_[0], phi_[1], crit_point[1] + pert * tangent_vector[0], phi_[2])
				eq2 = self.delta_mu_p(phi_[0], phi_[1], crit_point[1] + pert * tangent_vector[0], phi_[2])
				eq3 = self.delta_mu_c(phi_[0], phi_[1], crit_point[1] + pert * tangent_vector[0], phi_[2])
				return [eq1, eq2, eq3]

			root = fsolve(mu_init, [crit_point[1] - pert * tangent_vector[0], crit_point[2] - pert * tangent_vector[1], crit_point[2] + pert * tangent_vector[1]], xtol=1e-16)
			p1   = np.array([root[0], root[1]])
			p2   = np.array([crit_point[1] + pert * tangent_vector[0], root[2]])

			d1   = p1 - np.array([crit_point[1], crit_point[2]])
			d2   = p2 - np.array([crit_point[1], crit_point[2]])
			d1   = d1/np.linalg.norm(d1)
			d2   = d1/np.linalg.norm(d2)

			if np.linalg.norm(d1-d2)<1e-3:
				# print("Directions are too close.")
				continue

			if root[0] > 1 or root[0] < 0 or root[1] > 1 or root[1] < 0 or root[2] > 1 or root[2] < 0:
				# print("Breaking out...")
				continue
			
			if (np.abs(mu_init(root))>1e-14).any():
				# print(f"Bad root: phi1 = ({root[0], root[1]}), phi2 = ({crit_point[1] + pert * tangent_vector[0], root[2]})...", flush=True)
				continue
			# elif np.linalg.norm(p1-p2) < 1e-3: 
			# 	print(f"Too close...")
			# 	continue
			else:
				# print(f"pert = {pert}")
				# print(f"Found root: phi1 = ({root[0], root[1]}), phi2 = ({crit_point[1] + pert * tangent_vector[0], root[2]})...", flush=True)
				if abs(pert*tangent_vector[0]) < 1e-20:
					continue
				good_root = True
				delta_pp2.append(pert*tangent_vector[0])
				phi_p1.append(root[0])
				phi_c1.append(root[1])
				phi_c2.append(root[2])
				phi_p2.append(crit_point[1]+pert*tangent_vector[0])

			# breaking out of pert loop
			if good_root:
				break

		# print(f"Broken out of pert loop!", flush=True)

		if not good_root:
			pass
			# print(f"No neighboring solution found...", flush=True)

		# print("I have found the initial root outside of the critical point. Time to go beyond.", flush=True)
		# print(f"phi_p1 = {phi_p1}", flush=True)
		# print(f"phi_c1 = {phi_c1}", flush=True)
		# print(f"phi_p2 = {phi_p2}", flush=True)
		# print(f"phi_c2 = {phi_c2}", flush=True)

		return phi_p1, phi_p2, phi_c1, phi_c2, delta_pp2

	def calc_perts_in_p2(self, phi1, phi2, delta_phip2):

		a1 = np.float64(self.d_delta_mu_s_dpp1(phi1[0], phi1[1], phi2[0], phi2[1]))
		a2 = np.float64(self.d_delta_mu_s_dpc1(phi1[0], phi1[1], phi2[0], phi2[1]))
		a3 = np.float64(self.d_delta_mu_s_dpc2(phi1[0], phi1[1], phi2[0], phi2[1]))
		a4 = np.float64(-delta_phip2 * self.d_delta_mu_s_dpp2(phi1[0], phi1[1], phi2[0], phi2[1]))

		b1 = np.float64(self.d_delta_mu_p_dpp1(phi1[0], phi1[1], phi2[0], phi2[1]))
		b2 = np.float64(self.d_delta_mu_p_dpc1(phi1[0], phi1[1], phi2[0], phi2[1]))
		b3 = np.float64(self.d_delta_mu_p_dpc2(phi1[0], phi1[1], phi2[0], phi2[1]))
		b4 = np.float64(-delta_phip2 * self.d_delta_mu_p_dpp2(phi1[0], phi1[1], phi2[0], phi2[1]))

		c1 = np.float64(self.d_delta_mu_c_dpp1(phi1[0], phi1[1], phi2[0], phi2[1]))
		c2 = np.float64(self.d_delta_mu_c_dpc1(phi1[0], phi1[1], phi2[0], phi2[1]))
		c3 = np.float64(self.d_delta_mu_c_dpc2(phi1[0], phi1[1], phi2[0], phi2[1]))
		c4 = np.float64(-delta_phip2 * self.d_delta_mu_c_dpp2(phi1[0], phi1[1], phi2[0], phi2[1]))

		D  = np.float64(np.linalg.det(np.array([[a1,a2,a3],[b1,b2,b3],[c1,c2,c3]], dtype=np.float64)))
		Dx = np.float64(np.linalg.det(np.array([[a4,a2,a3],[b4,b2,b3],[c4,c2,c3]], dtype=np.float64)))
		Dy = np.float64(np.linalg.det(np.array([[a1,a4,a3],[b1,b4,b3],[c1,c4,c3]], dtype=np.float64)))
		Dz = np.float64(np.linalg.det(np.array([[a1,a2,a4],[b1,b2,b4],[c1,c2,c4]], dtype=np.float64)))

		# print(f"a1 = {a1}, a2 = {a2}, a3 = {a3}, a4 = {a4}")
		# print(f"b1 = {b1}, b2 = {b2}, b3 = {b3}, b4 = {b4}")
		# print(f"c1 = {c1}, c2 = {c2}, b3 = {c3}, b4 = {c4}")
		# print(f"D = {D}, Dx = {Dx}, Dy = {Dy}, Dz = {Dz}")
		# print(f"dx = {Dx/D}, dy = {Dy/D}, dz = {Dz/D}")

		return Dx/D, Dy/D, Dz/D

	def binodal_run_in_p2(self, crit_point, along_normal=False):
		phi_p1, phi_p2, phi_c1, phi_c2, delta_pp2 = self.find_solution_in_nbrhd_pert_in_p2(crit_point)
		if len(phi_p1) == 1:
			return np.array(phi_p1), np.array(phi_p2), np.array(phi_c1), np.array(phi_c2)

		# print(f"delta_pp2 = {delta_pp2}", flush=True)
		condition = (phi_p1[-1] < 1e-12 or phi_c1[-1] < 1e-12 or 1-phi_p1[-1]-phi_c1[-1] < 1e-12) and (phi_p2[-1] < 1e-12 or phi_c2[-1] < 1e-12 or 1-phi_p2[-1]-phi_c2[-1] < 1e-12)

		# for i in range(ncycles):
		iterr  = 0 
		max_it = 1e+5

		while not condition:
			iterr += 1
			if iterr > max_it:
				 # print("Completed iterations. Breaking out...", flush=True)
				break

			# print(f"@ i = {iterr}/{max_it}...", flush=True)

			phi1 = [phi_p1[-1], phi_c1[-1]]
			phi2 = [phi_p2[-1], phi_c2[-1]]
			delta_pp1, delta_pc1, delta_pc2 = self.calc_perts_in_p2(phi1, phi2, delta_pp2[-1])
			# print(f"delta_pp1 = {delta_pp1}, delta_pc1 = {delta_pc1}, delta_pp2 = {delta_pp2[-1]}, delta_pc2 = {delta_pc2}", flush=True)

			delta_1 = np.array([delta_pp1, delta_pc1])
			delta_2 = np.array([delta_pp2[-1], delta_pc2])

			delta_1 = delta_1/np.linalg.norm(delta_1)
			delta_2 = delta_2/np.linalg.norm(delta_2)

			if np.linalg.norm(delta_1 - delta_2) < 1e-3:
				# print("Moving in very similar directions.", flush=True)
				continue

			def dmu(phi_):
				eq1 = self.delta_mu_s(phi_[0], phi_[1], phi_p2[-1]+delta_pp2[-1], phi_[2])
				eq2 = self.delta_mu_p(phi_[0], phi_[1], phi_p2[-1]+delta_pp2[-1], phi_[2])
				eq3 = self.delta_mu_c(phi_[0], phi_[1], phi_p2[-1]+delta_pp2[-1], phi_[2])
				return [eq1, eq2, eq3]

			# print(f"Guess provided: phi1 = ({phi_p1[-1]+delta_pp1, phi_c1[-1]+delta_pc1}), phi2 = {phi_p2[-1]+delta_pp2[-1], phi_c2[-1]+delta_pc2}", flush=True)
			root = fsolve(dmu, [phi_p1[-1]+delta_pp1, phi_c1[-1]+delta_pc1, phi_c2[-1]+delta_pc2], xtol=1e-30)

			if root[0] > 1 or root[0] < 0 or root[1] > 1 or root[1] < 0 or root[2] > 1 or root[2] < 0 or root[0]+root[1]>1 or root[2]+phi_p2[-1]+delta_pp2[-1]>1:
				delta_pp2.append(delta_pp2[-1]/1.1)
				continue

			p1 = np.array([root[0], root[1]])
			p2 = np.array([phi_p2[-1]+delta_pp2[-1], root[2]])

			p1_ = np.array([phi_p1[-1], phi_c1[-1]])
			p2_ = np.array([phi_p2[-1], phi_c2[-1]])

			d1 = p1 - p1_ 
			d2 = p2 - p2_ 

			p1__ = np.array([phi_p1[-2], phi_c1[-2]])
			p2__ = np.array([phi_p2[-2], phi_c2[-2]])

			d1_ = p1_ - p1__
			d2_ = p2_ - p2__


			# if np.linalg.norm(root[1] - phi_c2[-1]+delta_pc2[-1]) > 1e-6:
			# 	print(f"p1 = {p1}, p2 = {p2}")
			# 	print(f"p1_ = {p1_}, p2_ = {p2_}")
			# 	print(f"p1__ = {p1__}, p2__ = {p2__}")
			# 	print ("PROBLEM!")

			if (np.abs(dmu(root))>1e-14).any():
				# print(f"Bad root: phi1 = ({root[0], root[1]}), phi2 = ({phi_p2[-1]+delta_pp2[-1], root[2]})...", flush=True)
				delta_pp2.append(delta_pp2[-1]/1.1)
				continue
			elif np.linalg.norm(p1-p2) < 1e-6: 
				# print(f"Too close: phi1 = ({root[0], root[1]}), phi2 = ({phi_p2[-1]+delta_pp2[-1], root[2]})...", flush=True)
				delta_pp2.append(delta_pp2[-1]/1.1)
				continue
			# elif np.dot(d1/np.linalg.norm(d1), d1_/np.linalg.norm(d1_)) <= 0 or np.dot(d2/np.linalg.norm(d2), d2_/np.linalg.norm(d2_)) <= 0:
				# print(f"Making an about turn: phi1 = ({root[0], root[1]}), phi2 = ({root[2], phi_p2[-1]+delta_pp2[-1]})...", flush=True)
				# delta_pp2.append(delta_pp2[-1]/2)
				# continue


			# print(f"Found root: phi1 = ({root[0], root[1]}), phi2 = ({phi_p2[-1]+delta_pp2[-1], root[2]})...", flush=True)

			phi_p1.append(root[0])
			phi_c1.append(root[1])
			phi_c2.append(root[2])
			phi_p2.append(phi_p2[-1]+delta_pp2[-1])
			delta_pp2.append(delta_pp2[-1])
			condition = (phi_p1[-1] < 1e-12 or phi_c1[-1] < 1e-12 or 1-phi_p1[-1]-phi_c1[-1] < 1e-12)

		phi_p1 = np.array(phi_p1)
		phi_c1 = np.array(phi_c1)
		phi_p2 = np.array(phi_p2)
		phi_c2 = np.array(phi_c2)

		return phi_p1, phi_p2, phi_c1, phi_c2

# end of class sym_mu_pc
