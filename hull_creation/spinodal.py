import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import tangent
import ternary 

class Spinodal:
	def __init__(self, inputs, crits=None):
		self.chi_sc = inputs["chi_sc"]
		self.chi_ps = inputs["chi_ps"]
		self.chi_pc = inputs["chi_pc"]
		self.vs     = inputs["vs"]
		self.vc     = inputs["vc"]
		self.vp     = inputs["vp"]
		self.crits  = crits
		return

	# edges of the spinodal in terms of phi_s (arbitrary choice!)
	def discriminant_s(self, phi_s):
		d = -4*self.vc*self.vp*(2*self.chi_pc + phi_s*self.vs*self.chi_pc**2 + \
			phi_s*self.vs*(self.chi_ps-self.chi_sc)**2 - 2*phi_s*\
			self.vs*self.chi_pc*(self.chi_ps+self.chi_sc))*(phi_s*self.vs+\
			(-1+phi_s)*self.vc*(-1+2*phi_s*self.vs*self.chi_sc))+(self.vp - \
			2*phi_s*self.vp*self.vs*self.chi_ps + \
			self.vc*(-1+2*phi_s*self.vs*self.chi_sc+(-1+phi_s)*self.vp*\
			(2*self.chi_pc+phi_s*self.vs*self.chi_pc**2 +phi_s*self.vs*(self.chi_ps-self.chi_sc)**2 \
			- 2*phi_s*self.vs*self.chi_pc*(self.chi_ps+self.chi_sc))))**2
		return d

	def denom_s(self, phi_s):
		return 1/(-2*self.vc*self.vp*(2*self.chi_pc+phi_s*self.vs*self.chi_pc**2+phi_s*self.vs*(self.chi_ps-self.chi_sc)**2 - 2*phi_s*self.vs*self.chi_pc*(self.chi_ps+self.chi_sc)))

	def prefac_s(self, phi_s):
		return self.vp - 2*phi_s*self.vp*self.vs*self.chi_ps+self.vc * (-1+2*phi_s*self.vs*self.chi_sc + (-1+phi_s) * self.vp *\
	(2*self.chi_pc + phi_s*self.vs*self.chi_pc**2 + phi_s * self.vs * (self.chi_ps - self.chi_sc) **2 - 2 * phi_s * self.vs * self.chi_pc *(self.chi_ps + self.chi_sc)))

	def root_up_s(self, phi_s):
		return self.denom_s(phi_s)*(self.prefac_s(phi_s) + np.sqrt(self.discriminant_s(phi_s)))

	def root_lo_s(self, phi_s):
		return self.denom_s(phi_s)*(self.prefac_s(phi_s) - np.sqrt(self.discriminant_s(phi_s)))

	# edges of the spinodal in terms of phi_s (arbitrary choice!)
	# solution in terms of phi_p
	def discriminant_p(self,phi_p): 
		return -4*self.vc*self.vs*(phi_p*self.vp+(-1+phi_p)*self.vc*(-1+2*phi_p*self.vp*self.chi_pc))*(2*self.chi_sc+phi_p*self.vp*\
	(self.chi_pc**2+(self.chi_ps-self.chi_sc)**2-2*self.chi_pc*(self.chi_ps+self.chi_sc)))+(self.vs-2*phi_p*self.vp*self.vs*self.chi_ps+self.vc*(-1-2*self.vs*self.chi_sc+phi_p**2*\
	self.vp*self.vs*(self.chi_pc**2+(self.chi_ps-self.chi_sc)**2-2*self.chi_pc*(self.chi_ps+self.chi_sc))+phi_p*(2*self.vs*self.chi_sc-self.vp*(self.vs*self.chi_pc**2+self.vs*(self.chi_ps-self.chi_sc)**2\
	-2*self.chi_pc*(1+self.vs*(self.chi_ps+self.chi_sc))))))**2

	def denom_p(self,phi_p): 
		return 1/(-2*self.vc*self.vs*(2*self.chi_sc+phi_p*self.vp*(self.chi_pc**2+(self.chi_ps-self.chi_sc)**2-2*self.chi_pc*(self.chi_ps+self.chi_sc))))
		
	def prefac_p(self, phi_p):
		return self.vs-2*phi_p*self.vp*self.vs*self.chi_ps+self.vc*(-1-2*self.vs*self.chi_sc+phi_p**2*self.vp*self.vs*(self.chi_pc**2+(self.chi_ps-self.chi_sc)**2-\
	2*self.chi_pc*(self.chi_ps+self.chi_sc))+phi_p*(2*self.vs*self.chi_sc-self.vp*(self.vs*self.chi_pc**2+self.vs*(self.chi_ps-self.chi_sc)**2-2*self.chi_pc*(1+self.vs*(self.chi_ps+self.chi_sc)))))

	def root_up_p(self, phi_p):
		r = self.denom_p(phi_p)*(self.prefac_p(phi_p)+np.sqrt(self.discriminant_p(phi_p)))
		return r
	
	def root_lo_p(self, phi_p): 
		r = self.denom_p(phi_p)*(self.prefac_p(phi_p)-np.sqrt(self.discriminant_p(phi_p)))
		return r

	# obtain critical points
	def obtain_crits(self):
		roots_up, roots_down = ternary.find_crit_point(self.vs, self.vc, self.vp, self.chi_sc, self.chi_ps, self.chi_pc, self.root_up_p, self.root_up_s, self.root_lo_p, self.root_lo_s)
		crits      = np.vstack ((roots_up, roots_down))
		print(f"crits = {crits}", flush=True)
		# get rid of the redundant ones
		threshold  = 1e-6
		crits, kept      = ternary.remove_close_rows (crits, threshold)
		crits_     = np.empty((0,3))
		print(f"crits = {crits}")
		for i in range(len(crits)):
			c = np.array([ crits[i][0], crits[i][1], 1-crits[i][0]-crits[i][1] ])
			print(f"c = {c}", flush=True)
			crits_ = np.vstack((crits_, c))

		self.crits = crits_
		return 

	def stability_plots(self, ax, tern_b, edges_b, crits_b):

		p_s_space = np.arange(0.001, 1-0.001, 0.001)
		p_s = np.repeat(p_s_space, len(p_s_space))

		p_p = np.zeros(p_s.shape)
		for i in range (len(p_s_space)):
			p_p[i*len(p_s_space):(i+1)*len(p_s_space)] = np.linspace (0.001, 1-p_s_space[i], len(p_s_space))

		vals = ternary.stab_crit (p_s, p_p, self.vs, self.vc, self.vp, self.chi_ps, self.chi_pc, self.chi_sc)

		to_keep = ~np.isnan(vals)

		vals = vals [to_keep]
		p_s  = p_s  [to_keep]
		p_p  = p_p  [to_keep]

		if len(vals) == 0:
			print (f"There will be no critical points and no spinodal region.", flush=True)

		vmax = np.max(vals)
		vmin = np.min(vals)
		# norm = colors.SymLogNorm(0.001, vmin=vmin, vmax=vmax) 
		norm = colors.Normalize(vmin=-1, vmax=1)
		vals[vals<0]  = -1
		vals[vals==0] = 0
		vals[vals>0]  = 1

		# Define colors
		start_color = "lightcoral"
		end_color   = "dodgerblue"

		# Number of color steps
		num_steps = 256

		# Create colormap
		cmap = colors.LinearSegmentedColormap.from_list(
			name="custom_cmap", colors=[start_color, end_color], N=num_steps
		)

		# Test the colormap
		norm = plt.Normalize(vmin=-1, vmax=1)

		cols = cmap(norm (vals))

		if np.sign (vmax) == np.sign (vmin):
			if np.sign (vmax) >=0:
				vmin = -vmax
				print (f"There is no unstable region.", flush=True)
			else:
				vmax = -vmin
				print ("There is mostly unstable region.", flush=True)

		else:
			print ("there exist unstable regions.", flush=True)
		
		# plot the thing
		ternary.plot(ax, tern_b, edges_b, crits_b, self.crits, self.chi_ps, self.chi_pc, self.chi_sc, p_s, p_p, cols, self.root_up_p, self.root_lo_p, self.root_up_s, self.root_lo_s)
		ternary.embelish(ax, tern_b)
		ax.grid()
		return

