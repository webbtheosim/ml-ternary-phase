import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, delaunay_plot_2d, tsearch, ConvexHull, distance
from scipy.optimize import fsolve 
import scipy.integrate as integrate
import sympy.matrices as smat
import mu
import spinodal
import copy
import time

vol_threshold   = 1e-9
conc_threshold  = 1e-9
# n_mesh          = 1001
n_species       = 3
max_free_energy = 100.0
cap_free_energy = max_free_energy - 10.0

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

class FreeEnergy:
    def __init__(self, inputs):
        self.n_mesh   = inputs["n_mesh"]
        self.chi_sc   = inputs["chi_sc"]
        self.chi_ps   = inputs["chi_ps"]
        self.chi_pc   = inputs["chi_pc"]
        self.vs       = inputs["vs"]
        self.vc       = inputs["vc"]
        self.vp       = inputs["vp"]
        self.v        = np.array([self.vs, self.vp, self.vc])
        self.chi      = np.array([[0, self.chi_ps, self.chi_sc], [self.chi_ps, 0, self.chi_pc], [self.chi_sc, self.chi_pc, 0]])
        self.spinodal = spinodal.Spinodal(inputs)
        # self.mu       = mu.sym_mu_ps(inputs, self.spinodal)
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
    
    # get the critical points
    def crit_condition (self, phi_s, phi_p):

        phi_c = 1-phi_p-phi_s
        t1    = 1/(self.vc*phi_c) + 1/(phi_p*self.vp) - 2*self.chi_pc
        t2    = (1/(self.vc*(phi_c)**2) - 1/(self.vs* phi_s**2))*(1/(self.vc*(phi_c)) + 1/(phi_p*self.vp) - 2*self.chi_pc) + (1/(self.vc*phi_c) + 1/(self.vs*phi_s) - 2*self.chi_sc)/(self.vc*(phi_c)**2) - 2*(1/(self.vc*phi_c) - self.chi_pc - self.chi_sc + self.chi_ps)/(self.vc*(phi_c)**2)

        u1    = (1/(self.vc*phi_c) + 1/(phi_p*self.vp) - 2*self.chi_pc)/(self.vc*(phi_c)**2) + (1/(self.vc*phi_c**2) - 1/(phi_p**2 * self.vp))*(1/(self.vc*phi_c) + 1/(self.vs*phi_s) - 2*self.chi_sc) - 2*(1/(self.vc*phi_c) + self.chi_ps - self.chi_sc - self.chi_pc)/(self.vc*phi_c**2)
        u2    = 1/(self.vc*phi_c) - self.chi_pc - self.chi_sc + self.chi_ps

        return t1*t2 - u1*u2
    
    def find_crit_point(self):
        RTOL = 1e-12
        def send_to_fsolve_r1 (phi_s):
            phi_p_upper = self.root_up_s (phi_s)
            return self.crit_condition (phi_s, phi_p_upper)

        def send_to_fsolve_r2 (phi_s):
            phi_p_lower = self.root_lo_s (phi_s)
            return self.crit_condition (phi_s, phi_p_lower)

        def send_to_fsolve_r3 (phi_p):
            phi_s_upper = self.root_up_p (phi_p)
            return self.crit_condition (phi_s_upper, phi_p)

        def send_to_fsolve_r4 (phi_p):
            phi_s_lower = self.root_lo_p (phi_p)
            return self.crit_condition (phi_s_lower, phi_p)
        
        guesses    = np.hstack((np.logspace(-9,-4, int(1e+2)), np.linspace(0.0002, 0.9999, int(1e+4)-int(200)), 1-np.logspace(-4,-9, int(1e+2))))
        roots_up   = np.empty((0,2))
        roots_down = np.empty((0,2))

        print(f"Solving for critical points in four sweeps. ", flush=True)
        print("\tIn sweep one...", flush=True)
        for g in guesses:
            root = fsolve (send_to_fsolve_r1, g)
            if abs(send_to_fsolve_r1(root)) < RTOL:
                if root[0] >= 1 or root[0] <= 0 or np.isnan(root[0]):
                    pass
                else:
                    r_up  = self.root_up_s (root[0])
                    r_tup = np.array([root[0], r_up])
                    if (r_tup >= 1).any() or (r_tup <= 0).any() or np.sum(r_tup) >= 1 or np.isnan(r_up):
                        pass
                    else:
                        if len(roots_up) == 0:
                            roots_up = np.vstack ((roots_up,r_tup))
                        else:
                            roots_up = np.vstack ((roots_up,r_tup))   
            else:
                pass
       # print(f"roots up = {roots_up}")

        print ("\tIn sweep two...", flush=True)
        for g in guesses:
            root = fsolve (send_to_fsolve_r3, g)
            if abs(send_to_fsolve_r3(root)) < RTOL:
                if root[0] >= 1 or root[0] <= 0 or np.isnan(root[0]):
                    pass
                else:
                    r_up  = self.root_up_p(root[0])
                    r_tup = np.array([r_up, root[0]])
                    if (r_tup >= 1).any() or (r_tup <= 0).any() or np.sum(r_tup) >= 1 or np.isnan(r_up):
                        pass
                    else:
                        if len(roots_up) == 0:
                            roots_up = np.vstack ((roots_up,r_tup))
                        else:
                            roots_up = np.vstack ((roots_up,r_tup))   
            else:
                pass
        # print(f"roots up = {roots_up}")

        print ("\tIn sweep three...", flush=True)
        for g in guesses:
            root = fsolve (send_to_fsolve_r2, g)
            if abs(send_to_fsolve_r2(root)) < RTOL:
                if root[0] >= 1 or root[0] <= 0 or np.isnan(root):
                    pass
                else:
                    r_lo = self.root_lo_s(root[0])
                    r_tup = np.array([root[0], r_lo])
                    if (r_tup >= 1).any() or (r_tup <= 0).any() or np.sum(r_tup) >= 1 or np.isnan(r_lo):
                        pass
                    else: 
                        if len(roots_down) == 0:
                            roots_down = np.vstack ((roots_down,r_tup))
                        else:
                            roots_down = np.vstack ((roots_down,r_tup))
            else:
                pass
        # print(f"roots down = {roots_down}")

        print ("\tIn sweep four...", flush=True)
        for g in guesses:
            root = fsolve (send_to_fsolve_r4, g)
            if abs(send_to_fsolve_r4(root)) < RTOL:
                if root[0] >= 1 or root[0] <= 0 or np.isnan(root):
                    pass
                else:
                    r_lo = self.root_lo_p(root[0])
                    r_tup = np.array([r_lo, root[0]])
                    if (r_tup >= 1).any() or (r_tup <= 0).any() or np.sum(r_tup) >= 1 or np.isnan(r_lo):
                        pass
                    else:
                        if len(roots_down) == 0:
                            roots_down = np.vstack ((roots_down,r_tup))
                        else:
                            roots_down = np.vstack ((roots_down,r_tup))
            else:
                pass
        # print(f"roots down = {roots_down}")

        # check for consistency
        # First determinant
        def first_det(phi_s, phi_p):
            phi_c = 1-phi_s-phi_p
            t1    = (1/(self.vp*phi_p) + 1/(self.vc * (1 - phi_p - phi_s)) - 2*self.chi_pc) 
            t2    = (1/(self.vc * phi_c**2) - 1/(self.vs * phi_s**2))*(1/(self.vp*phi_p) + 1/(self.vc*phi_c) - 2 * self.chi_pc) + (1/(self.vc * phi_c) + 1/(self.vs * phi_s) - 2*self.chi_sc)/(self.vc * phi_c**2) - 2*(1/(self.vc * phi_c) - self.chi_pc + self.chi_ps - self.chi_sc)/(self.vc * phi_c**2) 
            t3    = 1/(self.vc * phi_c) - self.chi_pc + self.chi_ps - self.chi_sc 
            t4    = (1/(self.vp * phi_p) + 1/(self.vc * phi_c) - 2 * self.chi_pc)/(self.vc * phi_c**2) + (-1/(self.vp * phi_p**2) + 1/(self.vc * phi_c**2)) * (1/(self.vc * phi_c) + 1/(self.vs * phi_s) - 2*self.chi_sc) - 2*(1/(self.vc * phi_c) - self.chi_pc + self.chi_ps - self.chi_sc)/(self.vc * phi_c**2)
            return t1*t2 - t3*t4
        
        # second determinant
        def second_det(phi_s, phi_p):
            phi_c = 1-phi_s-phi_p
            t1    = 1/(self.vc * phi_c) + 1/(self.vs * phi_s) - 2*self.chi_sc
            t2    = (1/(self.vp*phi_p) + 1/(self.vc * phi_c) - 2 * self.chi_pc)/(self.vc * phi_c ** 2) + (-1/(self.vp * phi_p**2) + 1/(self.vc * phi_c**2))*(1/(self.vc*phi_c) + 1/(self.vs*phi_s) - 2*self.chi_sc) - 2*(1/(self.vc * phi_c) - self.chi_pc + self.chi_ps - self.chi_sc)/(self.vc * phi_c ** 2)
            t3    = 1/(self.vc * phi_c) - self.chi_pc + self.chi_ps - self.chi_sc 
            t4    = (1/(self.vc*phi_c**2) - 1/(self.vs*phi_s**2))*(1/(self.vp*phi_p) + 1/(self.vc * phi_c) - 2*self.chi_pc) + (1/(self.vc*phi_c) + 1/(self.vs*phi_s) - 2*self.chi_sc)/(self.vc * phi_c**2) - 2*(1/(self.vc * phi_c) - self.chi_pc + self.chi_ps - self.chi_sc)/(self.vc * phi_c**2)
            return t1*t2 - t3*t4

        check1_up = np.abs(first_det (roots_up[:,0], roots_up[:,1])) < RTOL
        check2_up = np.abs(second_det(roots_up[:,0], roots_up[:,1])) < RTOL
        roots_up  = roots_up[np.logical_and(check1_up, check2_up)]

        check1_down = np.abs(first_det (roots_down[:,0], roots_down[:,1])) < RTOL
        check2_down = np.abs(second_det(roots_down[:,0], roots_down[:,1])) < RTOL
        roots_down  = roots_down[np.logical_and(check1_down, check2_down)]
        print("Complete critical point sweeps!", flush=True)

        crits       = np.vstack((roots_up, roots_down))
        threshold   = 1e-6
        crits, kept = remove_close_rows(crits, threshold)

        self.crits  = crits
        # print(f"first_det  = {first_det(self.crits[1][1], self.crits[1][0])}")
        # print(f"second_det = {second_det(self.crits[1][1], self.crits[1][0])}")
        # print(f"crit[1] = {self.crits[1]}")
        # print(f"phi_s   = {self.root_lo_p(self.crits[1][0])}")
        # print(f"phi_s   = {self.root_up_p(self.crits[1][0])}")

        return
    
    # now moving on to stuff that delves into free energy geometry
    def FH_free_energy(self, phi):
        phi2 = np.maximum(phi, conc_threshold * np.ones_like(phi))
        return np.sum(phi/self.v * np.log(phi2)) + 1 / 2 * phi.T @ self.chi @ phi
    
    # map concentration to coordinates
    def from_phi_to_xy(self, phi1_mesh, phi2_mesh):
        return np.array([(2 / np.sqrt(3)) * (1 - phi1_mesh / 2 - phi2_mesh), phi1_mesh]).T

    # map coordinates to concentration
    def from_xy_to_phi(self, pt):
        return np.array((pt[:, 1], 1 - (np.sqrt(3) * pt[:, 0] + pt[:, 1]) / 2, (np.sqrt(3) * pt[:, 0] - pt[:, 1]) / 2)).T
    
    # generate the mesh 
    def generate_hulls(self):
        phi_list = np.linspace(0, 1, self.n_mesh)
        phi_s_mesh, phi_p_mesh = np.meshgrid(phi_list, phi_list, indexing="ij")
        phi_s_mesh = phi_s_mesh.flatten()
        phi_p_mesh = phi_p_mesh.flatten()
        if len(self.crits) > 0:
            for c in self.crits:
                phi_s_mesh = np.hstack((phi_s_mesh, c[0]))
                phi_p_mesh = np.hstack((phi_p_mesh, c[1]))
        phi_c_mesh = 1.0 - phi_s_mesh - phi_p_mesh 

        # curate the compositions
        mask = (phi_c_mesh >= -conc_threshold)
        phi_s_mesh, phi_p_mesh, phi_c_mesh = phi_s_mesh[mask], phi_p_mesh[mask], phi_c_mesh[mask]

        # stack them up
        phi_mesh  = np.vstack((phi_s_mesh, phi_p_mesh, phi_c_mesh)).T 
        mesh_size = phi_mesh.shape[0]

        f_mesh = []
        for i in range(mesh_size):
            free_energy = self.FH_free_energy(phi_mesh[i])
            f_mesh.append(free_energy)
        f_mesh = np.array(f_mesh)

        xy_mesh       = self.from_phi_to_xy(phi_mesh[:,0], phi_mesh[:,1])
        surface_raw   = np.hstack((xy_mesh, f_mesh.reshape(-1,1)))
        surface_lid   = np.hstack((xy_mesh, max_free_energy * np.ones((mesh_size, 1))))
        surface_tot   = np.vstack((surface_raw, surface_lid))
        hull          = ConvexHull(surface_tot)
        num_simplices = hull.simplices.shape[0]
        simplex_mask  = np.ones(num_simplices, dtype=bool) 

        for i in range(num_simplices):
            simplex_pts = surface_tot[hull.simplices[i]]
            if simplex_pts[:, -1].max() >= cap_free_energy:
                simplex_mask[i] = False 
            else:
                simplex_pts -= simplex_pts[-1, :]
                simplex_vol  = np.abs(np.linalg.det(simplex_pts[:-1,:-1]))
                if simplex_vol < vol_threshold:
                    simplex_mask[i] = False 

        # get the relevant simplices
        surf_simplices = hull.simplices[simplex_mask]
        num_simplices  = surf_simplices.shape[0]

        # get into calculating phases and splits 
        nearest_neighbor_distance = (phi_list[1] - phi_list[0])/np.sqrt(3) * 2
        cutoff                    = nearest_neighbor_distance * 3

        # this is the container for how many phases a certain composition will split into 
        num_phases                = np.zeros((num_simplices))

        # get the splits and stuff 
        for i in range(num_simplices):
            simplex_pts   = surface_tot[surf_simplices[i]]
            A             = 1 * (cutoff > distance.squareform(distance.pdist(simplex_pts[:,:-1])))
            D             = np.diag(np.sum(A, axis=1))
            num_phases[i] = smat.Matrix(D-A).eigenvals()[0]

        self.surface    = surface_tot      # this is the surface in XY coordinates
        self.hull       = hull             # this is the hull for XY coordinates
        self.simplices  = surf_simplices   # simplices in XY coordinates
        self.num_phases = num_phases       # these are the phases that each simplex corresponds 
        self.cutoff     = cutoff           # these are the cutoffs
        return

    # get the phase splits
    def calc_phase_split(self, phi):
        EPS           = 1e-9
        pts_test      = np.append(self.from_phi_to_xy(phi[0], phi[1]), [0.0])
        simplex_id    = -1
        num_simplices = self.simplices.shape[0]

        for i in range(num_simplices):
            simplex          = self.surface[self.simplices[i]] # this is the simplex in question in xy coordinates
            vertices_phi     = self.from_xy_to_phi(simplex)        # this is the simplex in phi coordinates
            vertices_portion = np.linalg.solve(vertices_phi.T, phi.T)
            if vertices_portion.min() > -EPS:
                simplex_id = i
                break 

        # get the simplex and the A matrix
        simplex      = self.surface[self.simplices[simplex_id]]                   # get the simplex again
        A            = 1 * (self.cutoff > distance.squareform(distance.pdist(simplex[:, :-1]))) # get the A matrix to run the solve

        # first, we distribute the concentration phi to all the vertices (the solution is unique)
        vertices_phi     = self.from_xy_to_phi(simplex)
        vertices_portion = np.linalg.solve(vertices_phi, phi.T)

        # then we combine the identical phases
        tot_phase    = 0
        phase_belong = np.ones(n_species) * -1 

        for i in range(n_species):
            if phase_belong[i] < 0:
                phase_belong[i] = tot_phase
                tot_phase += 1
            for j in range(i+1, n_species):
                if A[i, j] > 0:
                    phase_belong[j] = phase_belong[i]

        # instantiate containers on how the solution splits 
        phase_composition = np.zeros((n_species, n_species))
        phase_portion     = np.zeros(n_species)
        for i in range(tot_phase):
            for j in range(n_species):
                if phase_belong[j] == i:
                    phase_portion[i] += vertices_portion[j]
                    phase_composition[i,:] += vertices_phi[j] * vertices_portion[j]
            phase_composition[i, :] /= phase_portion[i]

        ids = np.argsort(phase_composition[:tot_phase,2])
        phase_portion[:tot_phase]        = phase_portion[ids]
        phase_composition[:tot_phase, :] = phase_composition[ids, :]
        phase_portion[tot_phase:]        = np.nan
        phase_composition[tot_phase:, :] = np.nan
        return tot_phase, phase_composition, phase_portion, simplex_id

    # get the phase splits
    def calc_phase_split_specific(self, phi, simplex_id):

        # get the simplex and the A matrix
        simplex      = self.surface[self.simplices[simplex_id]]                   # get the simplex again
        A            = 1 * (self.cutoff > distance.squareform(distance.pdist(simplex[:, :-1]))) # get the A matrix to run the solve

        # first, we distribute the concentration phi to all the vertices (the solution is unique)
        vertices_phi     = self.from_xy_to_phi(simplex)
        # print(f"simplex phi = {vertices_phi}")
        # print(f"phi = {phi}")
        vertices_portion = np.linalg.solve(vertices_phi.T, phi.T)
        # print(f"portion = {vertices_portion}", flush=True)

        # then we combine the identical phases
        tot_phase    = 0
        phase_belong = np.ones(n_species) * -1 

        for i in range(n_species):
            if phase_belong[i] < 0:
                phase_belong[i] = tot_phase
                tot_phase += 1
            for j in range(i+1, n_species):
                if A[i, j] > 0:
                    phase_belong[j] = phase_belong[i]

        # instantiate containers on how the solution splits 
        phase_composition = np.zeros((n_species, n_species))
        phase_portion     = np.zeros(n_species)
        for i in range(tot_phase):
            for j in range(n_species):
                if phase_belong[j] == i:
                    phase_portion[i] += vertices_portion[j]
                    phase_composition[i,:] += vertices_phi[j] * vertices_portion[j]
            phase_composition[i, :] /= phase_portion[i]

        ids = np.argsort(phase_composition[:tot_phase,2])
        phase_portion[:tot_phase]        = phase_portion[ids]
        phase_composition[:tot_phase, :] = phase_composition[ids, :]
        phase_portion[tot_phase:]        = np.nan
        phase_composition[tot_phase:, :] = np.nan
        return tot_phase, phase_composition, phase_portion, simplex_id

    # get total splits
    def splits(self, mesh):
        phi_list = np.linspace(0.001, 0.999, mesh)
        phi_s_mesh, phi_p_mesh = np.meshgrid(phi_list, phi_list, indexing="ij")
        phi_s_mesh = phi_s_mesh.flatten()
        phi_p_mesh = phi_p_mesh.flatten()
        phi_c_mesh = 1 - phi_s_mesh - phi_p_mesh

        # curate the compositions 
        mask = (phi_c_mesh >= 0.00001)
        phi_s_mesh, phi_p_mesh, phi_c_mesh = phi_s_mesh[mask], phi_p_mesh[mask], phi_c_mesh[mask]

        # stack them up 
        phi_mesh  = np.vstack((phi_s_mesh, phi_p_mesh, phi_c_mesh)).T
        xy_mesh   = self.from_phi_to_xy(phi_mesh[:,0], phi_mesh[:,1])
        mesh_size = phi_mesh.shape[0]
        num_phase = np.zeros(mesh_size, dtype=int)
        simplex_id        = np.zeros(mesh_size, dtype=int)
        phase_portion     = np.zeros((mesh_size, n_species))
        phase_composition = np.zeros((mesh_size, n_species, n_species))
        for i in range(mesh_size):
            print(f"@ {i}/{mesh_size}", flush=True)
            (num_phase[i], phase_composition[i,:,:], phase_portion[i,:], simplex_id[i]) = self.calc_phase_split(phi_mesh[i])
        
        return phi_mesh, phase_composition, phase_portion, num_phase, simplex_id


