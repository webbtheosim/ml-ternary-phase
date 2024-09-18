import numpy as np


def tangent2 (vs, vc, vp, ps, pp, cpc, cps, csc, root_up_s, root_lo_s):

	dist_lo  = np.linalg.norm(pp - root_lo_s (ps))
	dist_up  = np.linalg.norm(pp - root_up_s (ps))
	print(f"up = {root_up_s(ps)}")
	print(f"lo = {root_lo_s(ps)}")
	print(f"dist_lo = {dist_lo}")
	print(f"dist_up = {dist_up}")

	if abs(dist_up) > 100:
		tang_slope = 0

	elif abs(dist_lo) > abs(dist_up):
		tang_slope = (-((2 * cpc + ps * vs * cpc**2 + ps * vs * (cps - csc)**2 - \
		2 * ps * vs * cpc * (cps + csc)) * (2 * vc * vp * cpc - \
		vc * vp * vs * cpc**2 + 2 * ps * vc * vp * vs * cpc**2 - \
		2 * vp * vs * cps + 2 * vc * vp * vs * cpc * cps - \
		4 * ps * vc * vp * vs * cpc * cps - vc * vp * vs * cps**2 + \
		2 * ps * vc * vp * vs * cps**2 + 2 * vc * vs * csc + \
		2 * vc * vp * vs * cpc * csc - \
		4 * ps * vc * vp * vs * cpc * csc + \
		2 * vc * vp * vs * cps * csc - \
		4 * ps * vc * vp * vs * cps * csc - vc * vp * vs * csc**2 + \
		2 * ps * vc * vp * vs * csc**2 + (-4 * vc * vp * vs * (cpc**2 + \
		(cps - csc)**2 - \
		2 * cpc * (cps + csc)) * (ps * vs + (-1 + \
		ps) * vc * (-1 + 2 * ps * vs * csc)) - \
		4 * vc * vp * (2 * cpc + ps * vs * cpc**2 + \
		ps * vs * (cps - csc)**2 - \
		2 * ps * vs * cpc * (cps + csc)) * (vs + \
		vc * (-1 + 2 * (-1 + 2 * ps) * vs * csc)) + \
		2 * (vc + vp * (-1 + 2 * ps * vs * cps) - \
		2 * ps * vc * vs * csc - (-1 + ps) * vc * vp * (2 * cpc + \
		ps * vs * cpc**2 + ps * vs * (cps - csc)**2 - \
		2 * ps * vs * cpc * (cps + csc))) * (2 * vp * vs * \
		cps - 2 * vc * vs * csc + \
		vc * vp * ((vs - 2 * ps * vs) * cpc**2 - (-1 + \
		2 * ps) * vs * (cps - csc)**2 + cpc * (-2 + \
		2 * (-1 + 2 * ps) * vs * (cps + csc)))))/(2 * \
		np.sqrt(-4 * vc * vp * (2 * cpc + ps * vs * cpc**2 + \
		ps * vs * (cps - csc)**2 - \
		2 * ps * vs * cpc * (cps + csc)) * (ps * vs + (-1 \
		+ ps) * vc * (-1 + 2 * ps * vs * csc)) + (vp - 2 * ps * vp * vs * cps + \
		vc * (-1 + \
		2 * ps * vs * csc + (-1 + ps) * vp * (2 * cpc + \
		ps * vs * cpc**2 + \
		ps * vs * (cps - csc)**2 - \
		2 * ps * vs * cpc * (cps + csc))))**2)))) + \
		vs * (cpc**2 + (cps - csc)**2 - \
		2 * cpc * (cps + csc)) * (-vc + vp - \
		2 * vc * vp * cpc + 2 * ps * vc * vp * cpc - \
		ps * vc * vp * vs * cpc**2 + ps**2 * vc * vp * vs * cpc**2 - \
		2 * ps * vp * vs * cps + 2 * ps * vc * vp * vs * cpc * cps - \
		2 * ps**2 * vc * vp * vs * cpc * cps - ps * vc * vp * vs * cps**2 + \
		ps**2 * vc * vp * vs * cps**2 + 2 * ps * vc * vs * csc + \
		2 * ps * vc * vp * vs * cpc * csc - \
		2 * ps**2 * vc * vp * vs * cpc * csc + \
		2 * ps * vc * vp * vs * cps * csc - \
		2 * ps**2 * vc * vp * vs * cps * csc - ps * vc * vp * vs * csc**2 + \
		ps**2 * vc * vp * vs * csc**2 + np.sqrt(-4 * vc * vp * (2 * cpc + \
		ps * vs * cpc**2 + ps * vs * (cps - csc)**2 - \
		2 * ps * vs * cpc * (cps + csc)) * (ps * vs + (-1 + \
		ps) * vc * (-1 + 2 * ps * vs * csc)) + (vp - \
		2 * ps * vp * vs * cps + \
		vc * (-1 + \
		2 * ps * vs * csc + (-1 + ps) * vp * (2 * cpc + \
		ps * vs * cpc**2 + ps * vs * (cps - csc)**2 - \
		2 * ps * vs * cpc * (cps + csc))))**2)))/(2 * vc \
		* vp * (2 * cpc + ps * vs * cpc**2 + ps * vs * (cps - csc)**2 - \
		2 * ps * vs * cpc * (cps + csc))**2)
		# print (f"tang_slope = {tang_slope}")

	else:
		tang_slope = (-((2 * cpc + ps * vs * cpc**2 + ps * vs * (cps - csc)**2 - \
		2 * ps * vs * cpc * (cps + csc)) * (2 * vc * vp * cpc - \
		vc * vp * vs * cpc**2 + 2 * ps * vc * vp * vs * cpc**2 - \
		2 * vp * vs * cps + 2 * vc * vp * vs * cpc * cps - \
		4 * ps * vc * vp * vs * cpc * cps - vc * vp * vs * cps**2 + \
		2 * ps * vc * vp * vs * cps**2 + 2 * vc * vs * csc + \
		2 * vc * vp * vs * cpc * csc - \
		4 * ps * vc * vp * vs * cpc * csc + \
		2 * vc * vp * vs * cps * csc - \
		4 * ps * vc * vp * vs * cps * csc - vc * vp * vs * csc**2 + \
		2 * ps * vc * vp * vs * csc**2 + (4 * vc * vp * vs * (cpc**2 + \
		(cps - csc)**2 - \
		2 * cpc * (cps + csc)) * (ps * vs + (-1 + \
		ps) * vc * (-1 + 2 * ps * vs * csc)) + \
		4 * vc * vp * (2 * cpc + ps * vs * cpc**2 + 
		ps * vs * (cps - csc)**2 - \
		2 * ps * vs * cpc * (cps + csc)) * (vs + \
		vc * (-1 + 2 * (-1 + 2 * ps) * vs * csc)) - \
		2 * (vc + vp * (-1 + 2 * ps * vs * cps) - 
		2 * ps * vc * vs * csc - (-1 + ps) * vc * vp * (2 * cpc + \
		ps * vs * cpc**2 + ps * vs * (cps - csc)**2 - 
		2 * ps * vs * cpc * (cps + csc))) * (2 * vp * vs * \
		cps - 2 * vc * vs * csc + \
		vc * vp * ((vs - 2 * ps * vs) * cpc**2 - (-1 + \
		2 * ps) * vs * (cps - csc) **2 + cpc * (-2 + 
		2 * (-1 + 2 * ps) * vs * (cps + csc))))) / (2 * \
		np.sqrt(-4 * vc * vp * (2 * cpc + ps * vs * cpc**2 + \
		ps * vs * (cps - csc)**2 - \
		2 * ps * vs * cpc * (cps + csc)) * (ps * vs + (-1 \
		+ ps) * vc * (-1 + 2 * ps * vs * csc)) + (vp - 2 * ps * vp * vs * cps + \
		vc * (-1 + 2 * ps * vs * csc + (-1 + ps) * vp * (2 * cpc + \
		ps * vs * cpc**2 + ps * vs * (cps - csc)**2 - \
		2 * ps * vs * cpc * (cps + csc))))**2)))) + \
		vs * (cpc**2 + (cps - csc)**2 - \
		2 * cpc * (cps + csc)) * (-vc + vp - \
		2 * vc * vp * cpc + 2 * ps * vc * vp * cpc - \
		ps * vc * vp * vs * cpc**2 + ps**2 * vc * vp * vs * cpc**2 - \
		2 * ps * vp * vs * cps + 2 * ps * vc * vp * vs * cpc * cps - \
		2 * ps**2 * vc * vp * vs * cpc * cps - ps * vc * vp * vs * cps**2 + \
		ps**2 * vc * vp * vs * cps**2 + 2 * ps * vc * vs * csc + \
		2 * ps * vc * vp * vs * cpc * csc - \
		2 * ps**2 * vc * vp * vs * cpc * csc + \
		2 * ps * vc * vp * vs * cps * csc - \
		2 * ps**2 * vc * vp * vs * cps * csc - ps * vc * vp * vs * csc**2 + \
		ps**2 * vc * vp * vs * csc**2 - np.sqrt(-4 * vc * vp * (2 * cpc + \
		ps * vs * cpc**2 + ps * vs * (cps - csc)**2 - \
		2 * ps * vs * cpc * (cps + csc)) * (ps * vs + (-1 + \
		ps) * vc * (-1 + 2 * ps * vs * csc)) + (vp - \
		2 * ps * vp * vs * cps + \
		vc * (-1 + \
		2 * ps * vs * csc + (-1 + ps) * vp * (2 * cpc + \
		ps * vs * cpc**2 + ps * vs * (cps - csc)**2 - \
		2 * ps * vs * cpc * (cps + csc))))**2)))/(2 * vc * \
		vp * (2 * cpc + ps * vs * cpc**2 + ps * vs * (cps - csc)**2 - \
		2 * ps * vs * cpc * (cps + csc))**2)
		# print (f"tang_slope = {tang_slope}")

	return tang_slope
