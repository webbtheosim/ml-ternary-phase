import numpy as np
import matplotlib.pyplot as plt
import FreeEnergy as FE
import linecache 
import pickle
import warnings 
import time
import argparse
import mpltern

parser = argparse.ArgumentParser(description='Create a skeleton solution for the binodal. This is a memory-intensive computation.')
parser.add_argument('--mesh',   metavar='mesh',    dest='mesh',          type=int,   action='store', help='enter mesh density.')
parser.add_argument('--chisc',  metavar='chi_sc',  dest='chi_sc',        type=float, action='store', help='enter S-C exchange parameter.')
parser.add_argument('--chips',  metavar='chi_ps',  dest='chi_ps',        type=float, action='store', help='enter P-S exchange parameter.')
parser.add_argument('--chipc',  metavar='chi_pc',  dest='chi_pc',        type=float, action='store', help='enter P-C exchange parameter.')
parser.add_argument('-vs',      metavar='vs',      dest='vs',            type=float, action='store', help='specific volume of solvent.'  )
parser.add_argument('-vc',      metavar='vc',      dest='vc',            type=float, action='store', help='specific volume of cosolvent.')
parser.add_argument('-vp',      metavar='vp',      dest='vp',            type=float, action='store', help='specific volume of polymer.'  )
parser.add_argument('--hull',   metavar='hull',    dest='hull',          type=str,   action='store', help='location of serialized hull structure.')
parser.add_argument('--img',    metavar='img',     dest='img',           type=str,   action='store', help='name of image to be created.', default=None)
args = parser.parse_args()


###################################################
def custom_warning_format(message, category, filename, lineno, line=None):
	line = linecache.getline(filename, lineno).strip()
	return f"There is a RunTimeWarning taking place on line {lineno}.\n"

warnings.formatwarning = custom_warning_format
###################################################

if __name__=="__main__":

	start = time.time()

	inputs = dict()
	inputs["vs"    ] = args.vs
	inputs["vc"    ] = args.vc
	inputs["vp"    ] = args.vp
	inputs["chi_sc"] = args.chi_sc
	inputs["chi_ps"] = args.chi_ps
	inputs["chi_pc"] = args.chi_pc
	inputs["n_mesh"] = args.mesh

	F = FE.FreeEnergy(inputs)
	print(f"Getting critical points...", end=' ', flush=True)
	F.find_crit_point()
	print("done!", flush=True)

	print(f"Generating hulls...", flush=True)
	F.generate_hulls()
	dfile = open(args.hull, 'wb')
	pickle.dump(F, dfile)
	dfile.close()

	stop = time.time()
	print(f"Time for hull computation is {stop-start} seconds.", flush=True)

