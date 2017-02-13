#!/usr/bin/env python

import sys
import os
import numpy as np
import train2 as tr

def help():
	print "Program concatenates all .npz files in given folder and recreates plots."

def parse_args():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('-f', '--folder', required=True, help="folder with \".npz\" files")
	parser.add_argument('-s', '--save_stats', action='store_true', help='save resulting \".npz\" file if defined')
	parser.add_argument('-p', '--save_plots', action='store_true', help='save plots if defined')
	return parser.parse_args()


def main():
	if len(sys.argv) == 0:
		help()
		sys.exit()

	args = parse_args()
	stats = sorted([f for f in os.listdir(args.folder) if (len(f) > 5 and f[:5] == "stats")])

	it = np.array([]) 	# number of iterations passed
	test_guess = np.array([])  	# correct guess (0-1) on test set
	test_guess_n = np.array([])  	# guess in tom 5 most probable on test set
	loss = np.array([])		# output of loss function layer

	for stat in stats:
		f = np.load(str(args.folder)+"/"+stat)
		it = np.concatenate((it, f['iter']))
		test_guess = np.concatenate((test_guess, f['test']))
		test_guess_n = np.concatenate((test_guess_n, f['test_n']))
		loss = np.concatenate((loss, f['loss']))

	tr.plot(it, test_guess, test_guess_n, [], [], loss, True, True)

	if args.save_stats:
		np.savez("stats" + str(int(it[-1])), iter=it, test_n=test_guess_n, test=test_guess,  # train_n=tr1, train=tr2,
											loss=loss)

if __name__ == "__main__":
	main()
