#!/usr/bin/env python

from neural_network import NeuralNetwork


def parse_args():
	import argparse
	import sys
	parser = argparse.ArgumentParser()
	parser.add_argument('-b', '--batch_size', required=True, type=int)
	parser.add_argument('-d', '--dataset', default="./boards2.npz")
	parser.add_argument('-i', '--iter_limit', type=int, default=100000)
	parser.add_argument('-l', '--load_snapshot', nargs=1)
	parser.add_argument('-m', '--mode', required=True, choices=['CPU', 'GPU'], help='CPU/GPU')
	parser.add_argument('-n', '--net', required=True,
		choices=['P', 'PWO', 'V'], help='Policy net, Policy without outcomes or Value network')
	parser.add_argument('-p', '--show_plots', action='store_true')
	parser.add_argument('-r', '--save_plots', action='store_true')
	parser.add_argument('-s', '--solver', required=True)
	parser.add_argument('-v', '--save_snapshot', action='store_true')

	if len(sys.argv) == 1:
		parser.print_help()
		sys.exit()

	args = parser.parse_args()

	if args.mode == 'CPU':
		args.mode = NeuralNetwork.ComputingMode['CPU']
	elif args.mode == 'GPU':
		args.mode = NeuralNetwork.ComputingMode['GPU']
	else:
		raise Exception("Wrong computing mode! Choose \"GPU\" or \"CPU\"")

	if args.net == "P":
		args.net = NeuralNetwork.TrainedNet['Policy']
	elif args.net == "PWO":
		args.net = NeuralNetwork.TrainedNet['PolicyWithOutcomes']
	elif args.net == "V":
		args.net = NeuralNetwork.TrainedNet['Value']
	else:
		raise Exception("Wrong trained net argument! Choose \"P\", \"PWO\" or \"V\".")

	return args

if __name__ == "__main__":
	args = parse_args()

	nn = NeuralNetwork(args, NeuralNetwork.TrainedNet['PolicyWithOutcomes'])
	nn.train_net()
