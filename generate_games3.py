#!/usr/bin/env python

import numpy as np
import arrays as arrs
import os
import sys
from progress_bar import printProgress as printProgressBar
from neural_network import NeuralNetwork
from optimalized import game_end


def export(moves, output):
	# get last file
	files = [f for f in os.listdir(output) if f.startswith(str(len(moves))+"(")]
	files.sort()

	if len(files) > 0:
		last_file = files[-1]
		index = int(last_file.split('(')[1].split(')')[0]) + 1
	else:
		index = 0

	name = str(len(moves))+"("+str(index)+").psq"
	f = open(output+"/"+name, 'w')

	out = list()
	out.append("Piskvorky 19x19, 0:0, 0\n")

	for move in moves:
		out.append(str(move[0])+","+str(move[1])+",0\n")
	out.append("\n\n0\n")
	f.write("".join(out))
	f.close()
	return


def get_random_opening(openings):
	opening_index = np.random.choice(openings.shape[0], 1)[0]
	opening = openings[opening_index]

	out1, out2 = np.copy(opening), np.copy(opening)
	out1[out1 == 2] = 0
	out2[out2 == 1] = 0
	out2[out2 == 2] = 1
	out = np.stack((out1, out2))
	return out


def append_opening_moves(moves, opening):
	o1 = b = np.stack(np.where(opening[0] > 0), axis=1)
	o2 = b = np.stack(np.where(opening[1] > 0), axis=1)

	# o1 went first
	if len(o2) > len(o1):
		o1, o2 = o2, o1

	for i in range(len(o2)):
		moves.append(list(o1[i]))
		moves.append(list(o2[i]))

	moves.append(list(o1[-1]))


def initialize_game(boards, moves, game_index, iteration_cnt, networks, openings):
	opening = get_random_opening(openings)
	append_opening_moves(moves[game_index], opening)

	if iteration_cnt % 2 == 0:
		boards[game_index, :, :, :] = opening[:, :, :]
	else:
		boards[game_index, :, :, :] = opening[::-1, :, :]

	# current network gets original opening because he starts on next turn
	networks[iteration_cnt % 2].load_input_sample(game_index, opening[:, :, :])
	# waiting NN gets reverted opening
	networks[(iteration_cnt + 1) % 2].load_input_sample(game_index, opening[::-1, :, :])


def choose_move(board, move_list):
	found = False
	i = 0

	# try to play one of recieved moves
	# - safety measure
	while i < len(move_list):
		y = move_list[i][0]
		x = move_list[i][1]

		if (board[:, y, x] == 0).all():
			found = True
			break
		i += 1

	# play randome move if all guessed positions are occupied
	if not found:
		exp = True
		while exp:
			x, y = np.random.choice(19, 2)

			if (board[:, y, x] == 0).all():
				exp = False

	return [y, x]


class NeuralNetworkArgs:
	def __init__(self, solver, net_type, batch_size, snapshot=None):
		self.solver = solver
		self.load_snapshot = snapshot
		self.batch_size = batch_size
		self.net = net_type
		self.mode = NeuralNetwork.ComputingMode['CPU']


def parse_args():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('-b', '--batch_size', default=32, type=int)
	parser.add_argument('-g', '--game_cnt', help="Number of games wanted.", type=int, required=True)
	parser.add_argument('-l', '--load_snapshot', required=True)
	parser.add_argument('-o', '--output', default="./generated_games")
	parser.add_argument('-p', '--openings', default="openings2.npz")
	parser.add_argument('--solver', required=True)
	parser.add_argument('--solver2', required=True)
	parser.add_argument('--snapshots2', required=True, nargs='+', help='list of snapshots to load')
	return parser.parse_args()


def main():
	args = parse_args()

	# set up caffe
	nn_args = NeuralNetworkArgs(args.solver, NeuralNetwork.TrainedNet['Policy'], args.batch_size, args.load_snapshot)
	nn2_args = NeuralNetworkArgs(args.solver2, NeuralNetwork.TrainedNet['Policy'], args.batch_size)
	nn = NeuralNetwork(nn_args, NeuralNetwork.Usage['Play'])
	nn2 = NeuralNetwork(nn2_args, NeuralNetwork.Usage['Play'])
	networks = [nn, nn2]

	# verify output directory
	if os.path.isfile(args.output):
		raise IOError("Directory needed as output")

	# if not exist, create directory
	if not os.path.isdir(args.output):
		os.mkdir(args.output)

	boards = np.zeros((args.batch_size, 2, 19, 19)).astype(int)
	moves = []
	for i in range(args.batch_size):
		moves.append([])
	game_cnt = 0
	iteration_cnt = 0

	my_board = 0
	enemy_board = 1

	f = np.load(args.openings)
	openings = f['boards']

	printProgressBar(game_cnt, args.game_cnt, prefix='Progress:', suffix=str(game_cnt) + '/' + str(args.game_cnt), length=50)

	#load opening moves
	nn.initialize_network_input()
	nn2.initialize_network_input()

	for game_index in range(args.batch_size):
		initialize_game(boards, moves, game_index, iteration_cnt, networks, openings)

	while game_cnt < args.game_cnt:
		if iteration_cnt % 1000 == 0:
			index = np.random.choice(len(args.snapshots2), 1)[0]
			nn2.restore_snapshot(args.snapshots2[index])

		playing_nn = iteration_cnt % 2
		waiting_nn = 1 - playing_nn

		output_move_list = networks[playing_nn].step(NeuralNetwork.Usage['Play'])

		# get result
		for game_index in range(args.batch_size):
			move_list = output_move_list[game_index]
			move = choose_move(boards[game_index], move_list)

			boards[game_index, playing_nn, move[0], move[1]] = 1
			moves[game_index].append(move)

			if game_end(boards[game_index, playing_nn], move):
				if len(moves[game_index]) <= 20:
					pass
				game_cnt += 1
				export(moves[game_index], args.output)
				moves[game_index] = []

				initialize_game(boards, moves, game_index, iteration_cnt+1, networks, openings)

				printProgressBar(game_cnt, args.game_cnt, prefix='Progress:',
					suffix=str(game_cnt) + '/' + str(args.game_cnt), length=50)

			# draw game
			elif np.count_nonzero(boards[game_index]) == 361:
				initialize_game(boards, moves, game_index, iteration_cnt+1, networks, openings)

			else:
				networks[playing_nn].set_single_input_value(game_index, my_board, move[0], move[1])
				networks[waiting_nn].set_single_input_value(game_index, enemy_board, move[0], move[1])
		iteration_cnt += 1
	return

if __name__ == "__main__":
	main()

