#!/usr/bin/env python

import numpy as np
import arrays as arrs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import caffe

"""
This module is used for interacting with neural network.
It trains neural networks and it's used by other scripts for
playing Gomoku.
"""


def train(solver, boards, moves, mode):
	# Function trains net and returns tuple respectively depending on "mode" argument
	# @param mode == "train" -> @return solver
	# @param mode == "test" -> @return acc(guessed move),acc(move in top 5 most probable),loss
	# @param mode == "play" -> @return list(top 5 probable moves)

	# load boards
	solver.net.blobs['data'].data[:, :, :, :] = boards[:, :, :, :]

	# load expected move
	if mode != "play":
		for i in range(len(moves)):
			solver.net.blobs['labels'].data[i] = (moves[i][0]) * 19 + (moves[i][1])

	# move is the most probable
	if mode == "train":
		solver.step(1)
		return solver

	elif mode == "test":
		solver.net.forward()
		return solver.net.blobs['accuracy'].data, solver.net.blobs['accuracy5'].data, solver.net.blobs['loss'].data

	elif mode == "play":
		solver.net.forward()
		# return most probable move
		return arrs.find_n_max_2d(solver.net.blobs['conv3'].data[0][0], 5)

	else:
		raise Exception("Wrong mode! (use train/test)")


def transform_train(solver, boards, moves):
	# Function transforms boards and moves (for better training results) and trains net

	transformations = [lambda x, y: x, arrs.fliplr, arrs.flipud]

	board_size = 19

	# iterate over transformations
	for t in transformations:  # range(0,1):#

		# flip board
		for i in range(len(moves)):
			boards[i, 0], boards[i, 1], moves[i] = t(boards[i, 0], board_size), \
													t(boards[i, 1], board_size), \
													t(moves[i], board_size)

		# iterate over rotations
		for r in range(0, 4):  # range(0,1):#

			# rotate board
			for i in range(len(moves)):
				boards[i][0], boards[i][1], moves[i] = arrs.rotate_clockwise(boards[i][0], board_size), \
														arrs.rotate_clockwise(boards[i][1], board_size), \
														arrs.rotate_clockwise(moves[i], board_size)

			solver = train(solver, boards, moves, "train")

		# flip board back
		for i in range(len(moves)):
			boards[i, 0], boards[i, 1], moves[i] = t(boards[i, 0], board_size),\
													t(boards[i, 1], board_size), \
													t(moves[i], board_size)

	return solver


# Function prepares data and passes them further to NN
def train_iter(solver, boards, moves, size, mode):
	# n random moves
	indices = np.random.choice(boards.shape[0], size)

	# n last moves
	# indices = range(len(moves) - size,len(moves))

	if mode != "play":
		boards = boards[indices, :, :]
		moves = moves[indices, :]
	else:
		boards = boards[indices[0], :, :]
		moves = moves[indices[0], :]

	# get padding size from net
	border = (solver.net.blobs['data'].data.shape[-1] - 19) / 2

	boards_out1 = -np.ones((size, 19 + 2 * border, 19 + 2 * border))
	boards_out2 = -np.ones((size, 19 + 2 * border, 19 + 2 * border))
	boards_out1[:, border:-border, border:-border] = boards
	boards_out2[:, border:-border, border:-border] = boards
	boards_out1[boards_out1 == 2] = 0
	boards_out2[boards_out2 == 1] = 0
	boards_out2[boards_out2 == 2] = 1

	# sample,y,x,player
	boards_out = np.stack([boards_out1, boards_out2], axis=3)

	# sample,player,y,x
	boards_out = boards_out.transpose([0, 3, 1, 2])

	# train on current board and move
	if mode == "train":
		# solver
		return transform_train(solver, boards_out, moves)
	elif mode == "test":
		# guess,n_guess,loss
		return train(solver, boards_out, moves, mode)
	elif mode == "play":
		# array of most probable moves
		return train(solver, boards_out, moves, mode)
	else:
		raise Exception("Wrong mode! (use train/test)")


# Function creates plot from computed values
# it - iteration number
# test_guess - guessed move on test dataset
# test_guess_n - move in top 5 most probable on test dataset
# train_guess - guessed move on training dataset
# train_guess_n - move in top 5 most probable on training dataset
# l - loss layer output
def plot(it, test_guess, test_guess_n, train_guess, train_guess_n, l, show_plot, save_plot):
	# number of points on plot
	points = 50

	length = len(it)

	# number of values averaged per one point in plot
	samples = length / points
	if not samples:
		samples = 1

	trash = length % samples

	# cut off values from the end so that length = points * samples
	if trash > 0:
		it = np.array(it[:-trash])
		test_guess = np.array(test_guess[:-trash])
		test_guess_n = np.array(test_guess_n[:-trash])
		l = np.array(l[:-trash])

	# samples * point
	length = len(it)

	(x, y1, y2) = (np.array(it[::length / points]),
					np.mean(test_guess.reshape(-1, samples), axis=1),
					np.mean(test_guess_n.reshape(-1, samples), axis=1)) \
		if len(it) > points else (it, test_guess, test_guess_n)

	# plt.subplot(1, 2, 1)
	plt.title('Accuracy of test net')

	plt.plot(x, y1, linestyle='-', marker='o', color='red')
	plt.plot(x, y2, linestyle='-', marker='o', color='blue')
	red = mpatches.Patch(color='red', label='move guessed')
	blue = mpatches.Patch(color='blue', label='move in top 5')
	plt.legend(handles=[red, blue], loc=2, borderaxespad=0.)
	plt.ylim((0, 1))
	plt.xlim((x[0], x[-1]) if len(x) else (0, 1))

	# (y1,y2) = (np.mean(train_guess.reshape(-1,length/points), axis=1),
	# np.mean(test_guess_n.reshape(-1,length/points), axis=1) if len(it)>points else (train_guess,train_guess_n)

	# plt.subplot(1, 2, 2)
	# plt.title('Accuracy of train net')
	# plt.plot(x, y1, linestyle='-', marker='o', color='red' )
	# plt.plot(x, y2, linestyle='-', marker='o', color='blue' )
	# red = mpatches.Patch(color='red', label='move guessed')
	# blue = mpatches.Patch(color='blue', label='move in top 5')
	# plt.legend(handles=[red,blue], loc=2, borderaxespad=0.)
	# plt.ylim((0,1))
	# plt.xlim((x[0],x[-1]) if len(x) else (0,1))

	# first save, then show (or save figure - fig = plt.gcf();fig.savefig('awd'))
	# new figure is created when show() is called
	if save_plot:
		plt.savefig('probs' + str(int(it[-1]) if length else "") + '.svg')

	if show_plot:
		plt.show()
	else:
		plt.clf()

	l = np.mean(l.reshape(-1, samples), axis=1) if len(it) > points else l

	plt.plot(x, l, linestyle='-', marker='o', color='red')
	red = mpatches.Patch(color='red', label='loss')
	plt.legend(handles=[red], loc=2, borderaxespad=0.)
	plt.ylim((0, max(l) + max(l) * 0.1 if len(l) else 1))
	# plt.xlim((it[0],it[-1] if len(it) else 0,1))
	# plt.xlim((it[0] if len(it) else 0,it[-1] if len(it) else 1))
	plt.xlim((it[0], it[-1]) if len(it) else (0, 1))

	if save_plot:
		plt.savefig('loss' + str(int(it[-1]) if len(it) else "") + '.svg')

	if show_plot:
		plt.show()
	else:
		plt.clf()


def parse_args():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('-s', '--solver', required=True)
	parser.add_argument('-i', '--iter_limit', type=int, default=100000)
	parser.add_argument('-d', '--dataset', default="./boards2.npz")
	parser.add_argument('-p', '--show_plots', action='store_true')
	parser.add_argument('-v', '--save_snapshot', action='store_true')
	parser.add_argument('-l', '--load_snapshot', nargs=1)
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()

	caffe.set_mode_cpu()
	solver = caffe.get_solver(args.solver)

	if args.load_snapshot is not None:
		solver.restore(args.load_snapshot[0])
		solver.test_nets[0].share_with(solver.net)

	f = np.load(args.dataset)

	boards = f['boards']
	moves = f['moves']

	it = np.array([]) 	# number of iterations passed
	test_guess = np.array([], dtype=float)  	# correct guess (0-1) on test set
	test_guess_n = np.array([], dtype=float)  	# guess in tom 5 most probable on test set
	train_guess = np.array([], dtype=float)  	# correct guess (0-1) on train set
	train_guess_n = np.array([], dtype=float)  	# guess in tom 5 most probable on train set
	loss = np.array([], dtype=float)		# output of loss function layer

	batch_size = 32

	try:
		while (not len(it)) or it[-1] < args.iter_limit:
			# solver = train_iter(solver,boards,moves,batch_size,"train")
			# guess,top_guess,loss = train_iter(solver,boards,moves,batch_size,"test")

			# first 2/3 of dataset - training data
			# rest - test data
			solver = train_iter(solver, boards[:len(boards) * 2 / 3], moves[:len(boards) * 2 / 3],
																							batch_size, "train")
			# train_guess,train_n_guess,loss = train_iter(solver,boards[:len(boards)*2/3],
			# moves[:len(boards)*2/3],batch_size,"test")
			guess, n_guess, cur_loss = train_iter(solver, boards[len(boards) * 2 / 3:],
																	moves[len(boards) * 2 / 3:], batch_size, "test")

			it = np.append(it, solver.iter)
			test_guess = np.append(test_guess, float(guess))
			test_guess_n = np.append(test_guess_n, float(n_guess))
			# train_guess = np.append(train_guess, float(train_guess))
			# train_guess_n = np.append(train_guess_n, float(train_n_guess))
			loss = np.append(loss, float(cur_loss))

	except:
		pass

	if args.save_snapshot:
		solver.snapshot()

		np.savez("stats" + str(int(it[-1])), iter=it, test=test_guess, test_n=test_guess_n,  # train_n=train_guess, train=train_guess_n,
											loss=loss)

	plot(it, test_guess, test_guess_n, train_guess, train_guess_n, loss, args.show_plots, args.save_snapshot)
