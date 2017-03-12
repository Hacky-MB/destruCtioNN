#!/usr/bin/env python

import caffe
import numpy as np
from enum import Enum
import itertools
import arrays


class NeuralNetwork:
	# can't import enum class on metacentrum (I mean, wtf...)
	# so I had to use this as a workaround
	TrainedNet = {'Policy': 1, "PolicyWithOutcomes": 2, "Value": 3}
	ComputingMode = {'CPU': 1, 'GPU': 2}
	Usage = {'Train': 1, 'Test': 2, 'Play': 3}

	def __init__(self, args, usage):
		self.set_computing_mode(args.mode)

		# REQUIRED: solver, load_snapshot, net, batch_size, mode
		# only for training: dataset, draw_plots, show_plots, save_snapshot, iter_limit

		self.solver = caffe.get_solver(args.solver)
		self.batch_size = args.batch_size

		if args.load_snapshot is not None:
			self.solver.restore(args.load_snapshot[0])
		self.solver.test_nets[0].share_with(self.solver.net)

		# get size of input array
		input_size = self.solver.net.blobs['data'].data.shape[-1]

		self.solver.net.blobs['data'].reshape(self.batch_size, 2, input_size, input_size)
		self.solver.net.blobs['labels'].reshape(self.batch_size, 1, 1, 1)

		self.net = args.net

		self.boards = np.array([])
		self.moves = np.array([])
		if self.net != NeuralNetwork.TrainedNet['Policy']:
			self.outcomes = []

		if usage != NeuralNetwork.Usage['Play']:
			# this data is not relevant when using NN
			# only with Gomoku client

			self.dataset = np.load(args.dataset)

			self.boards = []
			self.moves = []
			# only basic policy network doesnt need outcomes

			self.iter_limit = args.iter_limit
			self.save_snapshot = args.save_snapshot

			if args.save_plots or args.show_plots:
				import nn_plot as plot
				self.plot = plot.NNPlot(args.show_plots, args.save_plots)

		self.iter_cnt = 0
		self.train_set_len = len(self.boards) * 2/3

	def set_data(self, boards, moves, outcomes):
		self.boards = boards
		self.moves = moves
		self.outcomes = outcomes

	def set_computing_mode(self, mode):
		if mode == NeuralNetwork.ComputingMode['CPU']:
			caffe.set_mode_cpu()

		elif mode == NeuralNetwork.ComputingMode['GPU']:
			import os
			from subprocess import Popen, PIPE

			# list allocated GPUs
			p = Popen(["/usr/sbin/list_cache", "arien", "gpu_allocation"], stdin=PIPE, stdout=PIPE, stderr=PIPE)
			out, err = p.communicate()

			out = out.split('\n')
			gpu_id = None

			for line in out:
				if os.environ['PBS_JOBID'] in line:
					try:
						gpu_id = int(line.split('\t')[0][-1])
					except NameError:
						print "gpu id: " + str(gpu_id)
						raise NameError("GPU ID not parsed out correctly!")

			caffe.set_mode_gpu()
			caffe.set_device(gpu_id)
		else:
			raise Exception("Wrong mode! Choose \"cpu\" or \"gpu\".")

	def get_iter_cnt(self):
		if self.net == NeuralNetwork.TrainedNet['PolicyWithOutcomes']:
			return self.iter_cnt
		else:
			return self.solver.iter

	def train(self):
		try:
			while self.get_iter_cnt() < self.iter_limit:
				self.solver, train, train_n = self.train_iter(NeuralNetwork.Usage['Train'])
				test, test_n, loss = self.train_iter(NeuralNetwork.Usage['Test'])

				# this net doesn't use solver from framework
				# so solver.iter doesn't get update and can't be assigned
				# so I count iterations manually
				try:
					if self.net == NeuralNetwork.TrainedNet['PolicyWithOutcomes']:
						self.iter_cnt += 12

						self.plot.add_iter(self.iter_cnt, test, test_n, train, train_n, loss)
					else:
						self.plot.add_iter(self.solver.iter, test, test_n, train, train_n, loss)
				except AttributeError:
					pass

		except KeyboardInterrupt:
			if self.save_snapshot:
				self.solver.snapshot()
			pass
		except:
			import sys
			print "Unexpected error:", sys.exc_info()[0]
			if self.save_snapshot:
				self.solver.snapshot()
			raise

		try:
			self.plot.draw()
		except AttributeError:
			pass

	def generate_indices(self):
		rng = self.dataset['boards'].shape[0]
		indices = np.random.choice(rng, self.batch_size)

		# only policy network with outcomes requires special minibatch
		# (all outcomes must be win / lose (True/False))
		if self.net != NeuralNetwork.TrainedNet['PolicyWithOutcomes']:
			return indices

		win = True
		loss = False

		wins, loses = 0, 0

		for i in range(self.batch_size):
			if self.dataset['outcomes'][indices[i]] == win:
				wins += 1
			else:
				loses += 1

		if loses != 0:
			if wins > loses:
				different_outcome = loss
			else:
				different_outcome = win

		# iterate over indices
		for i in range(len(indices)):
			if self.dataset['outcomes'][indices[i]] == different_outcome:
				while True:
					tmp = np.random.randint(rng)
					if (tmp not in indices) and self.dataset['outcomes'][tmp] != different_outcome:
						break
				indices[i] = tmp
		return indices

	# Function prepares data and passes them further to NN
	def train_iter(self, usage):
		indices = self.generate_indices()

		# n last moves
		# indices = range(len(moves) - size,len(moves))

		self.boards = self.dataset['boards'][indices, :, :]
		self.moves = self.dataset['moves'][indices, :]

		if self.net != NeuralNetwork.TrainedNet['Policy']:
			self.outcomes = self.dataset['boards']['outcomes'][indices]


		# get padding size from net
		border = (self.solver.net.blobs['data'].data.shape[-1] - 19) / 2

		boards_out1 = -np.ones((self.batch_size, 19 + 2 * border, 19 + 2 * border))
		boards_out2 = -np.ones((self.batch_size, 19 + 2 * border, 19 + 2 * border))
		boards_out1[:, border:-border, border:-border] = self.boards
		boards_out2[:, border:-border, border:-border] = self.boards
		boards_out1[boards_out1 == 2] = 0
		boards_out2[boards_out2 == 1] = 0
		boards_out2[boards_out2 == 2] = 1

		# sample,y,x,player
		boards_out = np.stack([boards_out1, boards_out2], axis=3)

		# sample,player,y,x
		self.boards = boards_out.transpose([0, 3, 1, 2])

		# train on current board and move
		if usage == NeuralNetwork.Usage['Train']:
			return self.transform_train()
			# return train(solver, boards_out, moves, mode)
		elif usage == NeuralNetwork.Usage['Test']:
			# guess,n_guess,loss
			return self.step(usage)
		elif usage == NeuralNetwork.Usage['Play']:
			# array of most probable moves
			return self.step(usage)

	def transform_train(self):
		# Function transforms boards and moves (for better training results) and trains net

		transformations = [lambda x, y: x, arrays.fliplr, arrays.flipud]
		board_size = 19

		# iterate over transformations
		for t in transformations:
			for i in range(len(self.moves)):
				self.boards[i, 0], self.boards[i, 1], self.moves[i] = \
					t(self.boards[i, 0], board_size), \
					t(self.boards[i, 1], board_size), \
					t(self.moves[i], board_size)

			# iterate over rotations
			for r in range(0, 4):

				# rotate board
				for i in range(len(self.moves)):
					self.boards[i][0], self.boards[i][1], self.moves[i] = \
						arrays.rotate_clockwise(self.boards[i][0], board_size), \
						arrays.rotate_clockwise(self.boards[i][1], board_size), \
						arrays.rotate_clockwise(self.moves[i], board_size)

				solver, train_guess, train_guess_n = \
					self.step(NeuralNetwork.Usage['Train'])
			# flip board back
			for i in range(len(self.moves)):
				self.boards[i, 0], self.boards[i, 1], self.moves[i] = \
					t(self.boards[i, 0], board_size),\
					t(self.boards[i, 1], board_size), \
					t(self.moves[i], board_size)

		return self.solver, train_guess, train_guess_n

	def step(self, usage):
		# Function trains net and returns tuple respectively depending on "mode" argument
		# @param mode == "train" -> @return solver
		# @param mode == "test" -> @return acc(guessed move),acc(move in top 5 most probable),loss
		# @param mode == "play" -> @return list(top 5 probable moves)

		# load boards
		self.solver.net.blobs['data'].data[:, :, :, :] = self.boards[:, :, :, :]

		# load expected move
		if usage != NeuralNetwork.Usage['Play']:
			for i in range(len(self.moves)):
				self.solver.net.blobs['labels'].data[i] = (self.moves[i][0]) * 19 + (self.moves[i][1])

		# move is the most probable
		if usage == NeuralNetwork.Usage['Train']:

			if self.net == NeuralNetwork.TrainedNet['PolicyWithOutcomes']:
				self.solver.net.forward()

				# if not outcomes[0]:
				# 	solver.net.blobs['loss'].data *= -1

				self.solver.net.backward()

				# update weights
				for k in range(len(self.solver.net.layers)):
					if self.solver.net.layers[k].type == "Input":
						continue

					if len(self.solver.net.layers[k].blobs) == 0:
						continue

					self.solver.net.layers[k].blobs[0].diff[...] *= 0.001  # weights
					self.solver.net.layers[k].blobs[1].diff[...] *= 0.001  # biases

					# play with outcomes

					if not self.outcomes[0]:
						for i in range(2):  # params, biases
							# min = np.amin(solver.net.layers[k].blobs[i].diff)
							# max = np.amax(solver.net.layers[k].blobs[i].diff)
							shape = [range(t) for t in self.solver.net.layers[k].blobs[0].diff.shape]
							#
							for i1, i2, i3, i4 in itertools.product(*shape):
								self.solver.net.layers[k].blobs[0].diff[i1, i2, i3, i4] *= -1
							# 	solver.net.layers[k].blobs[0].diff[i1, i2, i3, i4] = max + min - \
							# 							solver.net.layers[k].blobs[0].diff[i1, i2, i3, i4]

					self.solver.net.layers[k].blobs[0].data[...] -= self.solver.net.layers[k].blobs[0].diff
					self.solver.net.layers[k].blobs[1].data[...] -= self.solver.net.layers[k].blobs[1].diff

				# last_conv = last_conv = sorted([l for l in solver.net.blobs.keys() if l[0:4] == "conv"])[-1]
				# la.show_l(solver, last_conv, "b", 256, 1)
			else:
				self.solver.step(1)

			return self.solver, self.solver.net.blobs['accuracy'].data, self.solver.net.blobs['accuracy5'].data

		elif usage == NeuralNetwork.Usage['Test']:
			self.solver.net.forward()
			return self.solver.net.blobs['accuracy'].data, \
				self.solver.net.blobs['accuracy5'].data, self.solver.net.blobs['loss'].data

		elif usage == NeuralNetwork.Usage['Play']:
			self.solver.net.forward()

			# find last convolution layer name
			last_conv = sorted([layer for layer in self.solver.net.blobs.keys() if layer[0:4] == "conv"])[-1]

			# return most probable move
			return arrays.find_n_max_2d(self.solver.net.blobs[last_conv].data[0].reshape(19, 19), 5)
