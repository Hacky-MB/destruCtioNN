#!/usr/bin/env python

import caffe
import numpy as np
import itertools
import arrays

class NeuralNetwork:
	# can't import enum module on metacentrum (I mean, wtf...)
	# so I had to use this as a workaround
	TrainedNet = {'Policy': 1, 'PolicyWithOutcomes': 2, 'Value': 3}
	ComputingMode = {'CPU': 4, 'GPU': 5}
	Usage = {'Train': 6, 'Test': 7, 'Play': 8}

	def __init__(self, args, usage):
		self.set_computing_mode(args.mode)

		# REQUIRED: solver, load_snapshot, net, batch_size, mode
		# only for training: dataset, draw_plots, show_plots, save_snapshot, iter_limit

		self.solver = caffe.get_solver(args.solver)
		self.batch_size = args.batch_size

		self.change_batch_size(self.batch_size)

		if args.load_snapshot is not None:

			# if argument is encapsulated in list
			if args.load_snapshot[0] != args.load_snapshot[0][0]:
				self.solver.restore(args.load_snapshot[0])
			else:
				self.solver.restore(args.load_snapshot)
		self.solver.test_nets[0].share_with(self.solver.net)

		self.net = args.net

		self.boards_minibatch = np.array([])
		self.moves_minibatch = np.array([])
		if self.net != NeuralNetwork.TrainedNet['Policy']:
			self.outcomes_minibatch = []

		if usage != NeuralNetwork.Usage['Play']:
			# this data is not relevant when using NN
			# only with Gomoku client

			self.dataset = np.load(args.dataset)

			self.boards_minibatch = []
			self.moves_minibatch = []
			# only basic policy network doesnt need outcomes

			self.iter_limit = args.iter_limit
			self.save_snapshot = args.save_snapshot

			self.draw_plots = args.save_plots or args.show_plots
			if self.draw_plots:
				import nn_plot as plt
				self.plot = plt.NNPlot(args.show_plots, args.save_plots)

			self.test = np.array([])
			self.test_n = np.array([])
			self.train = np.array([])
			self.train_n = np.array([])
			self.loss = np.array([])
		else:
			self.dataset = {'boards': np.array([]), 'outcomes': np.array([]), 'moves': np.array([])}

		self.iter_cnt = 0
		self.train_set_len = len(self.boards_minibatch) * 2 / 3

	def load_dataset(self, **kwargs):
		keys = ['boards', 'moves', 'outcomes']
		for key in keys:
			try:
				self.dataset[key] = np.copy(kwargs[key])
			except KeyError:
				pass

	def load_network_input(self, boards):
		"""
		Used for loading boards in play mode (other inputs are not needed)
		:param boards: numpy array (n,2,19,19) representing board
		"""
		border = (self.solver.net.blobs['data'].data.shape[-1] - 19) / 2

		tmp_array = -np.ones(self.solver.net.blobs['data'].data.shape)
		tmp_array[:, :, border:-border, border:-border] = boards[:, :, :, :]

		self.solver.net.blobs['data'].data[:, :, :, :] = np.copy(tmp_array)

	def change_batch_size(self, batch_size):
		input_size = self.solver.net.blobs['data'].data.shape[-1]

		self.solver.net.blobs['data'].reshape(batch_size, 2, input_size, input_size)
		self.solver.net.blobs['labels'].reshape(batch_size, 1, 1, 1)

	def set_computing_mode(self, mode):
		if mode == NeuralNetwork.ComputingMode['CPU']:
			caffe.set_mode_cpu()

		elif mode == NeuralNetwork.ComputingMode['GPU']:
			import os
			from subprocess import Popen, PIPE

			# list allocated GPUs
			p = Popen(["/usr/sbin/list_cache", "arien", "gpu_allocation"], stdin=PIPE, stdout=PIPE, stderr=PIPE)
			out, err = p.communicate()  # , "|", "grep", "$PBS_JOBID"

			#out = out.split('\n')
			gpu_id = None

			for line in out:
				if os.environ['PBS_JOBID'] in line:
					try:
						gpu_id = int(line.split('\t')[0][-1])
						print line
					except NameError:
						print "gpu id: " + str(gpu_id)
						raise NameError("GPU ID not parsed out correctly!")

			# TODO: GPU ID parsing not working correctly (code is correct,
			caffe.set_mode_gpu()
			#caffe.set_device(gpu_id)
			caffe.set_device(0)
		else:
			raise Exception("Wrong mode! Choose \"CPU\" or \"GPU\".")

	def get_iter_cnt(self):
		if self.net == NeuralNetwork.TrainedNet['PolicyWithOutcomes']:
			return self.iter_cnt
		else:
			return self.solver.iter

	def train_net(self):
		try:

			if self.draw_plots:
					import progress

			while self.get_iter_cnt() < self.iter_limit:
				self.solver, train, train_n = self.transform_data_and_compute(NeuralNetwork.Usage['Train'])
				test, test_n, loss = self.transform_data_and_compute(NeuralNetwork.Usage['Test'])

				# this net doesn't use solver from framework
				# so solver.iter doesn't get update and can't be assigned
				# so I count iterations manually

				it = self.get_iter_cnt()

				if self.draw_plots:
					progress.printProgress(it, self.iter_limit, prefix='Progress:', suffix='Complete', length=50, decimals=2)

					self.plot.add_iter(it, test, test_n, train, train_n, loss)

				self.test = np.append(self.test, test)
				self.test_n = np.append(self.test, test_n)
				self.train = np.append(self.train, train)
				self.train_n = np.append(self.train, train_n)
				self.loss = np.append(self.loss, loss)

		except KeyboardInterrupt:
			pass

		except:
			import sys
			print "Unexpected error:", sys.exc_info()[0]
			if self.save_snapshot:
				self.solver.snapshot()
			raise

		it = self.get_iter_cnt()

		if self.save_snapshot:
			self.solver.snapshot()

		np.savez("stats" + str(it), iter=self.iter_cnt, test=self.test,
			test_n=self.test_n, train=self.train, train_n=self.train_n, loss=self.loss)

		try:
			if self.plot.show_plot:
				print ""
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
	def transform_data_and_compute(self, usage):
		indices = self.generate_indices()

		# n last moves
		# indices = range(len(moves) - size,len(moves))

		self.boards_minibatch = self.dataset['boards'][indices, :, :]
		self.moves_minibatch = self.dataset['moves'][indices, :]

		if self.net != NeuralNetwork.TrainedNet['Policy']:
			self.outcomes_minibatch = self.dataset['outcomes'][indices]


		# get padding size from net
		border = (self.solver.net.blobs['data'].data.shape[-1] - 19) / 2

		boards_out1 = -np.ones((self.batch_size, 19 + 2 * border, 19 + 2 * border))
		boards_out2 = -np.ones((self.batch_size, 19 + 2 * border, 19 + 2 * border))
		boards_out1[:, border:-border, border:-border] = self.boards_minibatch
		boards_out2[:, border:-border, border:-border] = self.boards_minibatch
		boards_out1[boards_out1 == 2] = 0
		boards_out2[boards_out2 == 1] = 0
		boards_out2[boards_out2 == 2] = 1

		# sample,y,x,player
		boards_out = np.stack([boards_out1, boards_out2], axis=3)

		# sample,player,y,x
		self.boards_minibatch = boards_out.transpose([0, 3, 1, 2])

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
			for i in range(len(self.moves_minibatch)):
				self.boards_minibatch[i, 0], self.boards_minibatch[i, 1], self.moves_minibatch[i] = \
					t(self.boards_minibatch[i, 0], board_size), \
					t(self.boards_minibatch[i, 1], board_size), \
					t(self.moves_minibatch[i], board_size)

			# iterate over rotations
			for r in range(0, 4):

				# rotate board
				for i in range(len(self.moves_minibatch)):
					self.boards_minibatch[i][0], self.boards_minibatch[i][1], self.moves_minibatch[i] = \
						arrays.rotate_clockwise(self.boards_minibatch[i][0], board_size), \
						arrays.rotate_clockwise(self.boards_minibatch[i][1], board_size), \
						arrays.rotate_clockwise(self.moves_minibatch[i], board_size)

				# load boards
				#
				solver, train_guess, train_guess_n = \
					self.step(NeuralNetwork.Usage['Train'])

			# flip board back
			for i in range(len(self.moves_minibatch)):
				self.boards_minibatch[i, 0], self.boards_minibatch[i, 1], self.moves_minibatch[i] = \
					t(self.boards_minibatch[i, 0], board_size),\
					t(self.boards_minibatch[i, 1], board_size), \
					t(self.moves_minibatch[i], board_size)

		return self.solver, train_guess, train_guess_n

	def step(self, usage):
		# Function trains net and returns tuple respectively depending on "mode" argument
		# @param mode == "train" -> @return solver
		# @param mode == "test" -> @return acc(guessed move),acc(move in top 5 most probable),loss
		# @param mode == "play" -> @return list(top 5 probable moves)

		# load expected move
		if usage != NeuralNetwork.Usage['Play']:
			for i in range(len(self.moves_minibatch)):
				self.solver.net.blobs['labels'].data[i] = (self.moves_minibatch[i][0]) * 19 + (self.moves_minibatch[i][1])

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

					win_multiplier = 1

					if not self.outcomes_minibatch[0]:
						win_multiplier = -1

					self.solver.net.layers[k].blobs[0].data[...] -= win_multiplier * self.solver.net.layers[k].blobs[0].diff
					self.solver.net.layers[k].blobs[1].data[...] -= win_multiplier * self.solver.net.layers[k].blobs[1].diff

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

			if self.net == NeuralNetwork.TrainedNet['Value']:
				return self.solver.net.blobs['out'].data[0, 0, 0, 0]

			# find last convolution layer name
			last_conv = sorted([layer for layer in self.solver.net.blobs.keys() if layer[0:4] == "conv"])[-1]

			# return most probable move
			return arrays.find_n_max_2d(self.solver.net.blobs[last_conv].data[0].reshape(19, 19), 5)
