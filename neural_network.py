#!/usr/bin/env python

import caffe
import numpy as np
import itertools
import arrays
import os

class NeuralNetwork:
	# can't import enum module on metacentrum (I mean, wtf...)
	# so I had to use this as a workaround
	TrainedNet = {'Policy': 1, 'PolicyWithOutcomes': 2, 'Value': 3, 'WinPolicy': 4}
	ComputingMode = {'CPU': 10, 'GPU': 11}
	Usage = {'Train': 20, 'Test': 21, 'Play': 22}

	#solver, snapshot, batch_size
	# iter_limit, save_snapshot, show_plots
	def __init__(self, args, usage):
		self.set_computing_mode(args.mode)

		# REQUIRED: solver, load_snapshot, net, batch_size, mode
		# only for training: dataset, draw_plots, show_plots, save_snapshot, iter_limit

		self.solver = caffe.get_solver(args.solver)

		self.batch_size = None
		self.change_batch_size(args.batch_size)

		if args.load_snapshot is not None:
			self.restore_snapshot(args.load_snapshot)
		self.solver.test_nets[0].share_with(self.solver.net)

		self.net_type = args.net

		self.boards_minibatch = np.array([])
		self.moves_minibatch = np.array([])

		# only basic policy network doesnt need outcomes
		if self.net_type != NeuralNetwork.TrainedNet['Policy']:
			self.outcomes_minibatch = []

		self.learning_rate = 0.001

		if usage != NeuralNetwork.Usage['Play']:
			# this data is not relevant when using NN
			# only with Gomoku client

			self.dataset = np.load(args.dataset)

			self.iter_limit = args.iter_limit
			self.save_snapshot = args.save_snapshot

			self.show_progress = args.show_progress

			self.iter = np.array([])
			self.test = np.array([])
			self.test_n = np.array([])
			self.train = np.array([])
			self.train_n = np.array([])
			self.loss = np.array([])
		else:
			self.dataset = {'boards': np.array([]), 'outcomes': np.array([]), 'moves': np.array([])}

		self.iter_cnt = 0

		self.train_set_len = len(self.dataset['boards'])
		#elif self.train_set_len > 0:
		#	self.train_set_len = len(self.boards_minibatch) * 2 / 3
		#else:
		#	raise AttributeError("Dataset length has to be -1 or greater than 0!")

	def restore_snapshot(self, snapshot):
		self.solver.restore(snapshot)
		self.solver.test_nets[0].share_with(self.solver.net)

	def load_dataset(self, **kwargs):
		keys = ['boards', 'moves', 'outcomes']
		for key in keys:
			try:
				self.dataset[key] = np.copy(kwargs[key])
			except KeyError:
				pass

	def initialize_network_input(self):
		self.solver.net.blobs['data'].data[...] = -np.ones(self.solver.net.blobs['data'].data.shape)[...]

	def load_input_sample(self, index, data):
		border = (self.solver.net.blobs['data'].data.shape[-1] - 19) / 2
		self.solver.net.blobs['data'].data[index, :, border:-border, border:-border] = data[...]

	def change_batch_size(self, batch_size):
		assert batch_size > 0, "Batch size muse be bigger than 0!"
		self.batch_size = batch_size

		n, c, h, w = self.solver.net.blobs['data'].data.shape
		self.solver.net.blobs['data'].reshape(batch_size, c, h, w)

		n, c, h, w = self.solver.net.blobs['labels'].data.shape
		self.solver.net.blobs['labels'].reshape(batch_size, c, h, w)

	def set_single_input_value(self, board, player, y, x):
		assert player == 0 or player == 1, "\"Player\" value must be 0 (my board) or 1 (enemy board)!"

		border = (self.solver.net.blobs['data'].data.shape[-1] - 19) / 2
		y += border
		x += border
		self.solver.net.blobs['data'].data[board, player, y, x] = 1

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

			# TODO: GPU ID activation not working correctly (code is correct though)
			caffe.set_mode_gpu()
			#caffe.set_device(gpu_id)
			caffe.set_device(0)
		else:
			raise Exception("Wrong mode! Choose \"CPU\" or \"GPU\".")

	def get_iter_cnt(self):
		if self.net_type == NeuralNetwork.TrainedNet['PolicyWithOutcomes']:
			return self.iter_cnt
		else:
			return self.solver.iter

	def train_net(self):
		try:

			if self.show_progress:
					import progress_bar
					progress_bar.printProgress(0, self.iter_limit, prefix='Progress:', suffix='Complete', length=50, decimals=2)

			while self.get_iter_cnt() < self.iter_limit:
				self.transform_data_and_compute(NeuralNetwork.Usage['Train'])
				train = self.solver.net.blobs['accuracy'].data

				if self.net_type == NeuralNetwork.TrainedNet['Value']:
					train_n = 0
				else:
					train_n = self.solver.net.blobs['accuracy5'].data
				#test, test_n, loss = self.transform_data_and_compute(NeuralNetwork.Usage['Test'])

				loss = self.solver.net.blobs['loss'].data

				# Policy network using outcomes doesn't use solver from framework
				# so solver.iter doesn't get update and can't be assigned
				# so I count iterations manually
				if self.net_type == NeuralNetwork.TrainedNet['PolicyWithOutcomes']:
					self.iter_cnt += 12 * self.batch_size

					if self.iter_cnt % 24996 == 0:
						self.solver.snapshot()

						import os
						os.rename("./net_snapshot_iter_0.caffemodel", "./net_snapshot_iter_"+self.iter_cnt+".caffemodel")
						os.rename("./net_snapshot_iter_0.solverstate", "./net_snapshot_iter_"+self.iter_cnt+".solverstate")

					from math import floor, log10

					number_of_divisions = 8
					if number_of_divisions * self.iter_cnt / self.iter_limit != int(log10(0.001 / self.learning_rate)):
						self.learning_rate /= 5

				it = self.get_iter_cnt()

				if it > 40:
					import sys
					sys.exit()

				if self.show_progress:
					progress_bar.printProgress(it, self.iter_limit, prefix='Progress:', suffix='Complete', length=50, decimals=2)

				self.iter = np.append(self.iter, it)
				# self.test = np.append(self.test, test)
				# self.test_n = np.append(self.test, test_n)
				self.train = np.append(self.train, train)
				self.train_n = np.append(self.train_n, train_n)
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

		np.savez("stats" + str(it), iter=self.iter, train=self.train, train_n=self.train_n, loss=self.loss)

	def generate_indices(self):
		rng = self.dataset['boards'].shape[0]
		indices = np.random.choice(rng, self.batch_size)

		# only policy network with outcomes requires special minibatch
		# (all outcomes must be win / lose (True/False))

		if self.net_type == NeuralNetwork.TrainedNet['PolicyWithOutcomes']:
			#return indices

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
			for i in range(self.batch_size):
				if self.dataset['outcomes'][indices[i]] == different_outcome:
					while True:
						tmp = np.random.randint(rng)
						if (tmp not in indices) and self.dataset['outcomes'][tmp] != different_outcome:
							break
					indices[i] = tmp

		elif self.net_type == NeuralNetwork.TrainedNet['WinPolicy']:
			loss = False

			for i in range(self.batch_size):
				while self.dataset['outcomes'][indices[i]] == loss:
					indices[i] = np.random.choice(rng, 1)[0]

		# no point in training value network on early boards
		elif self.net_type == NeuralNetwork.TrainedNet['Value']:
			for i in range(self.batch_size):
				while np.sum(self.dataset['boards'][indices[i]]) < 20:
					indices[i] = np.random.choice(rng, 1)[0]

		return indices

	# Function prepares data and passes them further to NN
	def transform_data_and_compute(self, usage):
		indices = self.generate_indices()

		self.boards_minibatch = np.copy(self.dataset['boards'][indices, :, :])
		if self.net_type != NeuralNetwork.TrainedNet['Value']:
			self.moves_minibatch = np.copy(self.dataset['moves'][indices, :])

		# used with universal dataset
		# if self.net_type != NeuralNetwork.TrainedNet['Value']:
		# 	self.boards_minibatch = self.dataset['boards'][indices, :, :]
		# 	self.moves_minibatch = np.copy(self.dataset['moves'][indices, :])
		#
		# else:
		# 	self.boards_minibatch = np.copy(self.dataset['boards'][indices, :, :])
		# 	for i in range(self.batch_size):
		# 		y, x = self.dataset['moves'][indices[i]]
		# 		self.boards_minibatch[i, y, x] = 1

		if self.net_type != NeuralNetwork.TrainedNet['Policy']:
			tmp = []
			for a in self.dataset['outcomes'][indices]:
				tmp.append([[[int(a), int(not a)]]])
			self.outcomes_minibatch = np.stack(tmp, axis=0)

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
			if self.net_type == NeuralNetwork.TrainedNet['Value']:
				self.solver.net.blobs['labels'].data[...] = self.outcomes_minibatch[...]
			return self.transform_train()
			# return train(solver, boards_out, moves, mode)
		else:
			self.solver.net.blobs['data'].data[...] = self.boards_minibatch[...]

			if self.net_type == NeuralNetwork.TrainedNet['Value']:
				self.solver.net.blobs['labels'].data[...] = self.outcomes_minibatch[...]
			else:
				for i in range(len(self.moves_minibatch)):
					self.solver.net.blobs['labels'].data[i] = (self.moves_minibatch[i][0]) * 19 + (self.moves_minibatch[i][1])

			if usage == NeuralNetwork.Usage['Test']:
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
			# transform each two boards of minibatch
			for i in range(self.batch_size):
				self.boards_minibatch[i, 0], self.boards_minibatch[i, 1] = \
					t(self.boards_minibatch[i, 0], board_size), \
					t(self.boards_minibatch[i, 1], board_size)

			# transform each move
			if self.net_type != NeuralNetwork.TrainedNet['Value']:
				for i in range(self.batch_size):
					self.moves_minibatch[i] = t(self.moves_minibatch[i], board_size)

			# iterate  over rotations
			for r in range(0, 4):

				# rotate board
				for j in range(self.batch_size):
					self.boards_minibatch[j][0], self.boards_minibatch[j][1] = \
						arrays.rotate_clockwise(self.boards_minibatch[j][0], board_size), \
						arrays.rotate_clockwise(self.boards_minibatch[j][1], board_size)

				# load boards
				self.solver.net.blobs['data'].data[...] = self.boards_minibatch[...]

				# load moves
				if self.net_type != NeuralNetwork.TrainedNet['Value']:
					for i in range(self.batch_size):
						self.moves_minibatch[i] = arrays.rotate_clockwise(self.moves_minibatch[i], board_size)

					for i in range(len(self.moves_minibatch)):
						self.solver.net.blobs['labels'].data[i] = (self.moves_minibatch[i][0]) * 19 + (self.moves_minibatch[i][1])

				self.step(NeuralNetwork.Usage['Train'])

			# flip board back
			for i in range(self.batch_size):
				self.boards_minibatch[i, 0], self.boards_minibatch[i, 1] = \
					t(self.boards_minibatch[i, 0], board_size),\
					t(self.boards_minibatch[i, 1], board_size)

				# value network does not use moves
				if self.net_type != NeuralNetwork.TrainedNet['Value']:
					self.moves_minibatch[i] = t(self.moves_minibatch[i], board_size)

		return

	def step(self, usage):
		# Function trains net and returns tuple respectively depending on "mode" argument
		# @param mode == "train" -> @return solver
		# @param mode == "test" -> @return acc(guessed move),acc(move in top 5 most probable),loss
		# @param mode == "play" -> @return list(top 5 probable moves)

		# move is the most probable
		if usage == NeuralNetwork.Usage['Train']:

			if self.net_type == NeuralNetwork.TrainedNet['PolicyWithOutcomes']:
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

					self.solver.net.layers[k].blobs[0].diff[...] *= self.learning_rate  # weights
					self.solver.net.layers[k].blobs[1].diff[...] *= self.learning_rate  # biases

					# play with outcomes

					win_multiplier = 1

					if not self.outcomes_minibatch[0, 0, 0, 0]:
						win_multiplier = -1

					self.solver.net.layers[k].blobs[0].data[...] -= win_multiplier * self.solver.net.layers[k].blobs[0].diff
					self.solver.net.layers[k].blobs[1].data[...] -= win_multiplier * self.solver.net.layers[k].blobs[1].diff

				# from layer import show_l
				# last_conv = last_conv = sorted([l for l in solver.net.blobs.keys() if l[0:4] == "conv"])[-1]
				# la.show_l(solver, last_conv, "b", 256, 1)
			else:
				# from layer import show_l
				# last_conv = last_conv = sorted([l for l in self.solver.net.blobs.keys() if l[0:4] == "conv"])[-1]
				# show_l(self.solver, "conv1", "p", 32, 64)
				self.solver.step(1)

			return

		elif usage == NeuralNetwork.Usage['Test']:
			self.solver.net.forward()
			return self.solver.net.blobs['accuracy'].data, \
				self.solver.net.blobs['accuracy5'].data, self.solver.net.blobs['loss'].data

		elif usage == NeuralNetwork.Usage['Play']:
			self.solver.net.forward()

			# find last convolution layer name
			last_conv = sorted([l for l in self.solver.net.blobs.keys() if l[0:4] == "conv"])[-1]
			# show_l(self.solver, last_conv, "b", 1, 1)

			if self.net_type == NeuralNetwork.TrainedNet['Value']:
				return self.solver.net.blobs['out'].data[0, 0, 0, 0], self.solver.net.blobs['out'].data[0, 0, 0, 1]


			# return most probable move
			result_moves = []
			for i in range(self.solver.net.blobs['data'].data.shape[0]):
				result_moves.append(arrays.find_n_max_2d(self.solver.net.blobs[last_conv].data[i].reshape(19, 19), 10))
			return result_moves
