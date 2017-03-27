import caffe
import numpy as np
from neural_network import NeuralNetwork


class AI:
	def __init__(self, args):
		self.stone = 1
		self.my_board = 0
		self.enemy_board = 1

		self.board = np.zeros((2, 19, 19))
		self.policy_nn = NeuralNetwork(args, NeuralNetwork.Usage['Play'])

	def load_board(self, board):
		self.board = np.copy(board)

	def make_move(self):
		self.policy_nn.load_network_input(np.array([self.board]))
		# arr = self.nn.step(NeuralNetwork.Usage['Play'])

		# see last convolution layer
		# last_conv = sorted([l for l in solver.net.blobs.keys() if l[0:4] == "conv"])[-1]
		# layer.show_l(solver, last_conv, "b", 1, 1)

		found = False
		i = 0

		# try to play one of recieved moves
		# - safety measure
		# while i < 5:
		# 	y = arr[i][0]
		# 	x = arr[i][1]
		#
		# 	if (self.board[:, y, x] == 0).all():
		# 		found = True
		# 		break
		# 	i += 1

		# play randome move if all guessed positions are occupied
		if not found:
			exp = True
			while exp:
				x, y = np.random.choice(19, 2)

				if (self.board[:, y, x] == 0).all():
					exp = False

		self.board[self.my_board, y, x] = 1
		return x, y

	def enemy_move(self, x, y):
		self.board[self.enemy_board, y, x] = self.stone

	def reset_board(self):
		self.board = np.zeros((2, 19, 19))

	def print_board(self, board):
		pboard = self.board[self.my_board]+self.board[self.enemy_board]*2
		# print first line with x coordinates
		print "  ",
		for i in range(19):
			tmp = '{0:2d}'.format(i)
			print tmp,
		print""

		for row in range(len(pboard)):
			# print y coordinate at the start of line
			print'{0:2d}|'.format(row),

			for col in range(len(pboard[row])):
				if pboard[row][col] == 0:
					print "- ",
				elif pboard[row][col] == 1:
					print "X ",
				elif pboard[row][col] == 2:
					print "O ",
			print""
