#!/usr/bin/env python

import numpy as np


def print_board(board):
	# print first line with x coordinates
	print "  ",
	for i in range(19):
		tmp = '{0:2d}'.format(i)
		print tmp,
	print""

	for row in range(len(board)):
		# print y coordinate at the start of line
		print'{0:2d}|'.format(row),

		for col in range(len(board[row])):
			if board[row][col] == 0:
				print "- ",
			elif board[row][col] == 1:
				print "X ",
			elif board[row][col] == 2:
				print "O ",
		print""


def fliplr(arr, board_size):
	# flip move
	if np.array(arr).shape == (2,):
		return [arr[0], board_size-1-arr[1]]
	# flip board
	else:
		return np.fliplr(arr)


def flipud(arr, board_size):
	# flip move
	if np.array(arr).shape == (2, ):
		return [board_size-1-arr[0], arr[1]]
	# flip board
	else:
		return np.flipud(arr)


def rotate_clockwise(arr, board_size):
	# rotate move
	if np.array(arr).shape == (2,):
		return [arr[1], board_size-1-arr[0]]
	# rotate board
	else:
		return np.transpose(np.flipud(arr))


def rotate_counterclockwise(arr, board_size):
	# rotate move
	if np.array(arr).shape == (2,):
		return [board_size-1-arr[1], arr[0]]
	# rotate board
	else:
		return np.transpose(np.fliplf(arr))


def find_n_max_2d(array, n):
	# Function returns array [[row,col,value],...] containing coordinates and maximal value
	# 	n times

	assert n >= 1

	idx = np.argsort(array, axis=None)[-1:-n-1:-1]

	return [(i / 19, i % 19) for i in idx]

if __name__ == "__main__":
	a = [[1, 2, 3], [3, 4, 5]]
	print(a)
	print(find_n_max_2d(a, 1))
	print(find_n_max_2d(a, 2))
	print(find_n_max_2d(a, 3))
	print(find_n_max_2d(a, 4))
	print(find_n_max_2d(a, 5))


