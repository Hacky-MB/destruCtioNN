#!/usr/bin/env python

import numpy as np


def fliplr(arr, board_size):
	# flip move
	if np.array(arr).shape == (2,):
		return [arr[0], board_size-1-arr[1]]
	# flip board
	else:
		res = []
		for r in range(0, len(arr)):
			res.append(arr[r][::-1])
		return res


def flipud(arr, board_size):
	# flip move
	if np.array(arr).shape == (2, ):
		return [board_size-1-arr[0], arr[1]]
	# flip board
	else:
		res = []
		for r in range(len(arr), 0, -1):
			res.append(arr[r-1])
		return res


def rotate_clockwise(arr, board_size):
	# rotate move
	if np.array(arr).shape == (2,):
		return [arr[1], board_size-1-arr[0]]
	# rotate board
	else:
		tmp = zip(*arr[::-1])
		return tmp


def find_n_max_2d(array, n):
	# Function returns array [[row,col,value],...] containing coordinates and maximal value
	# 	n times
	ret = []

	if n < 1:
		return ret

	for i in range(len(array)):
		for j in range(len(array[i])):

			# fill "ret" (returned array)
			if len(ret) < n:
				ret.append([i, j, array[i][j]])

			else:
				# find minimal value in array
				min = np.min(ret, axis=0)[-1]

				# if value is bigger than minimal value
				if array[i][j] > min:

					# store value in place of minimal value
					for k in range(n):
						if ret[k][2] == min:
							ret[k] = [i, j, array[i][j]]
							break

	# no point in returning value, array is sorted already
	return [r[0:2] for r in sorted(ret, key=lambda t:t[2])][::-1]

if __name__ == "__main__":
	a = [[1, 2, 3], [3, 4, 5]]
	print(a)
	print(find_n_max_2d(a, 1))
	print(find_n_max_2d(a, 2))
	print(find_n_max_2d(a, 3))
	print(find_n_max_2d(a, 4))
	print(find_n_max_2d(a, 5))


