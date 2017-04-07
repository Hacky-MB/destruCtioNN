import numpy as np
cimport numpy as np
from cpython.mem cimport PyMem_Malloc #, PyMem_Realloc, PyMem_Free

DTYPE = np.int
ctypedef np.int_t DTYPE_t

cimport cython
#@cython.boundscheck(False) # turn off bounds-checking for entire function
#@cython.wraparound(False)  # turn off negative index wrapping for entire function

def stringify_array(np.ndarray[DTYPE_t, ndim=2] array):
	cdef int i,j
	cdef char* output = <char*>PyMem_Malloc(array.shape[0] * array.shape[1] * sizeof(char) + 1)
	output[array.shape[0] * array.shape[1]] = 0
	for i in range(array.shape[0]):
		for j in range(array.shape[1]):
			output[i*array.shape[0] + j] = <char>array[i, j]+48
	return <bytes>output


def is_row(np.ndarray[DTYPE_t, ndim=1] arr):
	if len(arr) == 5 and (arr == [1, 1, 1, 1, 1]).all():
		return True
	return False


def contains_row(np.ndarray[DTYPE_t, ndim=1]arr):
	cdef int i = 0
	cdef int j

	while i < len(arr) - 4:
		if is_row(arr[i:i+5]):
			return True
		else:
			# compute offset => shift to fist incorrect position
			for j in range(4,-1,-1):
				if arr[j] != 1:
					i += j
			i += 1
	return False


def game_end(np.ndarray[DTYPE_t, ndim=2] board, move):
	cdef int border = 4
	cdef int i, offset

	# shrink area if area around move sticks out of board
	cdef int x_bot = np.clip(move[1] - border, 0, 18)
	cdef int x_top = np.clip(move[1] + border, 0, 18)
	cdef int y_bot = np.clip(move[0] - border, 0, 18)
	cdef int y_top = np.clip(move[0] + border, 0, 18)

	cdef np.ndarray[DTYPE_t, ndim=2] area = board[y_bot:y_top+1, x_bot:x_top+1]

	# compute location of last move inside clipped area
	cdef np.ndarray[DTYPE_t, ndim=1] area_move = np.array([4, 4])

	if move[1] - border < 0:
		area_move[1] += move[1] - border
	if move[0] - border < 0:
		area_move[0] += move[0] - border
	move = area_move

	if contains_row(area[:, area_move[1]]) or contains_row(area[area_move[0], :]):
		return True

	# check diagonals
	for i in range(2):

		# y > x
		offset = area_move[1] - area_move[0]

		# get diagonal
		dg = np.diag(area, offset)

		# check each 5 positions
		if contains_row(dg):
			return True

		area = area.transpose()
		area_move = area_move[::-1]
	return False

def find_n_max_2d(np.ndarray[DTYPE_t, ndim=2] array, int n):
	# Function returns array [[row,col,value],...] containing coordinates and maximal value
	# 	n times
	assert n >= 1

	idx = np.argsort(array, axis=None)[-1:-n-1:-1]
	return [(i / 19, i % 19) for i in idx]
