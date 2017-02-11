#!/usr/bin/env python

import numpy as np

def fliplr(arr,boardSize):
	#flip move
	if (np.array(arr).shape == (2,)):
		return [arr[0],boardSize-1-arr[1]]
	#flip board
	else:
		res = []
		for r in range(0,len(arr)):
			res.append(arr[r][::-1])
		return res


def flipud(arr,boardSize):
	#flip move
	if (np.array(arr).shape == (2,)):
		return [boardSize-1-arr[0],arr[1]]
	#flip board
	else:
		res = []
		for r in range(len(arr),0,-1):
			res.append(arr[r-1])
		return res


def rotate_clockwise(arr,boardSize):
	#rotate move
	if (np.array(arr).shape == (2,)):
		return [arr[1],boardSize-1-arr[0]]
	#rotate board
	else:
		tmp = zip(*arr[::-1])
		return tmp


def find_n_max_2d(array,n):
	'''
	@return - array [row,col,value] containing coordinates and minimal value
				of size n
	'''
	ret = []

	if n < 1:
		return ret

	for i in range(len(array)):
		for j in range(len(array[i])):

			#initialize ret
			if len(ret) < n:
				ret.append([i,j,array[i][j]])

			else:
				#find minimal value in array
				min = np.min(ret,axis=0)[-1]

				#if value is bigger than minimal value
				if array[i][j] > min:

					#store value in place of minimal value
					for k in range(n):
						if ret[k][2] == min:
							ret[k] = [i,j,array[i][j]]
							break
	return ret






if __name__ == "__main__":
	a = [[1,2,3],[3,4,5]]
	print(a)
	print(find_n_max_2d(a,1))
	print(find_n_max_2d(a,2))
	print(find_n_max_2d(a,3))
	print(find_n_max_2d(a,4))
	print(find_n_max_2d(a,5))


