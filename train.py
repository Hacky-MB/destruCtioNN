#!/usr/bin/env python

import caffe
import time
import numpy as np
import sys
import os
import random
import math
from collections import deque
import arrays as arrs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

current_stone = 1
border = 2

def train_net(solver,board,x,y,player):
	transformations = [lambda x:x,arrs.fliplr2d,arrs.flipud2d]
	move = np.zeros((19,19), dtype = np.int)
	#move = board[player]
	move[x,y] = current_stone

	for t in transformations:#range(0,1):#

		#flip board
		board[0],board[1],move = t(board[0]),t(board[1]),t(move)
		
		for r in range(0,4):#range(0,1):#
			
			#rotate board
			board[0],board[1],move = arrs.rotate_clockwise(board[0]),arrs.rotate_clockwise(board[1]),arrs.rotate_clockwise(move)

			#load boards
			solver.net.blobs['data'].data[0][0][:][:] = board[player][:][:]
			solver.net.blobs['data'].data[0][1][:][:] = board[(player+1)%2][:][:]

			#expected move
			solver.net.blobs['labels'].data[0] = (x-2)*19+(y-2) # substract padding

			solver.step(1)
			#solver.net.forward()

		#flip board back
		board,move = t(board),t(move)
	
	return solver

def test_net(solver,board,x,y,player,n):
	transformations = [lambda x:x,arrs.fliplr2d,arrs.flipud2d]
	move = np.zeros((19,19), dtype = np.int)
	move[x,y] = current_stone

	guess = 0
	n_guess = 0

	for t in range(0,1):#transformations:

		#flip board
		#board[0],board[1],move = t(board[0]),t(board[1]),t(move)
		
		for r in range(0,1):
			
			#rotate board
			#board[0],board[1],move = arrs.rotate_clockwise(board[0]),arrs.rotate_clockwise(board[1]),arrs.rotate_clockwise(move)

			#load boards
			solver.net.blobs['data'].data[0][0][:][:] = board[player][:][:]
			solver.net.blobs['data'].data[0][1][:][:] = board[(player+1)%2][:][:]

			#expected move
			solver.net.blobs['labels'].data[0] = (x-2)*19+(y-2) # substract padding

			solver.net.forward()

			# move is the most probable
			tmp = arrs.find_n_max_2d(solver.net.blobs['conv3'].data[0][0],1)
			if tmp[0][0] == x-2 and tmp[0][1] == y-2:
				guess += 1

			# check if move is in top N probable moves
			tmp = arrs.find_n_max_2d(solver.net.blobs['conv3'].data[0][0],5)
			for k in range(len(tmp)):
				if tmp[k][0] == x-2 and tmp[k][1] == y-2:
					n_guess += 1
					break

		#flip board back
		#board,move = t(board),t(move)
	return guess,n_guess



def train_file(solver,file):
	line_cnt = 0

	boards = np.array([np.zeros((23,23)),np.zeros((23,23))])
	boards[0] = np.pad(np.zeros((19,19)),(2,2),'constant',constant_values=(2,2))
	boards[1] = np.pad(np.zeros((19,19)),(2,2),'constant',constant_values=(2,2))

	#f = open("../games/"+file,'r')
	f = open(file,'r')

	#skip first line
	f.readline()

	#iterate over lines in file
	while True:
		# read coordinates
		line = f.readline()
		coords = line.split(",")

		if len(coords) != 3:
			break

		#indexed by <x,y>, x,y =<0,18> + padding
		x = int(coords[0])+2
		y = int(coords[1])+2

		#train on current board and move
		#solver = train_net(solver,boards,x,y,line_cnt%2)
		solver = train_net(solver,boards,x,y,line_cnt%2)

		# do move on players board
		boards[line_cnt%2][x][y] = 1

		line_cnt += 1
	f.close()

	#print(solver.net.blobs['out'].data[0])
	#print(solver.net.blobs['conv3'].data[0][0])
	#print(solver.net.blobs['loss'].data)
	#print(arrs.find_n_max_2d(solver.net.blobs['conv3'].data[0][0],5))

	#print(solver.net.blobs['loss'].data)
	#print(solver.net.blobs['out'].data)

	return solver

def test_file(solver,file):
	line_cnt = 0

	boards = np.array([np.zeros((23,23)),np.zeros((23,23))])
	boards[0] = np.pad(np.zeros((19,19)),(2,2),'constant',constant_values=(2,2))
	boards[1] = np.pad(np.zeros((19,19)),(2,2),'constant',constant_values=(2,2))

	#f = open("../games/"+file,'r')
	f = open(file,'r')

	#skip first line
	f.readline()

	guess = 0		#direct guess
	n_guess = 0		#move in top N moves
	guess_cnt = 0	#total guesses

	#iterate over lines in file
	while True:
		# read coordinates
		line = f.readline()
		coords = line.split(",")

		if len(coords) != 3:
			break

		#indexed by <x,y>, x,y =<0,18> + padding
		x = int(coords[0])+2
		y = int(coords[1])+2

		#train on current board and move
		#solver = train_net(solver,boards,x,y,line_cnt%2)
		n1,n2 = test_net(solver,boards,x,y,line_cnt%2,5)

		guess += n1
		n_guess += n2
		guess_cnt += 1

		# do move on players board
		boards[line_cnt%2][x][y] = 1

		line_cnt += 1
	f.close()

	return float(guess)/guess_cnt,float(n_guess)/guess_cnt



if __name__ == "__main__":
	caffe.set_mode_cpu()
	solver = caffe.get_solver("net_solver.prototxt")


	#solver.restore("./1x0-35(6)/net_snapshot_iter_9240.solverstate")

	'''
	for layer_name, blob in solver.net.blobs.iteritems():
		print(layer_name + '\t' + str(blob.data.shape))
	try:
		print solver.net.params[ layer_name ][0].data
		print '  Bias:'
		print '  ', solver.net.params[ layer_name ][1].data
		print '  Data:'
		print '  ', solver.net.blobs[ layer_name ].data[0]
	except:
		pass
	print ''
	'''

	#file = "./1x0-35(6)/1x0-35(6).psq"
	file = "./left/left.psq"

	#x  - pocet iteracii
	#y1 - pocet trafeni/pocet iteracii
	#y2 - pocet v top5/pocet iteracii
	
	a1 = [0]
	y1 = [0]
	y2 = [0]

	for x in range(10):
		solver = train_file(solver,file)
		tmp = test_file(solver,file)

		a1.append(x+1)
		y1.append(tmp[0])
		y2.append(tmp[1])

	
	plt.plot(a1, y1, linestyle='-', marker='o', color='red' )
	plt.plot(a1, y2, linestyle='-', marker='o', color='blue' )

	red = mpatches.Patch(color='red', label='move guessed')
	blue = mpatches.Patch(color='blue', label='move in top 5')

	plt.legend(handles=[red,blue], loc=2, borderaxespad=0.)

	plt.ylim((0,1))
	plt.show()

	
	#solver.snapshot()
	sys.exit()	




	# see if there is snapshot stored
	snapshots = [f for f in os.listdir("./") if  (len(f) > 12 and f[-11:] == "solverstate")]

	if len(snapshots):
		solver.restore(snapshots[-1])
		f = open("./last_file",'r')

		#read file where we finished
		last_file = f.readline().strip()
	#else:
	#	solver = caffe.get_solver("net-solver.prototxt")

	# load name of files wih games
	files = os.listdir("../games")

	#store only valid games
	valid_files = deque([f for f in files if (len(f) > 4 and f[-3:] == "psq")])

	try:
		last_file
	except Exception as e:
		pass
	else:
		while True: 
			if valid_files[0] == last_file:
				valid_files.popleft()
				break

			valid_games.popleft()

	file = ""
	try:
		#iterate over stored games
		for file in valid_files:
			
			# generate board for each player			
			solver = train_file(solver,file)

	# save snapshot on keyboard interrupt
	except KeyboardInterrupt:
		solver.snapshot()
		f = open("./last_file",'w')
		f.write(file+"\n");
		f.close()
