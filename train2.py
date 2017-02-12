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

## Function trains net and returns tuple respectively depending on "mode" argument
# @param mode == "train" -> @return solver
# @param mode == "test" -> @return acc(guessed move),acc(move in top 5 most probable),loss
# @param mode == "play" -> @return list(top 5 probable moves)
def train(solver,boards,moves,mode):	
	#load boards
	solver.net.blobs['data'].data[:,:,:,:] = boards[:,:,:,:]

	#load expected move
	if mode != "play":
		for i in range(len(moves)):
			solver.net.blobs['labels'].data[i] = (moves[i][0])*19+(moves[i][1])

	# move is the most probable
	if mode == "train":
		solver.step(1)
		return solver

	elif mode == "test":
		solver.net.forward()
		return solver.net.blobs['accuracy'].data,solver.net.blobs['accuracy5'].data,solver.net.blobs['loss'].data

	elif mode == "play":
		solver.net.forward()
		#return most probable move
		return arrs.find_n_max_2d(solver.net.blobs['conv3'].data[0][0],5)

	else:
		raise Exception("Wrong mode! (use train/test)")

## Function transforms boards and moves (for better training results) and trains net
def transform_train(solver,boards,moves):
	transformations = [lambda x,y:x,arrs.fliplr,arrs.flipud]

	boardSize = 19

	#iterate over transformations
	for t in transformations:#range(0,1):#

		#flip board
		for i in range(len(moves)):
			boards[i,0],boards[i,1],moves[i] = t(boards[i,0],boardSize),t(boards[i,1],boardSize),t(moves[i],boardSize)
		
		#iterate over rotations
		for r in range(0,4):#range(0,1):#
			
			#rotate board
			for i in range(len(moves)):
				boards[i][0],boards[i][1],moves[i] = arrs.rotate_clockwise(boards[i][0],boardSize),arrs.rotate_clockwise(boards[i][1],boardSize),arrs.rotate_clockwise(moves[i],boardSize)

			solver = train(solver,boards,moves,"train")

		#flip board back
		for i in range(len(moves)):
			boards[i,0],boards[i,1],moves[i] = t(boards[i,0],boardSize),t(boards[i,1],boardSize),t(moves[i],boardSize)
	
	return solver


## Function prepares data 
def train_iter(solver,boards,moves,size,mode):
	line_cnt = 0

	# n random moves
	indices = np.random.choice(boards.shape[0], size)

	#n last moves
	#indices = range(len(moves) - size,len(moves))
	
	if mode != "play":
		boards = boards[indices, :, :]
		moves = moves[indices, :]
	else:
		boards = boards[indices[0], :, :]
		moves = moves[indices[0], :]

	#get padding size from net
	border = (solver.net.blobs['data'].data.shape[-1]-19)/2

	boardsOut1 = -np.ones((size, 19+2*border, 19+2*border))
	boardsOut2 = -np.ones((size, 19+2*border, 19+2*border))
	boardsOut1[:,border:-border, border:-border] = boards
	boardsOut2[:,border:-border, border:-border] = boards
	boardsOut1[boardsOut1==2] = 0
	boardsOut2[boardsOut2==1] = 0
	boardsOut2[boardsOut2==2] = 1

	#sample,y,x,player
	boardsOut = np.stack([boardsOut1, boardsOut2], axis=3)

	#sample,player,y,x
	boardsOut = boardsOut.transpose([0,3,1,2])

	#train on current board and move
	if mode == "train":
		#solver 
		return transform_train(solver,boardsOut,moves)
	elif mode == "test":
		#guess,n_guess,loss
		return train(solver,boardsOut,moves,mode)
	elif mode == "play":
		#array of most probable moves
		return train(solver,boardsOut,moves,mode)
	else:
		raise Exception("Wrong mode! (use train/test)")


## Function creates plot from computed values
#it - iteration number
#tst1 - guessed move on test dataset
#tst2 - move in top 5 most probable on test dataset
#tr1 - guessed move on trainging dataset
#tr2 - move in top 5 most probable on trainging dataset
#l - loss layer output
def plot(it,tst1,tst2,tr1,tr2,l):
	
	#number of points on plot
	points=50

	length = len(it)

	#number of values averaged per one point in plot
	samples = length/points  
	if not samples:
		samples = 1

	trash = length%samples

	#cut off values from the end so that length = points * samples
	if trash > 0:
		it= np.array(it[:-trash])
		tst1 = np.array(tst1[:-trash])
		tst2 = np.array(tst2[:-trash])
		l = np.array(l[:-trash])

	#samples * point
	length = len(it)

	(x,y1,y2) = (np.array(it[::length/points]),
				np.mean(tst1.reshape(-1,samples), axis=1),
				np.mean(tst2.reshape(-1,samples), axis=1)) if len(it)>points else (it,tst1,tst2)

	#plt.subplot(1, 2, 1)
	plt.title('Accuracy of test net')
	plt.plot(x, y1, linestyle='-', marker='o', color='red' )
	plt.plot(x, y2, linestyle='-', marker='o', color='blue' )
	red = mpatches.Patch(color='red', label='move guessed')
	blue = mpatches.Patch(color='blue', label='move in top 5')
	plt.legend(handles=[red,blue], loc=2, borderaxespad=0.)
	plt.ylim((0,1))
	plt.xlim((x[0],x[-1]) if len(x) else (0,1))

	#(y1,y2) = (np.mean(tr1.reshape(-1,length/points), axis=1),
	#			np.mean(tst2.reshape(-1,length/points), axis=1) if len(it)>points else (tr1,tr2)

	#plt.subplot(1, 2, 2)
	#plt.title('Accuracy of train net')
	#plt.plot(x, y1, linestyle='-', marker='o', color='red' )
	#plt.plot(x, y2, linestyle='-', marker='o', color='blue' )
	#red = mpatches.Patch(color='red', label='move guessed')
	#blue = mpatches.Patch(color='blue', label='move in top 5')
	#plt.legend(handles=[red,blue], loc=2, borderaxespad=0.)
	#plt.ylim((0,1))
	#plt.xlim((x[0],x[-1]) if len(x) else (0,1))

	#first save, then show (or save figure - fig = plt.gcf();fig.savefig('awd'))
	#new figure is created when show() is called
	#plt.savefig('probs'+str(it[-1] if length else "")+'.svg')

	if args.show_plots:
		plt.show()
	else:
		plt.clf()


	l = np.mean(l.reshape(-1,samples), axis=1) if len(it)>points else l

	plt.plot(x, l, linestyle='-', marker='o', color='red' )
	red = mpatches.Patch(color='red', label='loss')
	plt.legend(handles=[red], loc=2, borderaxespad=0.)
	plt.ylim((0, max(l)+max(l)*0.1 if len(l) else 1))
	#plt.xlim((it[0],it[-1] if len(it) else 0,1))
	#plt.xlim((it[0] if len(it) else 0,it[-1] if len(it) else 1))
	plt.xlim((it[0],it[-1]) if len(it) else (0,1))


	#plt.savefig('loss'+str(it[-1] if len(it) else "")+'.svg')

	if args.show_plots:
		plt.show()
	else:
		plt.clf()


def parseArgs():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('-s', '--solver', required=True)
	parser.add_argument('-i', '--iter_limit', type=int, default=100000)
	parser.add_argument('-d', '--dataset', default="./boards2.npz")
	parser.add_argument('-p', '--show_plots',action='store_true')
	parser.add_argument('-v', '--save_snapshot',action='store_true')
	parser.add_argument('-l', '--load_snapshot', nargs=1)
	return parser.parse_args()

if __name__ == "__main__":
	args = parseArgs()

	caffe.set_mode_cpu()
	solver = caffe.get_solver(args.solver)

	if args.load_snapshot != None:
		solver.restore(args.load_snapshot[0])
		solver.test_nets[0].share_with(solver.net)

	file = np.load(args.dataset)

	boards = file['boards']
	moves = file['moves']

	#it - pocet iteracii
	#tst1,tr1 - pravdepodobnost uhadnutia
	#tst2,tr2 - pravdepodobnost v top 5
	#l	- loss

	it = []
	tst1 = []
	tst2 = []
	tr1 = []
	tr2 = []
	l = []

	batch_size = 32


	try:
		while (not len(it)) or it[-1] < args.iter_limit:
		#for i in range(args.iter_limit):
			#solver = train_iter(solver,boards,moves,batch_size,"train")
			#guess,top_guess,loss = train_iter(solver,boards,moves,batch_size,"test")
			
			# first 2/3 of dataset - training data
			# rest - test data
			solver = train_iter(solver,boards[:len(boards)*2/3],moves[:len(boards)*2/3],batch_size,"train")
			#train_guess,train_n_guess,loss = train_iter(solver,boards[:len(boards)*2/3],moves[:len(boards)*2/3],batch_size,"test")
			test_guess,test_n_guess,loss = train_iter(solver,boards[len(boards)*2/3:],moves[len(boards)*2/3:],batch_size,"test")

			#print train_guess,test_guess
			#print train_n_guess,test_n_guess

			it.append(solver.iter)
			tst1.append(float(test_guess))
			tst2.append(float(test_n_guess))
			#tr1.append(float(train_guess))
			#tr2.append(float(train_n_guess))
			l.append(float(loss))

	except KeyboardInterrupt:
		pass

	if args.save_snapshot != False:
		solver.snapshot()

		np.savez("stats"+str(it[-1]), iter=it, test_n=tst1, test=tst2, #train_n=tr1, train=tr2, 
			loss = l)

	plot(it,tst1,tst2,tr1,tr2,l)


	'''
	# see if there is snapshot stored
	snapshots = [f for f in os.listdir("./") if  (len(f) > 12 and f[-11:] == "solverstate")]

	if len(snapshots):
		solver.restore(snapshots[-1])
	'''
