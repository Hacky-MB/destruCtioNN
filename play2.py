#!/usr/bin/env python

import caffe
import time
import numpy as np
import sys
import os
import random
import math
import socket
import train2

def parseArgs():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('-s', '--solver', required=True)
	parser.add_argument('-l', '--load_snapshot', required=True)
	return parser.parse_args()

def my_turn(solver,board):
	# predhodim solver,dosku,moves([[0,0]]),velkost,

	arr = train2.train_iter(solver,np.array([board]),np.array([[0,0]]),1,"play")

	found = False
	for i in range(5):
		y = arr[i][0]
		x = arr[i][1]

		if board[y,x] == 0:
			found = True
			break

	if not found:
		exp = True
		while exp:
			x,y = random.randint(0,18),random.randint(0,18)

			if board[y][x] == 0:
				exp = False

	board[y][x] = 1
	return board,x,y


def main():
	random.seed(time.clock())
	board = np.array([[0 for x in range(19)] for y in range(19)])
	
	boards = []
	moves = []

	HOST, PORT = "192.168.56.1", 27015

	sock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
	sock.bind((HOST,PORT))
	sock.listen(2)

	args = parseArgs()

	caffe.set_mode_cpu()
	solver = caffe.get_solver(args.solver)

	solver.restore(args.load_snapshot)
	solver.test_nets[0].share_with(solver.net)


	try:
		while True:
			connection,client_address = sock.accept()
			while True:


				data = connection.recv(20).strip()
				print("DATA: " + str(len(data)))

				#if connection was closed
				if not len(data):
					break

				#print recieved data
				for i in range(0,len(data)):
					print ord(data[i]),
				print("")

				index = 6

				#find index of coordinates
				for i in range(0,5):
					if (ord(data[i]) != 255):
						index = i
						break

				if index == 6:
					print("Coordinates not found in recieved packet")
					break

				x,y = ord(data[index]), ord(data[index+1])
				print(y,x)
				sys.stdout.flush()

				#initialize board
				if x == ord('i') and y == ord('n'):
					board = np.array([[0 for x in range(19)] for y in range(19)])
					boards.append([])
					moves.append([])

				else:
					#start first
					if x == ord('s') and y == ord('t'):
						boards[-1].append(np.copy(board).astype(np.uint8))
						board,x,y = my_turn(solver,board)
						moves[-1].append(np.asarray((y,x)))

					#make enemy move and my move
					else:
						board[y][x] = 2
						boards[-1].append(np.copy(board).astype(np.uint8))
						board,x,y = my_turn(solver,board)
						moves[-1].append(np.asarray((y,x)))

					output = bytearray([x,y])
					connection.sendall(output)
					
					print( "posielam x-" + str(x) + " y-" + str(y))
					print("")
	except:
		np.savez("games", boards=boards, moves=moves)


if __name__ == "__main__":
	main()
