#import caffe
import time
import numpy as np
import sys
import os
import random
import math
import socket

#board indexes
mine = 0
enemy = 1

#board position values
free = 0.0
occupied = 1.0

def my_turn(board,x,y):
	exp = True
	while exp:
		x,y = random.randint(0,18),random.randint(0,18)

		print("generujem " + str(x) + " " + str(y) + " " + str(board[mine][x][y]))

		if board[mine][x][y] == free and board[enemy][x][y] == free:
			board[mine][x][y] = occupied
			print("nastavujem ["+str(x)+"]["+str(y)+"]")
			exp = False
	return board,x,y


if __name__ == "__main__":
	random.seed(time.clock())
	board = [[[False for x in range(19)] for y in range(19)] for z in range(2)]

	HOST, PORT = "192.168.56.1", 27015

	sock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
	sock.bind((HOST,PORT))
	sock.listen(2)

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
			print(x,y)
			sys.stdout.flush()

			#initialize board
			if x == ord('i') and y == ord('n'):
				board = [[[free for a1 in range(19)] for a2 in range(19)] for a3 in range(2)]

			else:
				#start first
				if x == ord('s') and y == ord('t'):
					board,x,y = my_turn(board,x,y)

				#make enemy move and my move
				else:
					board[enemy][x][y] = occupied
					board,x,y = my_turn(board,x,y)

				output = bytearray([x,y])
				connection.sendall(output)
				
				print( "posielam " + str(x) + " " + str(y))
				print("")

